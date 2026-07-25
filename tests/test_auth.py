"""Tests for token storage, expiry, and refresh."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest

from gmail_mcp.auth import (
    AuthError,
    FileTokenStore,
    OAuthClient,
    Token,
    TokenManager,
)

from .conftest import FakeGmail, MemoryTokenStore


class TestToken:
    def test_fresh_token_not_expired(self) -> None:
        assert not Token("a", "r", time.time() + 3600).is_expired()

    def test_past_expiry_is_expired(self) -> None:
        assert Token("a", "r", time.time() - 1).is_expired()

    def test_expires_within_skew_counts_as_expired(self) -> None:
        # 60s out is inside the 120s safety margin, so it should refresh early
        # rather than risk arriving at Google already stale.
        assert Token("a", "r", time.time() + 60).is_expired()

    def test_round_trips_through_dict(self) -> None:
        token = Token("acc", "ref", 123.0, ("scope-a",))
        assert Token.from_dict(token.to_dict()) == token


class TestFileTokenStore:
    def test_save_and_load(self, tmp_path: Path) -> None:
        store = FileTokenStore(tmp_path / "tokens.json")
        token = Token("a", "r", 100.0)
        store.save("personal", token)
        assert store.load("personal") == token

    def test_missing_account_returns_none(self, tmp_path: Path) -> None:
        assert FileTokenStore(tmp_path / "tokens.json").load("nope") is None

    def test_file_is_owner_only(self, tmp_path: Path) -> None:
        path = tmp_path / "tokens.json"
        FileTokenStore(path).save("personal", Token("a", "r", 1.0))
        assert path.stat().st_mode & 0o077 == 0, "tokens must not be group/world readable"

    def test_accounts_are_independent(self, tmp_path: Path) -> None:
        store = FileTokenStore(tmp_path / "tokens.json")
        store.save("a", Token("ta", "ra", 1.0))
        store.save("b", Token("tb", "rb", 2.0))
        store.delete("a")
        assert store.load("a") is None
        assert store.load("b") is not None

    def test_corrupt_file_does_not_raise(self, tmp_path: Path) -> None:
        path = tmp_path / "tokens.json"
        path.write_text("{ not json", encoding="utf-8")
        assert FileTokenStore(path).load("personal") is None

    def test_no_leftover_temp_file(self, tmp_path: Path) -> None:
        path = tmp_path / "tokens.json"
        FileTokenStore(path).save("personal", Token("a", "r", 1.0))
        assert not (tmp_path / "tokens.tmp").exists()


class TestTokenManager:
    @pytest.fixture
    def manager_parts(self, fake_gmail: FakeGmail):
        http = httpx.AsyncClient(transport=httpx.MockTransport(fake_gmail.handler))
        store = MemoryTokenStore()
        manager = TokenManager(OAuthClient("cid", "csecret"), store, http=http)
        return manager, store, fake_gmail

    async def test_returns_stored_token_when_fresh(self, manager_parts) -> None:
        manager, store, fake = manager_parts
        store.save("personal", Token("still-good", "r", time.time() + 3600))
        assert await manager.access_token("personal") == "still-good"
        assert fake.refresh_count == 0

    async def test_refreshes_expired_token(self, manager_parts) -> None:
        manager, store, fake = manager_parts
        store.save("personal", Token("stale", "refresh-me", time.time() - 10))
        assert await manager.access_token("personal") == "fresh-1"
        assert fake.refresh_count == 1

    async def test_refreshed_token_is_persisted(self, manager_parts) -> None:
        manager, store, _ = manager_parts
        store.save("personal", Token("stale", "refresh-me", time.time() - 10))
        await manager.access_token("personal")
        stored = store.load("personal")
        assert stored is not None and stored.access_token == "fresh-1"

    async def test_refresh_token_survives_refresh(self, manager_parts) -> None:
        # Google omits refresh_token in refresh responses; losing it would mean
        # the account silently stops working at the next expiry.
        manager, store, _ = manager_parts
        store.save("personal", Token("stale", "keep-me", time.time() - 10))
        await manager.access_token("personal")
        stored = store.load("personal")
        assert stored is not None and stored.refresh_token == "keep-me"

    async def test_concurrent_calls_refresh_once(self, manager_parts) -> None:
        manager, store, fake = manager_parts
        store.save("personal", Token("stale", "refresh-me", time.time() - 10))
        results = await asyncio.gather(
            *(manager.access_token("personal") for _ in range(8))
        )
        assert fake.refresh_count == 1, "per-account lock should collapse the stampede"
        assert len(set(results)) == 1

    async def test_unauthenticated_account_raises_actionable_error(
        self, manager_parts
    ) -> None:
        manager, _, _ = manager_parts
        with pytest.raises(AuthError, match="gmail-mcp-setup login"):
            await manager.access_token("never-logged-in")

    async def test_missing_refresh_token_raises(self, manager_parts) -> None:
        manager, store, _ = manager_parts
        store.save("personal", Token("stale", "", time.time() - 10))
        with pytest.raises(AuthError, match="no refresh token"):
            await manager.access_token("personal")

    async def test_rejected_refresh_raises_actionable_error(
        self, manager_parts
    ) -> None:
        manager, store, fake = manager_parts
        fake.fail_with["/token"] = 400
        store.save("personal", Token("stale", "revoked", time.time() - 10))
        with pytest.raises(AuthError, match="revoked"):
            await manager.access_token("personal")

    def test_authenticated_accounts_filters(self, manager_parts) -> None:
        manager, store, _ = manager_parts
        store.save("personal", Token("a", "r", time.time() + 60))
        assert manager.authenticated_accounts(["personal", "work-main"]) == ["personal"]


class TestOAuthClient:
    def test_reads_installed_shape(self, tmp_path: Path) -> None:
        path = tmp_path / "cs.json"
        path.write_text(
            json.dumps({"installed": {"client_id": "a", "client_secret": "b"}})
        )
        assert OAuthClient.from_secrets_file(path).client_id == "a"

    def test_reads_web_shape(self, tmp_path: Path) -> None:
        path = tmp_path / "cs.json"
        path.write_text(json.dumps({"web": {"client_id": "w", "client_secret": "s"}}))
        assert OAuthClient.from_secrets_file(path).client_id == "w"

    def test_missing_file_names_the_path(self, tmp_path: Path) -> None:
        with pytest.raises(AuthError, match="absent.json"):
            OAuthClient.from_secrets_file(tmp_path / "absent.json")

    def test_wrong_shape_is_explained(self, tmp_path: Path) -> None:
        path = tmp_path / "cs.json"
        path.write_text(json.dumps({"something": "else"}))
        with pytest.raises(AuthError, match="installed"):
            OAuthClient.from_secrets_file(path)
