"""OAuth 2.0 token storage and refresh.

Tokens are stored in the OS keychain when one is available (macOS Keychain,
Windows Credential Locker, Freedesktop Secret Service) and fall back to a
``0600`` file under the config directory otherwise. Refresh happens lazily and
asynchronously: a token is renewed on first use after it goes stale, so the
server never blocks at startup and never needs a scheduled refresh job.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import httpx

TOKEN_URI = "https://oauth2.googleapis.com/token"
KEYRING_SERVICE = "gmail-mcp"

# Refresh this many seconds before the token actually expires, so a call that
# takes a moment to reach Google doesn't arrive with a just-expired token.
EXPIRY_SKEW_SECONDS = 120


class AuthError(RuntimeError):
    """Raised when an account is not authenticated or refresh fails."""


@dataclass(frozen=True)
class Token:
    """A stored OAuth token for one account."""

    access_token: str
    refresh_token: str
    expires_at: float
    scopes: tuple[str, ...] = ()

    def is_expired(self, now: float | None = None) -> bool:
        current = time.time() if now is None else now
        return current >= (self.expires_at - EXPIRY_SKEW_SECONDS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "scopes": list(self.scopes),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Token":
        return cls(
            access_token=str(raw.get("access_token", "")),
            refresh_token=str(raw.get("refresh_token", "")),
            expires_at=float(raw.get("expires_at", 0.0)),
            scopes=tuple(raw.get("scopes", ())),
        )


class TokenStore(Protocol):
    """Storage backend for per-account tokens."""

    def load(self, account: str) -> Token | None: ...
    def save(self, account: str, token: Token) -> None: ...
    def delete(self, account: str) -> None: ...


class KeyringTokenStore:
    """Stores tokens in the OS keychain via the ``keyring`` package."""

    def __init__(self, service: str = KEYRING_SERVICE) -> None:
        import keyring  # imported lazily so the file store works without it

        self._keyring = keyring
        self._service = service

    def load(self, account: str) -> Token | None:
        raw = self._keyring.get_password(self._service, account)
        if not raw:
            return None
        return Token.from_dict(json.loads(raw))

    def save(self, account: str, token: Token) -> None:
        self._keyring.set_password(
            self._service, account, json.dumps(token.to_dict())
        )

    def delete(self, account: str) -> None:
        try:
            self._keyring.delete_password(self._service, account)
        except Exception:  # noqa: BLE001 - absent entry is not an error here
            pass


class FileTokenStore:
    """Stores tokens in a ``0600`` JSON file. Fallback for headless machines."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path).expanduser()

    def _read_all(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _write_all(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Write via a private temp file so the token is never briefly readable.
        tmp = self._path.with_suffix(".tmp")
        tmp.touch(mode=0o600, exist_ok=True)
        tmp.chmod(0o600)
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self._path)
        self._path.chmod(0o600)

    def load(self, account: str) -> Token | None:
        raw = self._read_all().get(account)
        return Token.from_dict(raw) if raw else None

    def save(self, account: str, token: Token) -> None:
        data = self._read_all()
        data[account] = token.to_dict()
        self._write_all(data)

    def delete(self, account: str) -> None:
        data = self._read_all()
        if data.pop(account, None) is not None:
            self._write_all(data)


def default_token_store(config_dir: Path) -> TokenStore:
    """Return the keychain store if usable, else the file store.

    ``keyring`` installs a null backend when no OS keychain is reachable, so we
    probe it with a real round trip rather than trusting the import to succeed.
    """
    try:
        store = KeyringTokenStore()
        probe = Token("probe", "probe", 0.0)
        store.save("__gmail_mcp_probe__", probe)
        ok = store.load("__gmail_mcp_probe__") is not None
        store.delete("__gmail_mcp_probe__")
        if ok:
            return store
    except Exception:  # noqa: BLE001 - any keyring failure means fall back
        pass
    return FileTokenStore(config_dir / "tokens.json")


@dataclass(frozen=True)
class OAuthClient:
    """The Google Cloud OAuth client shared by every account."""

    client_id: str
    client_secret: str

    @classmethod
    def from_secrets_file(cls, path: Path) -> "OAuthClient":
        """Load credentials from a Google Cloud ``client_secret.json``.

        Accepts both the ``installed`` (Desktop app) and ``web`` shapes, since
        the Cloud console hands out either depending on how the client was
        created.
        """
        path = Path(path).expanduser()
        if not path.exists():
            raise AuthError(
                f"OAuth client secrets not found at {path}. Download the "
                "Desktop-app credentials from Google Cloud Console "
                "(APIs & Services > Credentials) and save them there."
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        block = raw.get("installed") or raw.get("web")
        if not block:
            raise AuthError(
                f"{path} does not look like a Google OAuth client file "
                "(expected an 'installed' or 'web' key)."
            )
        return cls(
            client_id=str(block["client_id"]),
            client_secret=str(block["client_secret"]),
        )


class TokenManager:
    """Hands out fresh access tokens, refreshing them as needed.

    A per-account lock ensures that concurrent tool calls on the same account
    trigger at most one refresh; the losers of the race await the winner and
    reuse its result.
    """

    def __init__(
        self,
        client: OAuthClient,
        store: TokenStore,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = client
        self._store = store
        self._http = http
        self._locks: dict[str, asyncio.Lock] = {}
        self._cache: dict[str, Token] = {}

    def _lock_for(self, account: str) -> asyncio.Lock:
        if account not in self._locks:
            self._locks[account] = asyncio.Lock()
        return self._locks[account]

    def authenticated_accounts(self, names: list[str]) -> list[str]:
        """Return the subset of ``names`` that have a stored token."""
        return [n for n in names if self._store.load(n) is not None]

    async def access_token(self, account: str) -> str:
        """Return a valid access token for ``account``, refreshing if stale.

        Raises:
            AuthError: If the account was never authenticated, or if Google
                rejects the refresh token (typically because access was revoked
                or the password changed).
        """
        async with self._lock_for(account):
            token = self._cache.get(account) or self._store.load(account)
            if token is None:
                raise AuthError(
                    f"Account {account!r} is not authenticated. Run "
                    f"'gmail-mcp-setup login {account}' and complete the "
                    "browser sign-in."
                )
            if not token.is_expired():
                self._cache[account] = token
                return token.access_token

            refreshed = await self._refresh(account, token)
            self._store.save(account, refreshed)
            self._cache[account] = refreshed
            return refreshed.access_token

    async def _refresh(self, account: str, token: Token) -> Token:
        if not token.refresh_token:
            raise AuthError(
                f"Account {account!r} has no refresh token stored. Re-run "
                f"'gmail-mcp-setup login {account}'."
            )
        payload = {
            "client_id": self._client.client_id,
            "client_secret": self._client.client_secret,
            "refresh_token": token.refresh_token,
            "grant_type": "refresh_token",
        }
        try:
            if self._http is not None:
                response = await self._http.post(TOKEN_URI, data=payload)
            else:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(TOKEN_URI, data=payload)
        except httpx.HTTPError as exc:
            raise AuthError(
                f"Could not reach Google to refresh {account!r}: {exc}"
            ) from exc

        if response.status_code != 200:
            raise AuthError(
                f"Refresh failed for {account!r} (HTTP {response.status_code}). "
                "The token was probably revoked; re-run "
                f"'gmail-mcp-setup login {account}'."
            )

        data = response.json()
        return replace(
            token,
            access_token=str(data["access_token"]),
            expires_at=time.time() + float(data.get("expires_in", 3600)),
            # Google omits refresh_token on refresh responses; keep the old one.
            refresh_token=str(data.get("refresh_token") or token.refresh_token),
        )
