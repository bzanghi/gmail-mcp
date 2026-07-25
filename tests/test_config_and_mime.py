"""Tests for config parsing and RFC 2822 composition."""

from __future__ import annotations

import base64
import json
from email import message_from_bytes
from pathlib import Path

import pytest

from gmail_mcp.config import Config, ConfigError, load_config
from gmail_mcp.mime import build_message, reply_metadata

from .conftest import PLAIN_MESSAGE


def decode_raw(raw: str):
    """Decode a base64url message back into an email object for assertions."""
    return message_from_bytes(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))


class TestConfig:
    def test_loads_accounts(self, config: Config) -> None:
        assert config.account_names() == ["personal", "work-main"]

    def test_get_by_alias(self, config: Config) -> None:
        assert config.get("work-main").email == "me@corp.com"

    def test_unknown_alias_lists_valid_ones(self, config: Config) -> None:
        with pytest.raises(ConfigError, match="personal, work-main"):
            config.get("nope")

    def test_resolve_empty_means_all(self, config: Config) -> None:
        assert len(config.resolve([])) == 2

    def test_resolve_subset(self, config: Config) -> None:
        assert [a.name for a in config.resolve(["personal"])] == ["personal"]

    def test_relative_secrets_path_resolved_against_config_dir(
        self, config: Config, config_file: Path
    ) -> None:
        assert config.client_secrets_file == config_file.parent / "client_secret.json"

    def test_missing_file_names_the_path(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="No config found"):
            load_config(tmp_path / "absent.json")

    def test_invalid_json_is_explained(self, tmp_path: Path) -> None:
        path = tmp_path / "accounts.json"
        path.write_text("{ nope", encoding="utf-8")
        with pytest.raises(ConfigError, match="not valid JSON"):
            load_config(path)

    def test_empty_accounts_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "accounts.json"
        path.write_text(json.dumps({"accounts": []}), encoding="utf-8")
        with pytest.raises(ConfigError, match="non-empty"):
            load_config(path)

    def test_duplicate_alias_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "accounts.json"
        path.write_text(
            json.dumps({"accounts": [{"name": "a"}, {"name": "a"}]}), encoding="utf-8"
        )
        with pytest.raises(ConfigError, match="Duplicate"):
            load_config(path)

    def test_account_without_name_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "accounts.json"
        path.write_text(json.dumps({"accounts": [{"email": "x@y.z"}]}), encoding="utf-8")
        with pytest.raises(ConfigError, match="needs a 'name'"):
            load_config(path)

    def test_five_accounts_supported(self, tmp_path: Path) -> None:
        """The client's actual case: one personal plus four business accounts."""
        path = tmp_path / "accounts.json"
        names = ["personal", "work-main", "work-sales", "work-billing", "work-support"]
        path.write_text(
            json.dumps({"accounts": [{"name": n} for n in names]}), encoding="utf-8"
        )
        assert load_config(path).account_names() == names


class TestBuildMessage:
    def test_basic_headers(self) -> None:
        raw = build_message(["a@b.com"], "Hi", "Body text")
        message = decode_raw(raw)
        assert message["To"] == "a@b.com"
        assert message["Subject"] == "Hi"
        assert "Body text" in message.get_payload(decode=True).decode()

    def test_multiple_recipients_joined(self) -> None:
        message = decode_raw(build_message(["a@b.com", "c@d.com"], "s", "b"))
        assert message["To"] == "a@b.com, c@d.com"

    def test_cc_and_bcc(self) -> None:
        message = decode_raw(
            build_message(["a@b.com"], "s", "b", cc=["c@c.com"], bcc=["d@d.com"])
        )
        assert message["Cc"] == "c@c.com"
        assert message["Bcc"] == "d@d.com"

    def test_no_recipients_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one recipient"):
            build_message([], "s", "b")

    def test_reply_headers_applied(self) -> None:
        message = decode_raw(
            build_message(
                ["a@b.com"],
                "Re: x",
                "b",
                reply_headers={"In-Reply-To": "<id@x>", "References": "<id@x>"},
            )
        )
        assert message["In-Reply-To"] == "<id@x>"
        assert message["References"] == "<id@x>"

    def test_empty_reply_headers_omitted(self) -> None:
        message = decode_raw(
            build_message(["a@b.com"], "s", "b", reply_headers={"In-Reply-To": ""})
        )
        assert message["In-Reply-To"] is None

    def test_html_alternative_makes_multipart(self) -> None:
        message = decode_raw(
            build_message(["a@b.com"], "s", "plain", html_body="<p>rich</p>")
        )
        assert message.is_multipart()

    def test_unicode_subject_and_body_survive(self) -> None:
        message = decode_raw(build_message(["a@b.com"], "Grüße 😀", "naïve café — ok"))
        assert "Grüße" in str(message["Subject"]) or "=?utf-8?" in str(message["Subject"])
        assert "naïve café" in message.get_payload(decode=True).decode("utf-8")

    def test_output_is_urlsafe_base64(self) -> None:
        raw = build_message(["a@b.com"], "s", "b")
        assert "+" not in raw and "/" not in raw


class TestReplyMetadata:
    def test_prefixes_subject_once(self) -> None:
        assert reply_metadata(PLAIN_MESSAGE)["subject"] == "Re: Q3 numbers"

    def test_does_not_double_prefix(self) -> None:
        original = json.loads(json.dumps(PLAIN_MESSAGE))
        original["payload"]["headers"].append({"name": "Subject", "value": "Re: Q3"})
        original["payload"]["headers"] = [
            h for h in original["payload"]["headers"] if h["name"] != "Subject"
        ] + [{"name": "Subject", "value": "Re: Q3 numbers"}]
        assert reply_metadata(original)["subject"] == "Re: Q3 numbers"

    def test_carries_thread_id(self) -> None:
        assert reply_metadata(PLAIN_MESSAGE)["thread_id"] == "thread-1"

    def test_in_reply_to_is_message_id(self) -> None:
        assert reply_metadata(PLAIN_MESSAGE)["in_reply_to"] == "<abc123@example.com>"

    def test_reply_to_defaults_to_from(self) -> None:
        assert reply_metadata(PLAIN_MESSAGE)["reply_to"] == "Alice <alice@example.com>"

    def test_references_accumulate_chain(self) -> None:
        original = json.loads(json.dumps(PLAIN_MESSAGE))
        original["payload"]["headers"].append(
            {"name": "References", "value": "<older@example.com>"}
        )
        references = reply_metadata(original)["references"]
        assert "<older@example.com>" in references
        assert "<abc123@example.com>" in references

    def test_missing_headers_produce_safe_defaults(self) -> None:
        meta = reply_metadata({"id": "x", "payload": {}})
        assert meta["subject"] == "(no subject)"
        assert meta["in_reply_to"] == ""
