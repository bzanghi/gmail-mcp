"""End-to-end tests of the MCP tools against the fake Gmail API.

These go through ``mcp.call_tool``, so the registered JSON schemas and Pydantic
validation are exercised the same way Claude Desktop would exercise them.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from gmail_mcp import server
from gmail_mcp.config import Config
from gmail_mcp.gmail_client import GmailClient

from .conftest import FakeGmail


@pytest.fixture(autouse=True)
def wire_state(monkeypatch: pytest.MonkeyPatch, config: Config, gmail_client: GmailClient):
    """Point the server's lazy state at the fake Gmail API."""

    async def fake_get() -> tuple[Config, GmailClient]:
        return config, gmail_client

    monkeypatch.setattr(server._state, "get", fake_get)
    yield


async def call(name: str, **arguments: Any) -> str:
    """Invoke a tool by name and return its text output.

    Schema-validation failures surface as ``ToolError`` rather than a return
    value; both reach the model as an error, so they are normalised to the same
    "Error: ..." string here.
    """
    from mcp.server.fastmcp.exceptions import ToolError

    try:
        result = await server.mcp.call_tool(name, {"params": arguments})
    except ToolError as exc:
        return f"Error: {exc}"
    content = result[0] if isinstance(result, tuple) else result
    blocks = content if isinstance(content, list) else [content]
    return "\n".join(getattr(b, "text", str(b)) for b in blocks)


class TestListAccounts:
    async def test_lists_configured_aliases(self) -> None:
        data = json.loads(await call("gmail_list_accounts"))
        assert [a["name"] for a in data["accounts"]] == ["personal", "work-main"]

    async def test_reports_authentication_state(self) -> None:
        data = json.loads(await call("gmail_list_accounts"))
        assert all(a["authenticated"] for a in data["accounts"])

    async def test_includes_description_for_account_choice(self) -> None:
        data = json.loads(await call("gmail_list_accounts"))
        assert data["accounts"][0]["description"] == "Personal"


class TestSearch:
    async def test_returns_messages(self) -> None:
        output = await call("gmail_search_messages", account="personal", query="")
        assert "Q3 numbers" in output

    async def test_json_format_is_structured(self) -> None:
        data = json.loads(
            await call(
                "gmail_search_messages",
                account="personal",
                query="",
                response_format="json",
            )
        )
        assert data["count"] >= 1
        assert data["messages"][0]["account"] == "personal"

    async def test_omits_bodies_to_save_context(self) -> None:
        data = json.loads(
            await call(
                "gmail_search_messages", account="personal", response_format="json"
            )
        )
        assert "body" not in data["messages"][0]

    async def test_exposes_next_page_token(self) -> None:
        data = json.loads(
            await call(
                "gmail_search_messages",
                account="personal",
                limit=1,
                response_format="json",
            )
        )
        assert data["next_page_token"] == "page-2"

    async def test_unknown_account_returns_actionable_error(self) -> None:
        output = await call("gmail_search_messages", account="nonexistent")
        assert output.startswith("Error:")
        assert "personal, work-main" in output

    async def test_limit_above_maximum_is_rejected(self) -> None:
        output = await call("gmail_search_messages", account="personal", limit=500)
        assert "Error" in output or "less than or equal to 100" in output

    async def test_api_failure_is_reported_not_raised(
        self, fake_gmail: FakeGmail
    ) -> None:
        fake_gmail.fail_with["/messages"] = 500
        output = await call("gmail_search_messages", account="personal")
        assert output.startswith("Error:")


class TestReadMessage:
    async def test_returns_body(self) -> None:
        output = await call(
            "gmail_read_message", account="personal", message_id="msg-plain"
        )
        assert "beat plan by 12%" in output

    async def test_strips_quoted_history(self) -> None:
        output = await call(
            "gmail_read_message", account="personal", message_id="msg-plain"
        )
        assert "When will the numbers land?" not in output

    async def test_body_budget_truncates(self) -> None:
        data = json.loads(
            await call(
                "gmail_read_message",
                account="personal",
                message_id="msg-plain",
                body_chars=20,
                response_format="json",
            )
        )
        assert data["messages"][0]["body_truncated"] is True

    async def test_html_only_message_is_readable(self) -> None:
        output = await call(
            "gmail_read_message", account="personal", message_id="msg-html"
        )
        assert "Balance: $1,204.55" in output
        assert "<p>" not in output

    async def test_missing_message_explains(self) -> None:
        output = await call(
            "gmail_read_message", account="personal", message_id="no-such-id"
        )
        assert output.startswith("Error:")
        assert "Not found" in output


class TestReadThread:
    async def test_returns_every_message(self) -> None:
        data = json.loads(
            await call(
                "gmail_read_thread",
                account="personal",
                thread_id="thread-1",
                response_format="json",
            )
        )
        assert data["count"] == 2

    async def test_includes_bodies(self) -> None:
        output = await call(
            "gmail_read_thread", account="personal", thread_id="thread-1"
        )
        assert "Due in 30 days" in output


class TestSend:
    async def test_sends_and_reports_id(self, fake_gmail: FakeGmail) -> None:
        data = json.loads(
            await call(
                "gmail_send_message",
                account="personal",
                to=["bob@example.com"],
                subject="Hello",
                body="Hi Bob",
            )
        )
        assert data["status"] == "sent"
        assert data["message_id"] == "sent-1"
        assert len(fake_gmail.sent) == 1

    async def test_uses_the_named_account_token(self, fake_gmail: FakeGmail) -> None:
        await call(
            "gmail_send_message",
            account="work-main",
            to=["bob@example.com"],
            body="Hi",
        )
        send_request = [r for r in fake_gmail.requests if r.url.path.endswith("/send")][-1]
        assert send_request.headers["Authorization"] == "Bearer tok-work"

    async def test_invalid_address_blocked_before_sending(
        self, fake_gmail: FakeGmail
    ) -> None:
        output = await call(
            "gmail_send_message", account="personal", to=["not-an-email"], body="Hi"
        )
        assert "Error" in output or "valid email" in output
        assert fake_gmail.sent == []

    async def test_reply_threads_and_derives_subject(
        self, fake_gmail: FakeGmail
    ) -> None:
        await call(
            "gmail_send_message",
            account="personal",
            to=["alice@example.com"],
            body="Thanks!",
            reply_to_message_id="msg-plain",
        )
        sent = fake_gmail.sent[-1]
        assert sent["threadId"] == "thread-1"
        import base64

        raw = base64.urlsafe_b64decode(sent["raw"] + "==").decode()
        assert "Subject: Re: Q3 numbers" in raw
        assert "In-Reply-To: <abc123@example.com>" in raw

    async def test_reply_without_recipient_uses_original_sender(
        self, fake_gmail: FakeGmail
    ) -> None:
        await call(
            "gmail_send_message",
            account="personal",
            to=[],
            body="Thanks!",
            reply_to_message_id="msg-plain",
        )
        import base64

        raw = base64.urlsafe_b64decode(fake_gmail.sent[-1]["raw"] + "==").decode()
        assert "alice@example.com" in raw

    async def test_empty_body_rejected(self, fake_gmail: FakeGmail) -> None:
        output = await call(
            "gmail_send_message", account="personal", to=["a@b.com"], body=""
        )
        assert "Error" in output or "at least 1 character" in output
        assert fake_gmail.sent == []


class TestDraft:
    async def test_creates_draft_without_sending(self, fake_gmail: FakeGmail) -> None:
        data = json.loads(
            await call(
                "gmail_create_draft",
                account="personal",
                to=["bob@example.com"],
                subject="Later",
                body="Draft body",
            )
        )
        assert data["status"] == "draft_created"
        assert data["draft_id"] == "draft-1"
        assert fake_gmail.sent == [], "creating a draft must not send anything"


class TestLabels:
    async def test_lists_labels_with_ids(self) -> None:
        data = json.loads(await call("gmail_list_labels", account="personal"))
        assert {"id": "Label_12", "name": "Receipts", "type": "user"} in data["labels"]

    async def test_marks_read_by_removing_unread(self, fake_gmail: FakeGmail) -> None:
        await call(
            "gmail_modify_labels",
            account="personal",
            message_id="msg-plain",
            remove_labels=["UNREAD"],
        )
        assert fake_gmail.modifications[-1]["removeLabelIds"] == ["UNREAD"]

    async def test_no_labels_specified_is_rejected(
        self, fake_gmail: FakeGmail
    ) -> None:
        output = await call(
            "gmail_modify_labels", account="personal", message_id="msg-plain"
        )
        assert output.startswith("Error:")
        assert fake_gmail.modifications == []


class TestCheckInboxes:
    async def test_covers_all_accounts_by_default(self) -> None:
        data = json.loads(await call("gmail_check_inboxes", response_format="json"))
        assert data["checked_at_accounts"] == ["personal", "work-main"]

    async def test_counts_unread_across_accounts(self) -> None:
        data = json.loads(await call("gmail_check_inboxes", response_format="json"))
        # Two UNREAD messages in the fake, per account, across two accounts.
        assert data["total_matching"] == 4

    async def test_flags_important_and_starred(self) -> None:
        data = json.loads(await call("gmail_check_inboxes", response_format="json"))
        assert data["flagged_count"] >= 1

    async def test_returns_no_bodies(self) -> None:
        data = json.loads(await call("gmail_check_inboxes", response_format="json"))
        preview = data["accounts"][0]["preview"][0]
        assert "body" not in preview

    async def test_subset_of_accounts(self) -> None:
        data = json.loads(
            await call(
                "gmail_check_inboxes", accounts=["personal"], response_format="json"
            )
        )
        assert data["checked_at_accounts"] == ["personal"]

    async def test_markdown_summary_is_compact(self) -> None:
        output = await call("gmail_check_inboxes")
        assert "# Inbox check" in output
        assert len(output) < 4000, "scheduled output must stay small"

    async def test_one_failing_account_does_not_sink_the_sweep(
        self, fake_gmail: FakeGmail, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = GmailClient.list_message_ids

        async def flaky(self, account, *args, **kwargs):  # type: ignore[no-untyped-def]
            if account == "personal":
                raise RuntimeError("simulated outage")
            return await original(self, account, *args, **kwargs)

        monkeypatch.setattr(GmailClient, "list_message_ids", flaky)
        data = json.loads(await call("gmail_check_inboxes", response_format="json"))
        by_account = {a["account"]: a for a in data["accounts"]}
        assert by_account["personal"]["error"] is not None
        assert by_account["work-main"]["error"] is None
        assert by_account["work-main"]["matching"] > 0

    async def test_unknown_account_reports_valid_ones(self) -> None:
        output = await call("gmail_check_inboxes", accounts=["ghost"])
        assert output.startswith("Error:")
        assert "personal, work-main" in output


class TestToolRegistration:
    async def test_all_nine_tools_registered(self) -> None:
        names = {t.name for t in await server.mcp.list_tools()}
        assert names == {
            "gmail_list_accounts",
            "gmail_search_messages",
            "gmail_read_message",
            "gmail_read_thread",
            "gmail_send_message",
            "gmail_create_draft",
            "gmail_modify_labels",
            "gmail_list_labels",
            "gmail_check_inboxes",
        }

    async def test_read_tools_are_marked_read_only(self) -> None:
        tools = {t.name: t for t in await server.mcp.list_tools()}
        for name in ("gmail_search_messages", "gmail_read_message", "gmail_check_inboxes"):
            assert tools[name].annotations.readOnlyHint is True

    async def test_send_is_not_marked_read_only(self) -> None:
        tools = {t.name: t for t in await server.mcp.list_tools()}
        assert tools["gmail_send_message"].annotations.readOnlyHint is False

    async def test_every_tool_has_a_description(self) -> None:
        for tool in await server.mcp.list_tools():
            assert tool.description and len(tool.description) > 80
