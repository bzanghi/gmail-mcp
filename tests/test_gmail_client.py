"""Tests for the Gmail REST client against the mock transport."""

from __future__ import annotations

import pytest

from gmail_mcp.gmail_client import GmailApiError, GmailClient

from .conftest import FakeGmail


class TestListing:
    async def test_returns_ids(self, gmail_client: GmailClient) -> None:
        ids, _ = await gmail_client.list_message_ids("personal", limit=10)
        assert "msg-plain" in ids

    async def test_query_is_forwarded(
        self, gmail_client: GmailClient, fake_gmail: FakeGmail
    ) -> None:
        await gmail_client.list_message_ids("personal", query="is:unread", limit=5)
        assert fake_gmail.requests[-1].url.params["q"] == "is:unread"

    async def test_unread_query_filters(self, gmail_client: GmailClient) -> None:
        ids, _ = await gmail_client.list_message_ids("personal", query="is:unread")
        assert "msg-multipart" not in ids  # the only message without UNREAD

    async def test_next_page_token_surfaced(self, gmail_client: GmailClient) -> None:
        _, token = await gmail_client.list_message_ids("personal", limit=1)
        assert token == "page-2"

    async def test_no_token_when_all_results_fit(
        self, gmail_client: GmailClient
    ) -> None:
        _, token = await gmail_client.list_message_ids("personal", limit=50)
        assert token is None

    async def test_page_token_forwarded(
        self, gmail_client: GmailClient, fake_gmail: FakeGmail
    ) -> None:
        await gmail_client.list_message_ids("personal", page_token="page-2")
        assert fake_gmail.requests[-1].url.params["pageToken"] == "page-2"


class TestFetching:
    async def test_metadata_format_requests_only_needed_headers(
        self, gmail_client: GmailClient, fake_gmail: FakeGmail
    ) -> None:
        await gmail_client.get_message("personal", "msg-plain", fmt="metadata")
        params = fake_gmail.requests[-1].url.params
        assert params["format"] == "metadata"
        assert "Subject" in fake_gmail.requests[-1].url.query.decode()

    async def test_full_format_has_no_header_filter(
        self, gmail_client: GmailClient, fake_gmail: FakeGmail
    ) -> None:
        await gmail_client.get_message("personal", "msg-plain", fmt="full")
        assert "metadataHeaders" not in fake_gmail.requests[-1].url.query.decode()

    async def test_batch_preserves_order(self, gmail_client: GmailClient) -> None:
        ids = ["msg-html", "msg-plain", "msg-multipart"]
        results = await gmail_client.get_messages("personal", ids)
        assert [r["id"] for r in results] == ids

    async def test_empty_batch_makes_no_requests(
        self, gmail_client: GmailClient, fake_gmail: FakeGmail
    ) -> None:
        before = len(fake_gmail.requests)
        assert await gmail_client.get_messages("personal", []) == []
        assert len(fake_gmail.requests) == before

    async def test_batch_failure_propagates(self, gmail_client: GmailClient) -> None:
        with pytest.raises(GmailApiError):
            await gmail_client.get_messages("personal", ["msg-plain", "does-not-exist"])

    async def test_thread_returns_all_messages(
        self, gmail_client: GmailClient
    ) -> None:
        thread = await gmail_client.get_thread("personal", "thread-1")
        assert len(thread["messages"]) == 2


class TestPathInjection:
    """IDs arrive as model-supplied tool arguments and must not steer the URL.

    Assertions use ``raw_path`` -- the bytes actually sent -- because httpx's
    ``.path`` property percent-decodes for display and would show traversal
    that never leaves the process.
    """

    async def test_traversal_in_message_id_cannot_retarget_endpoint(
        self, gmail_client: GmailClient, fake_gmail: FakeGmail
    ) -> None:
        with pytest.raises(GmailApiError):
            await gmail_client.get_message("personal", "../../settings/forwarding")
        raw = fake_gmail.requests[-1].url.raw_path.decode()
        assert "/messages/..%2F..%2Fsettings%2Fforwarding" in raw
        assert "/settings/forwarding" not in raw

    async def test_traversal_in_thread_id_is_encoded(
        self, gmail_client: GmailClient, fake_gmail: FakeGmail
    ) -> None:
        await gmail_client.get_thread("personal", "../messages")
        raw = fake_gmail.requests[-1].url.raw_path.decode()
        assert "/threads/..%2Fmessages" in raw

    async def test_traversal_in_modify_id_is_encoded(
        self, gmail_client: GmailClient, fake_gmail: FakeGmail
    ) -> None:
        await gmail_client.modify_message(
            "personal", "../../drafts", add_label_ids=["STARRED"]
        )
        raw = fake_gmail.requests[-1].url.raw_path.decode()
        assert "/messages/..%2F..%2Fdrafts/modify" in raw

    async def test_ordinary_ids_are_unchanged(
        self, gmail_client: GmailClient, fake_gmail: FakeGmail
    ) -> None:
        await gmail_client.get_message("personal", "msg-plain")
        raw = fake_gmail.requests[-1].url.raw_path.decode()
        assert "/messages/msg-plain" in raw


class TestBearerTokens:
    async def test_authorization_header_present(
        self, gmail_client: GmailClient, fake_gmail: FakeGmail
    ) -> None:
        await gmail_client.get_profile("personal")
        assert fake_gmail.requests[-1].headers["Authorization"] == "Bearer tok-personal"

    async def test_each_account_uses_its_own_token(
        self, gmail_client: GmailClient, fake_gmail: FakeGmail
    ) -> None:
        await gmail_client.get_profile("personal")
        await gmail_client.get_profile("work-main")
        assert fake_gmail.requests[-2].headers["Authorization"] == "Bearer tok-personal"
        assert fake_gmail.requests[-1].headers["Authorization"] == "Bearer tok-work"


class TestWrites:
    async def test_send_posts_raw(
        self, gmail_client: GmailClient, fake_gmail: FakeGmail
    ) -> None:
        await gmail_client.send_message("personal", "cmF3")
        assert fake_gmail.sent[-1]["raw"] == "cmF3"

    async def test_send_includes_thread_id_when_replying(
        self, gmail_client: GmailClient, fake_gmail: FakeGmail
    ) -> None:
        await gmail_client.send_message("personal", "cmF3", thread_id="thread-1")
        assert fake_gmail.sent[-1]["threadId"] == "thread-1"

    async def test_draft_wraps_message(
        self, gmail_client: GmailClient, fake_gmail: FakeGmail
    ) -> None:
        await gmail_client.create_draft("personal", "cmF3")
        assert fake_gmail.drafts[-1]["message"]["raw"] == "cmF3"

    async def test_modify_sends_both_label_lists(
        self, gmail_client: GmailClient, fake_gmail: FakeGmail
    ) -> None:
        await gmail_client.modify_message(
            "personal", "msg-plain", add_label_ids=["STARRED"], remove_label_ids=["UNREAD"]
        )
        modification = fake_gmail.modifications[-1]
        assert modification["addLabelIds"] == ["STARRED"]
        assert modification["removeLabelIds"] == ["UNREAD"]

    async def test_labels_listed(self, gmail_client: GmailClient) -> None:
        labels = await gmail_client.list_labels("personal")
        assert {"id": "Label_12", "name": "Receipts", "type": "user"} in labels


class TestErrorMessages:
    """Error text is the model's only route to recovery, so assert on content."""

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (401, "re-run"),
            (403, "Permission denied"),
            (404, "Not found"),
            (429, "rate limit"),
            (500, "transient"),
        ],
    )
    async def test_status_codes_map_to_guidance(
        self,
        gmail_client: GmailClient,
        fake_gmail: FakeGmail,
        status: int,
        expected: str,
    ) -> None:
        fake_gmail.fail_with["/profile"] = status
        with pytest.raises(GmailApiError, match=expected):
            await gmail_client.get_profile("personal")

    async def test_error_names_the_account(
        self, gmail_client: GmailClient, fake_gmail: FakeGmail
    ) -> None:
        fake_gmail.fail_with["/profile"] = 403
        with pytest.raises(GmailApiError, match="work-main"):
            await gmail_client.get_profile("work-main")

    async def test_google_detail_is_included(
        self, gmail_client: GmailClient, fake_gmail: FakeGmail
    ) -> None:
        fake_gmail.fail_with["/profile"] = 403
        with pytest.raises(GmailApiError, match="injected failure"):
            await gmail_client.get_profile("personal")
