"""Tests for payload parsing and rendering."""

from __future__ import annotations

import pytest

from gmail_mcp.formatting import (
    decode_body_data,
    extract_body,
    format_date,
    headers_to_dict,
    html_to_text,
    internal_date_to_iso,
    render_messages_markdown,
    strip_quoted_reply,
    summarize_message,
    truncate,
)

from .conftest import (
    HTML_ONLY_MESSAGE,
    MULTIPART_MESSAGE,
    PLAIN_MESSAGE,
    b64,
)


class TestDecoding:
    def test_decodes_unpadded_base64url(self) -> None:
        assert decode_body_data(b64("hello world")) == "hello world"

    def test_handles_url_unsafe_characters(self) -> None:
        # '?' and '~' encode to bytes that differ between standard and URL-safe
        # base64, so this catches an alphabet mix-up.
        original = "a?b~c>d<e"
        assert decode_body_data(b64(original)) == original

    def test_empty_input_returns_empty(self) -> None:
        assert decode_body_data("") == ""

    def test_malformed_input_returns_empty_not_raises(self) -> None:
        assert decode_body_data("!!!not base64!!!") == ""

    def test_invalid_utf8_is_replaced_not_fatal(self) -> None:
        import base64 as _b64

        data = _b64.urlsafe_b64encode(b"caf\xff").decode().rstrip("=")
        assert decode_body_data(data).startswith("caf")


class TestHeaders:
    def test_lowercases_names(self) -> None:
        headers = headers_to_dict(PLAIN_MESSAGE["payload"])
        assert headers["subject"] == "Q3 numbers"
        assert headers["from"] == "Alice <alice@example.com>"

    def test_first_occurrence_wins(self) -> None:
        payload = {
            "headers": [
                {"name": "Received", "value": "first"},
                {"name": "Received", "value": "second"},
            ]
        }
        assert headers_to_dict(payload)["received"] == "first"

    def test_missing_headers_key(self) -> None:
        assert headers_to_dict({}) == {}


class TestBodyExtraction:
    def test_plain_text_body(self) -> None:
        body = extract_body(PLAIN_MESSAGE["payload"])
        assert "beat plan by 12%" in body

    def test_prefers_plain_over_html_in_multipart(self) -> None:
        body = extract_body(MULTIPART_MESSAGE["payload"])
        assert "Due in 30 days" in body
        assert "<p>" not in body

    def test_falls_back_to_html_when_no_plain_part(self) -> None:
        body = extract_body(HTML_ONLY_MESSAGE["payload"])
        assert "Your statement is ready." in body
        assert "Balance: $1,204.55" in body
        assert "<p>" not in body

    def test_drops_style_and_script_content(self) -> None:
        body = extract_body(HTML_ONLY_MESSAGE["payload"])
        assert "color:red" not in body

    def test_skips_attachment_parts(self) -> None:
        # The PDF part has an attachmentId and no data; it must not appear.
        body = extract_body(MULTIPART_MESSAGE["payload"])
        assert "invoice.pdf" not in body

    def test_empty_payload_returns_empty_string(self) -> None:
        assert extract_body({}) == ""


class TestHtmlToText:
    def test_block_tags_become_newlines(self) -> None:
        assert "one" in html_to_text("<p>one</p><p>two</p>")
        assert "two" in html_to_text("<p>one</p><p>two</p>")

    def test_entities_are_decoded(self) -> None:
        assert html_to_text("<p>a &amp; b</p>") == "a & b"

    def test_malformed_html_does_not_raise(self) -> None:
        assert "text" in html_to_text("<p><b>text</p></i>")


class TestQuotedReplyStripping:
    def test_removes_on_wrote_attribution_and_below(self) -> None:
        stripped = strip_quoted_reply(extract_body(PLAIN_MESSAGE["payload"]))
        assert "beat plan by 12%" in stripped
        assert "When will the numbers land?" not in stripped
        assert "wrote:" not in stripped

    def test_removes_outlook_original_message_marker(self) -> None:
        body = "My reply.\n\n-----Original Message-----\nFrom: someone"
        assert strip_quoted_reply(body) == "My reply."

    def test_removes_trailing_quote_block_without_attribution(self) -> None:
        body = "Answer here.\n\n> old question\n> more old"
        assert strip_quoted_reply(body) == "Answer here."

    def test_keeps_body_with_no_quoting(self) -> None:
        assert strip_quoted_reply("Just a note.") == "Just a note."

    def test_does_not_strip_quote_followed_by_new_text(self) -> None:
        body = "> quoted\n\nMy actual reply below the quote."
        assert "My actual reply below the quote." in strip_quoted_reply(body)


class TestTruncate:
    def test_short_text_untouched(self) -> None:
        assert truncate("short", 100) == ("short", False)

    def test_marks_truncation(self) -> None:
        text, was_truncated = truncate("x" * 500, 100)
        assert was_truncated
        assert text.endswith("[…]")

    def test_breaks_on_word_boundary_when_close(self) -> None:
        source = "alpha beta gamma delta epsilon zeta"
        text, was_truncated = truncate(source, 20)
        assert was_truncated
        assert not text.replace(" […]", "").endswith(("gamm", "delt"))

    def test_zero_limit_returns_original(self) -> None:
        assert truncate("abc", 0) == ("abc", False)


class TestDates:
    def test_rfc2822_to_utc(self) -> None:
        assert format_date("Mon, 02 Feb 2026 09:15:00 +0000") == "2026-02-02 09:15 UTC"

    def test_converts_offset_to_utc(self) -> None:
        assert format_date("Mon, 02 Feb 2026 11:15:00 +0200") == "2026-02-02 09:15 UTC"

    def test_unparseable_date_returned_verbatim(self) -> None:
        assert format_date("not a date") == "not a date"

    def test_empty_date(self) -> None:
        assert format_date("") == ""

    def test_internal_date_epoch_ms(self) -> None:
        assert internal_date_to_iso("1770000000000").endswith("UTC")

    @pytest.mark.parametrize("value", [None, "", "abc"])
    def test_bad_internal_date_returns_empty(self, value: object) -> None:
        assert internal_date_to_iso(value) == ""  # type: ignore[arg-type]


class TestSummarize:
    def test_metadata_only_omits_body(self) -> None:
        summary = summarize_message(PLAIN_MESSAGE, "work-main")
        assert "body" not in summary
        assert summary["account"] == "work-main"
        assert summary["subject"] == "Q3 numbers"
        assert summary["unread"] is True

    def test_includes_body_when_budget_given(self) -> None:
        summary = summarize_message(PLAIN_MESSAGE, "work-main", body_chars=500)
        assert "beat plan by 12%" in summary["body"]
        assert summary["body_truncated"] is False

    def test_sets_truncated_flag(self) -> None:
        summary = summarize_message(PLAIN_MESSAGE, "work-main", body_chars=10)
        assert summary["body_truncated"] is True

    def test_cc_included_only_when_present(self) -> None:
        assert "cc" not in summarize_message(PLAIN_MESSAGE, "a")
        assert summarize_message(MULTIPART_MESSAGE, "a")["cc"] == "cfo@example.com"

    def test_missing_subject_gets_placeholder(self) -> None:
        summary = summarize_message({"id": "x", "payload": {}}, "a")
        assert summary["subject"] == "(no subject)"

    def test_falls_back_to_internal_date_when_header_missing(self) -> None:
        message = {"id": "x", "internalDate": "1770000000000", "payload": {}}
        assert summarize_message(message, "a")["date"].endswith("UTC")


class TestMarkdownRendering:
    def test_empty_list_message(self) -> None:
        assert "No messages found" in render_messages_markdown([], "Results")

    def test_marks_unread(self) -> None:
        summaries = [summarize_message(PLAIN_MESSAGE, "work-main")]
        assert "[UNREAD]" in render_messages_markdown(summaries, "Results")

    def test_includes_ids_for_follow_up_calls(self) -> None:
        summaries = [summarize_message(PLAIN_MESSAGE, "work-main")]
        assert "msg-plain" in render_messages_markdown(summaries, "Results")

    def test_notes_truncation_to_the_model(self) -> None:
        summaries = [summarize_message(PLAIN_MESSAGE, "w", body_chars=10)]
        assert "body_chars" in render_messages_markdown(summaries, "Results")
