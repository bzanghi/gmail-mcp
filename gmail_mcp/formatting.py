"""Parsing Gmail payloads and rendering them for an LLM.

Two jobs live here:

1. Turning Gmail's nested MIME payload into flat, readable fields.
2. Keeping responses small. Gmail messages routinely run to tens of kilobytes
   of quoted history and HTML; dumping that into the context window is the
   single easiest way to make a mail server unusable. Bodies are truncated at
   an explicit character budget and quoted trailers are dropped.
"""

from __future__ import annotations

import base64
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any

# Body characters returned by default before truncation. Roughly 500 tokens --
# enough to see what a message says without flooding the context.
DEFAULT_BODY_CHARS = 2000
MAX_BODY_CHARS = 20000


def decode_body_data(data: str) -> str:
    """Decode Gmail's base64url body payload to text.

    Gmail omits padding, and bodies occasionally contain bytes that aren't
    valid UTF-8, so decoding is deliberately lenient -- a mangled character is
    better than a failed tool call.
    """
    if not data:
        return ""
    padded = data + "=" * (-len(data) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded)
    except Exception:  # noqa: BLE001 - malformed payloads should not crash a read
        return ""
    return raw.decode("utf-8", errors="replace")


class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML-to-text conversion for messages with no plain-text part."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in ("script", "style", "head"):
            self._skip_depth += 1
        elif tag in ("br", "p", "div", "tr", "li"):
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "head") and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._chunks.append(data)

    def text(self) -> str:
        joined = "".join(self._chunks)
        # Collapse the run of blank lines that block-tag handling produces.
        return re.sub(r"\n{3,}", "\n\n", joined).strip()


def html_to_text(html: str) -> str:
    """Strip tags from an HTML body, keeping paragraph breaks."""
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html)
    except Exception:  # noqa: BLE001 - malformed HTML is common in real mail
        pass
    return parser.text()


def headers_to_dict(payload: dict[str, Any]) -> dict[str, str]:
    """Flatten Gmail's ``[{name, value}]`` header list, lower-casing names."""
    result: dict[str, str] = {}
    for header in payload.get("headers", []) or []:
        name = str(header.get("name", "")).lower()
        if name and name not in result:
            result[name] = str(header.get("value", ""))
    return result


def extract_body(payload: dict[str, Any]) -> str:
    """Pull the best available text body out of a Gmail payload.

    Prefers ``text/plain``; falls back to converting ``text/html``. Walks the
    full MIME tree, so ``multipart/mixed`` wrapping ``multipart/alternative``
    (the usual shape once an attachment is present) resolves correctly.
    """
    plain: list[str] = []
    html: list[str] = []

    def walk(part: dict[str, Any]) -> None:
        mime = str(part.get("mimeType", ""))
        body = part.get("body", {}) or {}
        data = body.get("data", "")

        # Skip parts that are attachments rather than inline content.
        if body.get("attachmentId") and not data:
            return

        if data:
            if mime == "text/plain":
                plain.append(decode_body_data(data))
            elif mime == "text/html":
                html.append(decode_body_data(data))

        for child in part.get("parts", []) or []:
            walk(child)

    walk(payload)

    if plain:
        return "\n".join(p for p in plain if p.strip()).strip()
    if html:
        return html_to_text("\n".join(html))
    return ""


def extract_attachments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """List attachment metadata: filename, MIME type, and size in bytes.

    Metadata only -- content is never downloaded, since a single PDF would
    blow the context budget. Knowing an attachment exists and what it is
    called is what makes "did the invoice arrive?" answerable.
    """
    found: list[dict[str, Any]] = []

    def walk(part: dict[str, Any]) -> None:
        filename = str(part.get("filename", "") or "")
        body = part.get("body", {}) or {}
        if filename and (body.get("attachmentId") or body.get("size")):
            found.append(
                {
                    "filename": filename,
                    "mime_type": str(part.get("mimeType", "")),
                    "size_bytes": int(body.get("size", 0) or 0),
                }
            )
        for child in part.get("parts", []) or []:
            walk(child)

    walk(payload)
    return found


def strip_quoted_reply(body: str) -> str:
    """Drop the quoted trailer from a reply.

    Recognises the two conventions that cover almost all real mail: an
    ``On <date>, <person> wrote:`` attribution line, and a run of ``>`` quoted
    lines. Anything before the first such marker is kept.
    """
    lines = body.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^On .{5,120}\bwrote:$", stripped):
            return "\n".join(lines[:index]).strip()
        if stripped.startswith("-----Original Message-----"):
            return "\n".join(lines[:index]).strip()
    # Trailing block of quoted lines with nothing unquoted after it.
    while lines and (not lines[-1].strip() or lines[-1].lstrip().startswith(">")):
        lines.pop()
    return "\n".join(lines).strip()


def truncate(text: str, limit: int) -> tuple[str, bool]:
    """Truncate to ``limit`` characters, returning ``(text, was_truncated)``.

    Cuts on a word boundary when one is nearby so the tail isn't a broken word.
    """
    if limit <= 0 or len(text) <= limit:
        return text, False
    window = text[:limit]
    boundary = window.rfind(" ")
    if boundary > limit * 0.8:
        window = window[:boundary]
    return window.rstrip() + " […]", True


def format_date(raw: str) -> str:
    """Normalise a Date header to ``YYYY-MM-DD HH:MM UTC``.

    Falls back to the raw header when it can't be parsed -- an odd-looking date
    is more useful than a missing one.
    """
    if not raw:
        return ""
    try:
        from email.utils import parsedate_to_datetime

        parsed = parsedate_to_datetime(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:  # noqa: BLE001 - malformed Date headers are common
        return raw


def internal_date_to_iso(internal_date: str | int | None) -> str:
    """Convert Gmail's ``internalDate`` (epoch ms, as a string) to ISO-8601."""
    if internal_date in (None, ""):
        return ""
    try:
        seconds = int(internal_date) / 1000
    except (TypeError, ValueError):
        return ""
    return (
        datetime.fromtimestamp(seconds, tz=timezone.utc)
        .strftime("%Y-%m-%d %H:%M UTC")
    )


def summarize_message(
    message: dict[str, Any], account: str, body_chars: int = 0
) -> dict[str, Any]:
    """Reduce a Gmail message to the fields worth putting in the context.

    Args:
        message: A message resource from ``messages.get``.
        account: The account alias, echoed back so multi-account results stay
            unambiguous when merged into one list.
        body_chars: Character budget for the body. ``0`` omits the body
            entirely and returns only Gmail's own snippet, which is what
            listings use.

    Returns:
        A dict with ``account``, ``id``, ``thread_id``, ``from``, ``to``,
        ``subject``, ``date``, ``snippet``, ``labels``, ``unread``, and
        (when ``body_chars`` > 0) ``body`` plus ``body_truncated``.
    """
    payload = message.get("payload", {}) or {}
    headers = headers_to_dict(payload)
    label_ids = message.get("labelIds", []) or []

    result: dict[str, Any] = {
        "account": account,
        "id": message.get("id", ""),
        "thread_id": message.get("threadId", ""),
        "from": headers.get("from", ""),
        "to": headers.get("to", ""),
        "subject": headers.get("subject", "(no subject)"),
        "date": format_date(headers.get("date", ""))
        or internal_date_to_iso(message.get("internalDate")),
        "snippet": (message.get("snippet", "") or "").strip(),
        "labels": label_ids,
        "unread": "UNREAD" in label_ids,
    }
    if headers.get("cc"):
        result["cc"] = headers["cc"]

    attachments = extract_attachments(payload)
    result["has_attachments"] = bool(attachments)
    if attachments:
        result["attachments"] = attachments

    if body_chars > 0:
        body = strip_quoted_reply(extract_body(payload))
        text, was_truncated = truncate(body, body_chars)
        result["body"] = text
        result["body_truncated"] = was_truncated

    return result


def render_messages_markdown(messages: list[dict[str, Any]], title: str) -> str:
    """Render summarized messages as compact Markdown."""
    if not messages:
        return f"# {title}\n\nNo messages found."

    lines = [f"# {title}", "", f"{len(messages)} message(s)", ""]
    for msg in messages:
        flag = "**[UNREAD]** " if msg.get("unread") else ""
        lines.append(f"## {flag}{msg['subject']}")
        lines.append(f"- **Account**: {msg['account']}")
        lines.append(f"- **From**: {msg['from']}")
        if msg.get("date"):
            lines.append(f"- **Date**: {msg['date']}")
        lines.append(f"- **Message ID**: `{msg['id']}`")
        if msg.get("snippet"):
            lines.append(f"- **Snippet**: {msg['snippet']}")
        for attachment in msg.get("attachments", []):
            size_kb = max(1, attachment["size_bytes"] // 1024)
            lines.append(
                f"- **Attachment**: {attachment['filename']} "
                f"({attachment['mime_type']}, {size_kb} KB)"
            )
        if msg.get("body"):
            lines.append("")
            lines.append(msg["body"])
            if msg.get("body_truncated"):
                lines.append("")
                lines.append(
                    "_Body truncated. Re-read this message with a larger "
                    "`body_chars` for the full text._"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
