"""Building RFC 2822 messages for sending and drafting.

Gmail's ``send`` and ``drafts.create`` endpoints take a base64url-encoded
message rather than structured fields, so composition happens here.
"""

from __future__ import annotations

import base64
from email.message import EmailMessage
from typing import Any


def build_message(
    to: list[str],
    subject: str,
    body: str,
    sender: str = "",
    cc: list[str] | None = None,
    bcc: list[str] | None = None,
    html_body: str = "",
    reply_headers: dict[str, str] | None = None,
) -> str:
    """Compose a message and return it base64url-encoded for the Gmail API.

    Args:
        to: Recipient addresses. At least one is required.
        subject: Subject line.
        body: Plain-text body.
        sender: ``From`` address. Gmail overrides this with the authenticated
            account, so it only matters for send-as aliases.
        cc: Carbon-copy addresses.
        bcc: Blind carbon-copy addresses.
        html_body: Optional HTML alternative. When given, the message becomes
            ``multipart/alternative`` with ``body`` as the plain-text fallback.
        reply_headers: ``In-Reply-To`` and ``References`` values, which are what
            actually make a reply thread correctly in the recipient's client.
            ``threadId`` alone only threads it on the sender's side.

    Returns:
        The base64url-encoded (unpadded) message, ready for the ``raw`` field.

    Raises:
        ValueError: If no recipients are supplied.
    """
    if not to:
        raise ValueError("At least one recipient is required in 'to'.")

    message = EmailMessage()
    message["To"] = ", ".join(to)
    message["Subject"] = subject
    if sender:
        message["From"] = sender
    if cc:
        message["Cc"] = ", ".join(cc)
    if bcc:
        message["Bcc"] = ", ".join(bcc)

    for header, value in (reply_headers or {}).items():
        if value:
            message[header] = value

    message.set_content(body)
    if html_body:
        message.add_alternative(html_body, subtype="html")

    return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")


def reply_metadata(original: dict[str, Any]) -> dict[str, str]:
    """Derive reply headers and subject from the message being replied to.

    Returns a dict with ``subject``, ``thread_id``, ``in_reply_to``,
    ``references``, and ``reply_to`` (the address to send to). ``References``
    accumulates the existing chain plus this message's ID, which is what keeps
    long threads intact.
    """
    from .formatting import headers_to_dict

    payload = original.get("payload", {}) or {}
    headers = headers_to_dict(payload)

    subject = headers.get("subject", "")
    if subject and not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"

    message_id = headers.get("message-id", "")
    existing_refs = headers.get("references", "")
    references = f"{existing_refs} {message_id}".strip() if message_id else existing_refs

    return {
        "subject": subject or "(no subject)",
        "thread_id": original.get("threadId", ""),
        "in_reply_to": message_id,
        "references": references,
        "reply_to": headers.get("reply-to") or headers.get("from", ""),
        # Everyone else on the original, for reply-all. The caller is
        # responsible for removing the sending account's own address.
        "original_to": headers.get("to", ""),
        "original_cc": headers.get("cc", ""),
    }


def split_addresses(raw: str) -> list[str]:
    """Split a header value into individual addresses.

    Uses email.utils.getaddresses so that display names containing commas
    ("Doe, Jane" <jane@x.com>) do not split into fragments.
    """
    from email.utils import getaddresses

    return [addr for _, addr in getaddresses([raw or ""]) if addr]


def dedupe_addresses(addresses: list[str], exclude: list[str]) -> list[str]:
    """Drop duplicates and excluded addresses, preserving order.

    Matching is case-insensitive because mail addresses are compared that way
    in practice, and this is what stops a reply-all from CC'ing the sender.
    """
    blocked = {a.strip().lower() for a in exclude if a and a.strip()}
    seen: set[str] = set()
    result: list[str] = []
    for address in addresses:
        key = address.strip().lower()
        if not key or key in blocked or key in seen:
            continue
        seen.add(key)
        result.append(address.strip())
    return result
