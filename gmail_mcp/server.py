"""MCP server exposing multiple Gmail accounts to Claude Desktop.

Every tool takes an ``account`` alias (``personal``, ``work-main``, ...) so the
model can act across accounts without the user switching anything. The one
exception is :func:`gmail_check_inboxes`, which sweeps all accounts at once and
is the tool intended for scheduled use.
"""

from __future__ import annotations

import asyncio
import json
from enum import Enum
from typing import Any, Optional

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .auth import AuthError, OAuthClient, TokenManager, default_token_store
from .config import Config, ConfigError, load_config
from .formatting import (
    DEFAULT_BODY_CHARS,
    MAX_BODY_CHARS,
    render_messages_markdown,
    summarize_message,
)
from .gmail_client import GmailApiError, GmailClient
from .mime import build_message, reply_metadata

mcp = FastMCP("gmail_mcp")

# Subjects/snippets shown per account by the inbox sweep. Deliberately small:
# this tool runs on a schedule, so its output lands in the context repeatedly.
CHECK_PREVIEW_PER_ACCOUNT = 5


class ResponseFormat(str, Enum):
    """Output format for tool responses."""

    MARKDOWN = "markdown"
    JSON = "json"


class _AppState:
    """Lazily-initialised server state.

    Config and credentials are loaded on first tool call rather than at import
    time so that a setup mistake surfaces as a readable tool error inside
    Claude Desktop instead of a silent failure to start.
    """

    def __init__(self) -> None:
        self._config: Config | None = None
        self._client: GmailClient | None = None
        self._http: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    async def get(self) -> tuple[Config, GmailClient]:
        async with self._lock:
            if self._config is None or self._client is None:
                config = load_config()
                oauth = OAuthClient.from_secrets_file(config.client_secrets_file)
                store = default_token_store(config.config_dir)
                self._http = httpx.AsyncClient(timeout=30.0)
                manager = TokenManager(oauth, store, http=self._http)
                self._config = config
                self._client = GmailClient(token_manager=manager, http=self._http)
            return self._config, self._client

    def reset(self) -> None:
        """Drop cached state. Used by tests."""
        self._config = None
        self._client = None
        self._http = None


_state = _AppState()


def _handle_error(exc: Exception) -> str:
    """Convert an exception into an actionable message for the model."""
    if isinstance(exc, (ConfigError, AuthError, GmailApiError, ValueError)):
        return f"Error: {exc}"
    return f"Error: Unexpected {type(exc).__name__}: {exc}"


def _render(
    messages: list[dict[str, Any]],
    title: str,
    fmt: ResponseFormat,
    extra: dict[str, Any] | None = None,
) -> str:
    """Render summarized messages in the requested format."""
    if fmt == ResponseFormat.MARKDOWN:
        return render_messages_markdown(messages, title)
    payload: dict[str, Any] = {"count": len(messages), "messages": messages}
    payload.update(extra or {})
    return json.dumps(payload, indent=2)


# --------------------------------------------------------------------------
# Input models
# --------------------------------------------------------------------------


class _Base(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )


class ListAccountsInput(_Base):
    """No parameters; listing accounts is unconditional."""


class SearchInput(_Base):
    """Input for searching one account."""

    account: str = Field(
        ...,
        description="Account alias to search, e.g. 'personal' or 'work-main'. "
        "Call gmail_list_accounts first if unsure.",
        min_length=1,
    )
    query: str = Field(
        default="",
        description="Gmail search query using Gmail's own syntax, e.g. "
        "'from:alice@example.com is:unread', 'subject:invoice after:2026/01/01', "
        "'has:attachment larger:5M'. Empty returns the most recent mail.",
        max_length=2000,
    )
    limit: int = Field(
        default=20,
        description="Maximum messages to return (1-100).",
        ge=1,
        le=100,
    )
    page_token: Optional[str] = Field(
        default=None,
        description="Token from a previous response's 'next_page_token' to "
        "fetch the following page.",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="'markdown' for readable output, 'json' for structured data.",
    )


class ReadMessageInput(_Base):
    """Input for reading a single message."""

    account: str = Field(..., description="Account alias holding the message.", min_length=1)
    message_id: str = Field(
        ...,
        description="Gmail message ID, as returned by gmail_search_messages.",
        min_length=1,
    )
    body_chars: int = Field(
        default=DEFAULT_BODY_CHARS,
        description=f"Body characters to return before truncating "
        f"(1-{MAX_BODY_CHARS}). Raise this when the truncated body cut off "
        "something you need.",
        ge=1,
        le=MAX_BODY_CHARS,
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class ReadThreadInput(_Base):
    """Input for reading a full conversation."""

    account: str = Field(..., description="Account alias holding the thread.", min_length=1)
    thread_id: str = Field(
        ...,
        description="Gmail thread ID, as returned in a message's 'thread_id'.",
        min_length=1,
    )
    body_chars: int = Field(
        default=1000,
        description="Body characters per message in the thread. Lower than the "
        "single-message default because threads multiply the total.",
        ge=1,
        le=MAX_BODY_CHARS,
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class SendInput(_Base):
    """Input for sending mail."""

    account: str = Field(..., description="Account alias to send from.", min_length=1)
    to: list[str] = Field(
        default_factory=list,
        description="Recipient email addresses. May be omitted only when "
        "'reply_to_message_id' is set, in which case the original sender is "
        "used automatically.",
        max_length=50,
    )
    subject: str = Field(default="", description="Subject line.", max_length=1000)
    body: str = Field(..., description="Plain-text message body.", min_length=1)
    cc: list[str] = Field(default_factory=list, description="CC addresses.", max_length=50)
    bcc: list[str] = Field(default_factory=list, description="BCC addresses.", max_length=50)
    reply_to_message_id: Optional[str] = Field(
        default=None,
        description="Message ID being replied to. When set, the subject, "
        "threading headers, and thread are derived from it automatically, so "
        "'to' and 'subject' may be left to their defaults.",
    )

    @field_validator("to", "cc", "bcc")
    @classmethod
    def validate_addresses(cls, values: list[str]) -> list[str]:
        cleaned = [v.strip() for v in values if v and v.strip()]
        for address in cleaned:
            if "@" not in address:
                raise ValueError(
                    f"{address!r} is not a valid email address (no '@')."
                )
        return cleaned


class DraftInput(SendInput):
    """Input for creating a draft. Identical in shape to sending."""


class ModifyLabelsInput(_Base):
    """Input for changing labels on a message."""

    account: str = Field(..., description="Account alias holding the message.", min_length=1)
    message_id: str = Field(..., description="Gmail message ID.", min_length=1)
    add_labels: list[str] = Field(
        default_factory=list,
        description="Label IDs to add. System labels include 'STARRED', "
        "'IMPORTANT', 'UNREAD', 'INBOX'. Use gmail_list_labels for user labels.",
        max_length=20,
    )
    remove_labels: list[str] = Field(
        default_factory=list,
        description="Label IDs to remove. Removing 'UNREAD' marks a message "
        "read; removing 'INBOX' archives it.",
        max_length=20,
    )


class ListLabelsInput(_Base):
    """Input for listing labels."""

    account: str = Field(..., description="Account alias to list labels for.", min_length=1)


class CheckInboxesInput(_Base):
    """Input for the periodic multi-account sweep."""

    accounts: list[str] = Field(
        default_factory=list,
        description="Account aliases to check. Empty (the default) checks every "
        "configured account.",
        max_length=25,
    )
    query: str = Field(
        default="is:unread in:inbox",
        description="Gmail query applied to each account. The default finds "
        "unread inbox mail; narrow it with e.g. 'is:unread in:inbox newer_than:1d'.",
        max_length=500,
    )
    max_per_account: int = Field(
        default=25,
        description="Cap on messages counted per account (1-100).",
        ge=1,
        le=100,
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


@mcp.tool(
    name="gmail_list_accounts",
    annotations={
        "title": "List Configured Gmail Accounts",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def gmail_list_accounts(params: ListAccountsInput) -> str:
    """List the Gmail accounts this server can act on, and their auth status.

    Call this first in a session when you do not already know the account
    aliases. Every other tool takes one of these aliases.

    Args:
        params (ListAccountsInput): No fields.

    Returns:
        str: JSON with schema:
        {
            "accounts": [
                {
                    "name": str,          # alias to pass as 'account' (e.g. "work-main")
                    "email": str,         # the Gmail address, for display
                    "description": str,   # what this account is used for
                    "authenticated": bool # false means setup login is still needed
                }
            ]
        }

        Error response: "Error: <message>"

    Examples:
        - Use when: "Which of my inboxes can you see?"
        - Use when: you need an alias before calling any other gmail_ tool.
    """
    try:
        config, client = await _state.get()
        authed = set(
            client.token_manager.authenticated_accounts(config.account_names())
        )
        return json.dumps(
            {
                "accounts": [
                    {**a.to_dict(), "authenticated": a.name in authed}
                    for a in config.accounts
                ]
            },
            indent=2,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as text
        return _handle_error(exc)


@mcp.tool(
    name="gmail_search_messages",
    annotations={
        "title": "Search Gmail Messages",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def gmail_search_messages(params: SearchInput) -> str:
    """Search one Gmail account and return message headers and snippets.

    Returns metadata only -- no message bodies. Follow up with
    gmail_read_message for the full text of anything that looks relevant. This
    keeps large result sets from flooding the context.

    Args:
        params (SearchInput): Validated parameters containing:
            - account (str): Account alias to search
            - query (str): Gmail search syntax (e.g. "from:bob is:unread")
            - limit (int): Max messages, 1-100 (default 20)
            - page_token (Optional[str]): Pagination token from a prior call
            - response_format (ResponseFormat): 'markdown' or 'json'

    Returns:
        str: Markdown listing, or JSON with schema:
        {
            "count": int,
            "next_page_token": str | null,   # pass back as page_token for more
            "messages": [
                {
                    "account": str, "id": str, "thread_id": str,
                    "from": str, "to": str, "subject": str, "date": str,
                    "snippet": str, "labels": [str], "unread": bool
                }
            ]
        }

        Error response: "Error: <message>"

    Examples:
        - Use when: "Any unread mail from the bank this week?" ->
          query="from:bank is:unread newer_than:7d"
        - Use when: "Find the invoice thread" -> query="subject:invoice"
        - Don't use when: you want every account at once (use gmail_check_inboxes)
        - Don't use when: you already have a message ID (use gmail_read_message)

    Error Handling:
        - Unknown account alias returns an error naming the valid aliases.
        - Expired credentials return an error naming the re-login command.
    """
    try:
        config, client = await _state.get()
        account = config.get(params.account)
        ids, next_token = await client.list_message_ids(
            account.name,
            query=params.query,
            limit=params.limit,
            page_token=params.page_token,
        )
        raw_messages = await client.get_messages(account.name, ids, fmt="metadata")
        messages = [summarize_message(m, account.name) for m in raw_messages]
        title = f"Search results in '{account.name}'"
        if params.query:
            title += f" for: {params.query}"
        return _render(
            messages, title, params.response_format, {"next_page_token": next_token}
        )
    except Exception as exc:  # noqa: BLE001
        return _handle_error(exc)


@mcp.tool(
    name="gmail_read_message",
    annotations={
        "title": "Read a Gmail Message",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def gmail_read_message(params: ReadMessageInput) -> str:
    """Read one message in full, including its body.

    The body is plain text (HTML-only mail is converted), has quoted reply
    trailers removed, and is truncated to 'body_chars'. Raise 'body_chars' and
    call again if the text was cut off mid-thought.

    Args:
        params (ReadMessageInput): Validated parameters containing:
            - account (str): Account alias holding the message
            - message_id (str): Gmail message ID from a search
            - body_chars (int): Body character budget (default 2000)
            - response_format (ResponseFormat): 'markdown' or 'json'

    Returns:
        str: Markdown, or JSON with schema:
        {
            "count": 1,
            "messages": [
                {
                    "account": str, "id": str, "thread_id": str,
                    "from": str, "to": str, "cc": str, "subject": str,
                    "date": str, "snippet": str, "labels": [str],
                    "unread": bool, "body": str, "body_truncated": bool
                }
            ]
        }

        Error response: "Error: <message>"

    Examples:
        - Use when: "What does that message from Sarah actually say?"
        - Don't use when: you want the whole conversation (use gmail_read_thread)
    """
    try:
        config, client = await _state.get()
        account = config.get(params.account)
        raw = await client.get_message(account.name, params.message_id, fmt="full")
        message = summarize_message(raw, account.name, body_chars=params.body_chars)
        return _render([message], f"Message in '{account.name}'", params.response_format)
    except Exception as exc:  # noqa: BLE001
        return _handle_error(exc)


@mcp.tool(
    name="gmail_read_thread",
    annotations={
        "title": "Read a Gmail Thread",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def gmail_read_thread(params: ReadThreadInput) -> str:
    """Read every message in a conversation, oldest first.

    Use this when the context of a back-and-forth matters -- who said what, in
    what order -- rather than a single message in isolation.

    Args:
        params (ReadThreadInput): Validated parameters containing:
            - account (str): Account alias holding the thread
            - thread_id (str): Gmail thread ID
            - body_chars (int): Body budget per message (default 1000)
            - response_format (ResponseFormat): 'markdown' or 'json'

    Returns:
        str: Markdown, or JSON with the same message schema as
        gmail_read_message, with one entry per message in the thread.

        Error response: "Error: <message>"

    Examples:
        - Use when: "Summarise the whole exchange with the contractor."
        - Use when: you need to reply in context and want prior messages first.
    """
    try:
        config, client = await _state.get()
        account = config.get(params.account)
        thread = await client.get_thread(account.name, params.thread_id, fmt="full")
        messages = [
            summarize_message(m, account.name, body_chars=params.body_chars)
            for m in thread.get("messages", [])
        ]
        return _render(
            messages,
            f"Thread {params.thread_id} in '{account.name}'",
            params.response_format,
        )
    except Exception as exc:  # noqa: BLE001
        return _handle_error(exc)


async def _compose(
    client: GmailClient, account_name: str, params: SendInput
) -> tuple[str, str | None]:
    """Build the raw message for a send or draft, resolving reply context.

    Returns ``(base64url_message, thread_id)``.
    """
    reply_headers: dict[str, str] = {}
    thread_id: str | None = None
    to = list(params.to)
    subject = params.subject

    if params.reply_to_message_id:
        original = await client.get_message(
            account_name, params.reply_to_message_id, fmt="metadata"
        )
        meta = reply_metadata(original)
        thread_id = meta["thread_id"] or None
        reply_headers = {
            "In-Reply-To": meta["in_reply_to"],
            "References": meta["references"],
        }
        if not subject:
            subject = meta["subject"]
        if not to and meta["reply_to"]:
            to = [meta["reply_to"]]

    if not to:
        raise ValueError(
            "No recipients. Provide 'to', or set 'reply_to_message_id' so the "
            "original sender can be used."
        )

    raw = build_message(
        to=to,
        subject=subject,
        body=params.body,
        cc=params.cc,
        bcc=params.bcc,
        reply_headers=reply_headers,
    )
    return raw, thread_id


@mcp.tool(
    name="gmail_send_message",
    annotations={
        "title": "Send a Gmail Message",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def gmail_send_message(params: SendInput) -> str:
    """Send an email from one of the configured accounts.

    This delivers mail immediately and cannot be undone. Confirm the recipient,
    account, and content with the user before calling it. To prepare something
    for the user to review first, use gmail_create_draft instead.

    Args:
        params (SendInput): Validated parameters containing:
            - account (str): Account alias to send from
            - to (list[str]): Recipients (may be omitted when replying)
            - subject (str): Subject (derived from the original when replying)
            - body (str): Plain-text body
            - cc (list[str]), bcc (list[str]): Optional copies
            - reply_to_message_id (Optional[str]): Reply in-thread to this message

    Returns:
        str: JSON with schema:
        {"status": "sent", "account": str, "message_id": str, "thread_id": str}

        Error response: "Error: <message>"

    Examples:
        - Use when: the user explicitly asked you to send a specific email.
        - Don't use when: the user asked you to "write" or "prepare" something
          (use gmail_create_draft).

    Error Handling:
        - An address without '@' is rejected before anything is sent.
        - Missing recipients (and no reply context) returns an error.
    """
    try:
        config, client = await _state.get()
        account = config.get(params.account)
        raw, thread_id = await _compose(client, account.name, params)
        result = await client.send_message(account.name, raw, thread_id=thread_id)
        return json.dumps(
            {
                "status": "sent",
                "account": account.name,
                "message_id": result.get("id", ""),
                "thread_id": result.get("threadId", ""),
            },
            indent=2,
        )
    except Exception as exc:  # noqa: BLE001
        return _handle_error(exc)


@mcp.tool(
    name="gmail_create_draft",
    annotations={
        "title": "Create a Gmail Draft",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def gmail_create_draft(params: DraftInput) -> str:
    """Save an email as a draft without sending it.

    The draft appears in the account's Drafts folder for the user to review,
    edit, and send themselves. Prefer this over gmail_send_message whenever the
    user has not clearly asked for the mail to go out immediately.

    Args:
        params (DraftInput): Same fields as gmail_send_message.

    Returns:
        str: JSON with schema:
        {"status": "draft_created", "account": str, "draft_id": str,
         "message_id": str, "thread_id": str}

        Error response: "Error: <message>"

    Examples:
        - Use when: "Draft a reply to this and I'll look it over."
        - Use when: you are unsure whether the user wants it sent.
    """
    try:
        config, client = await _state.get()
        account = config.get(params.account)
        raw, thread_id = await _compose(client, account.name, params)
        result = await client.create_draft(account.name, raw, thread_id=thread_id)
        message = result.get("message", {}) or {}
        return json.dumps(
            {
                "status": "draft_created",
                "account": account.name,
                "draft_id": result.get("id", ""),
                "message_id": message.get("id", ""),
                "thread_id": message.get("threadId", ""),
            },
            indent=2,
        )
    except Exception as exc:  # noqa: BLE001
        return _handle_error(exc)


@mcp.tool(
    name="gmail_modify_labels",
    annotations={
        "title": "Add or Remove Gmail Labels",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def gmail_modify_labels(params: ModifyLabelsInput) -> str:
    """Add and/or remove labels on a message.

    This is how you mark mail read ('remove_labels': ["UNREAD"]), archive it
    ('remove_labels': ["INBOX"]), star it ('add_labels': ["STARRED"]), or file
    it under a user label. Nothing here deletes mail.

    Args:
        params (ModifyLabelsInput): Validated parameters containing:
            - account (str): Account alias holding the message
            - message_id (str): Gmail message ID
            - add_labels (list[str]): Label IDs to add
            - remove_labels (list[str]): Label IDs to remove

    Returns:
        str: JSON with schema:
        {"status": "modified", "account": str, "message_id": str,
         "labels": [str]}   # the message's labels after the change

        Error response: "Error: <message>"

    Examples:
        - Use when: "Mark that one as read" -> remove_labels=["UNREAD"]
        - Use when: "Archive it" -> remove_labels=["INBOX"]
        - Don't use when: you need a user label's ID (call gmail_list_labels)
    """
    try:
        if not params.add_labels and not params.remove_labels:
            return (
                "Error: Specify at least one label in 'add_labels' or "
                "'remove_labels'."
            )
        config, client = await _state.get()
        account = config.get(params.account)
        result = await client.modify_message(
            account.name,
            params.message_id,
            add_label_ids=params.add_labels,
            remove_label_ids=params.remove_labels,
        )
        return json.dumps(
            {
                "status": "modified",
                "account": account.name,
                "message_id": result.get("id", params.message_id),
                "labels": result.get("labelIds", []),
            },
            indent=2,
        )
    except Exception as exc:  # noqa: BLE001
        return _handle_error(exc)


@mcp.tool(
    name="gmail_list_labels",
    annotations={
        "title": "List Gmail Labels",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def gmail_list_labels(params: ListLabelsInput) -> str:
    """List an account's labels and their IDs.

    Label IDs, not names, are what gmail_modify_labels expects. User-created
    labels have opaque IDs like 'Label_12'; system labels are named ('INBOX').

    Args:
        params (ListLabelsInput): Containing:
            - account (str): Account alias to list labels for

    Returns:
        str: JSON with schema:
        {
            "account": str,
            "labels": [{"id": str, "name": str, "type": str}]  # type: system|user
        }

        Error response: "Error: <message>"

    Examples:
        - Use when: "File this under my Receipts label" -- look up the ID first.
    """
    try:
        config, client = await _state.get()
        account = config.get(params.account)
        labels = await client.list_labels(account.name)
        return json.dumps(
            {
                "account": account.name,
                "labels": [
                    {
                        "id": label.get("id", ""),
                        "name": label.get("name", ""),
                        "type": label.get("type", ""),
                    }
                    for label in labels
                ],
            },
            indent=2,
        )
    except Exception as exc:  # noqa: BLE001
        return _handle_error(exc)


@mcp.tool(
    name="gmail_check_inboxes",
    annotations={
        "title": "Check All Gmail Inboxes",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def gmail_check_inboxes(params: CheckInboxesInput) -> str:
    """Sweep every configured inbox and return one short summary.

    Built for scheduled use: it queries all accounts concurrently and returns
    counts plus a handful of subject lines per account, never full bodies. Use
    it to answer "anything I need to look at?" in a single call, then follow up
    with gmail_read_message on whatever stands out.

    Messages Gmail itself marks IMPORTANT or STARRED are flagged, so the
    prioritisation reflects the user's own Gmail signals rather than guesswork
    about which words sound urgent.

    Args:
        params (CheckInboxesInput): Validated parameters containing:
            - accounts (list[str]): Aliases to check; empty means all
            - query (str): Per-account Gmail query
              (default "is:unread in:inbox")
            - max_per_account (int): Cap per account, 1-100 (default 25)
            - response_format (ResponseFormat): 'markdown' or 'json'

    Returns:
        str: Markdown summary, or JSON with schema:
        {
            "total_matching": int,          # across all checked accounts
            "flagged_count": int,           # important or starred
            "checked_at_accounts": [str],
            "accounts": [
                {
                    "account": str,
                    "matching": int,        # capped by max_per_account
                    "flagged": int,
                    "error": str | null,    # set if this account alone failed
                    "preview": [
                        {"id": str, "thread_id": str, "from": str,
                         "subject": str, "date": str, "snippet": str,
                         "flagged": bool}
                    ]
                }
            ]
        }

        Error response: "Error: <message>"

    Examples:
        - Use when: "Anything urgent across my inboxes?"
        - Use when: running on a schedule to report new mail.
        - Don't use when: you need one account in depth (use gmail_search_messages)

    Error Handling:
        - One account failing does not fail the sweep; its 'error' field is set
          and the other accounts still report.
    """
    try:
        config, client = await _state.get()
        accounts = config.resolve(params.accounts)

        async def check(account_name: str) -> dict[str, Any]:
            try:
                ids, _ = await client.list_message_ids(
                    account_name, query=params.query, limit=params.max_per_account
                )
                raw = await client.get_messages(
                    account_name, ids[:CHECK_PREVIEW_PER_ACCOUNT], fmt="metadata"
                )
                preview = []
                flagged_total = 0
                for message in raw:
                    summary = summarize_message(message, account_name)
                    labels = set(summary.get("labels", []))
                    is_flagged = bool(labels & {"IMPORTANT", "STARRED"})
                    flagged_total += int(is_flagged)
                    preview.append(
                        {
                            "id": summary["id"],
                            "thread_id": summary["thread_id"],
                            "from": summary["from"],
                            "subject": summary["subject"],
                            "date": summary["date"],
                            "snippet": summary["snippet"],
                            "flagged": is_flagged,
                        }
                    )
                return {
                    "account": account_name,
                    "matching": len(ids),
                    "flagged": flagged_total,
                    "error": None,
                    "preview": preview,
                }
            except Exception as exc:  # noqa: BLE001 - isolate per-account failure
                return {
                    "account": account_name,
                    "matching": 0,
                    "flagged": 0,
                    "error": str(exc),
                    "preview": [],
                }

        results = await asyncio.gather(*(check(a.name) for a in accounts))
        total = sum(r["matching"] for r in results)
        flagged = sum(r["flagged"] for r in results)

        if params.response_format == ResponseFormat.JSON:
            return json.dumps(
                {
                    "total_matching": total,
                    "flagged_count": flagged,
                    "checked_at_accounts": [a.name for a in accounts],
                    "accounts": results,
                },
                indent=2,
            )

        lines = [
            "# Inbox check",
            "",
            f"**{total} message(s)** matching `{params.query}` across "
            f"{len(accounts)} account(s); **{flagged} flagged** "
            "(important or starred).",
            "",
        ]
        for result in results:
            lines.append(f"## {result['account']}")
            if result["error"]:
                lines.append(f"- Could not check: {result['error']}")
                lines.append("")
                continue
            if not result["matching"]:
                lines.append("- Nothing matching.")
                lines.append("")
                continue
            lines.append(
                f"- {result['matching']} matching, {result['flagged']} flagged"
            )
            for item in result["preview"]:
                mark = "⚑ " if item["flagged"] else ""
                lines.append(
                    f"  - {mark}**{item['subject']}** — {item['from']}"
                    f"{' — ' + item['date'] if item['date'] else ''} "
                    f"(`{item['id']}`)"
                )
            if result["matching"] > len(result["preview"]):
                lines.append(
                    f"  - _…and {result['matching'] - len(result['preview'])} "
                    "more; use gmail_search_messages for the full list._"
                )
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"
    except Exception as exc:  # noqa: BLE001
        return _handle_error(exc)


def main() -> None:
    """Entry point: run the server over stdio for Claude Desktop."""
    mcp.run()


if __name__ == "__main__":
    main()
