"""Shared fixtures: a fake Gmail API backed by httpx.MockTransport.

The fake implements the subset of Gmail v1 the server uses, with real message
payloads (nested MIME, base64url bodies, quoted replies), so the tests exercise
genuine parsing rather than pre-digested dicts.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from gmail_mcp.auth import Token, TokenManager
from gmail_mcp.config import load_config
from gmail_mcp.gmail_client import GmailClient


def b64(text: str) -> str:
    """Encode a body the way Gmail does: base64url, padding stripped."""
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


def header_list(**kwargs: str) -> list[dict[str, str]]:
    return [{"name": k.replace("_", "-").title(), "value": v} for k, v in kwargs.items()]


PLAIN_MESSAGE: dict[str, Any] = {
    "id": "msg-plain",
    "threadId": "thread-1",
    "internalDate": "1770000000000",
    "snippet": "Quarterly numbers are in",
    "labelIds": ["INBOX", "UNREAD", "IMPORTANT"],
    "payload": {
        "mimeType": "text/plain",
        "headers": header_list(
            From="Alice <alice@example.com>",
            To="work@example.com",
            Subject="Q3 numbers",
            Date="Mon, 02 Feb 2026 09:15:00 +0000",
            Message_ID="<abc123@example.com>",
        ),
        "body": {
            "data": b64(
                "Quarterly numbers are in and we beat plan by 12%.\n"
                "\n"
                "On Sun, 01 Feb 2026, Bob <bob@example.com> wrote:\n"
                "> When will the numbers land?\n"
            )
        },
    },
}

MULTIPART_MESSAGE: dict[str, Any] = {
    "id": "msg-multipart",
    "threadId": "thread-2",
    "internalDate": "1770100000000",
    "snippet": "Invoice attached",
    "labelIds": ["INBOX"],
    "payload": {
        "mimeType": "multipart/mixed",
        "headers": header_list(
            From="Billing <billing@vendor.com>",
            To="billing@example.com",
            Cc="cfo@example.com",
            Subject="Invoice 4471",
            Date="Tue, 03 Feb 2026 14:00:00 +0000",
        ),
        "parts": [
            {
                "mimeType": "multipart/alternative",
                "body": {},
                "parts": [
                    {
                        "mimeType": "text/plain",
                        "body": {"data": b64("Invoice 4471 is attached. Due in 30 days.")},
                    },
                    {
                        "mimeType": "text/html",
                        "body": {"data": b64("<p>Invoice 4471 is attached.</p>")},
                    },
                ],
            },
            {
                "mimeType": "application/pdf",
                "filename": "invoice.pdf",
                "body": {"attachmentId": "att-1", "size": 51200},
            },
        ],
    },
}

HTML_ONLY_MESSAGE: dict[str, Any] = {
    "id": "msg-html",
    "threadId": "thread-3",
    "internalDate": "1770200000000",
    "snippet": "Your statement is ready",
    "labelIds": ["INBOX", "UNREAD"],
    "payload": {
        "mimeType": "text/html",
        "headers": header_list(
            From="Bank <no-reply@bank.com>",
            To="personal@example.com",
            Subject="Statement ready",
            Date="Wed, 04 Feb 2026 08:00:00 +0000",
        ),
        "body": {
            "data": b64(
                "<html><head><style>p{color:red}</style></head><body>"
                "<p>Your statement is ready.</p><p>Balance: $1,204.55</p>"
                "</body></html>"
            )
        },
    },
}

ALL_MESSAGES = {
    m["id"]: m for m in (PLAIN_MESSAGE, MULTIPART_MESSAGE, HTML_ONLY_MESSAGE)
}

LABELS = [
    {"id": "INBOX", "name": "INBOX", "type": "system"},
    {"id": "UNREAD", "name": "UNREAD", "type": "system"},
    {"id": "Label_12", "name": "Receipts", "type": "user"},
]


class FakeGmail:
    """Records requests and serves canned Gmail responses."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.sent: list[dict[str, Any]] = []
        self.drafts: list[dict[str, Any]] = []
        self.modifications: list[dict[str, Any]] = []
        self.refresh_count = 0
        self.sent_drafts: list[str] = []
        # Per-path failure injection: path suffix -> status code (always fails).
        self.fail_with: dict[str, int] = {}
        # Per-path transient failure: path suffix -> remaining failures before
        # the call starts succeeding. Used to test retry recovery.
        self.fail_times: dict[str, int] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path

        # Failure injection is checked first so that /token can be failed too,
        # which is how a revoked refresh token is simulated.
        for suffix, status in self.fail_with.items():
            if path.endswith(suffix):
                return httpx.Response(
                    status, json={"error": {"message": "injected failure"}}
                )

        for suffix, remaining in list(self.fail_times.items()):
            if remaining > 0 and path.endswith(suffix):
                self.fail_times[suffix] = remaining - 1
                return httpx.Response(
                    503, json={"error": {"message": "transient failure"}}
                )

        if path.endswith("/token"):
            self.refresh_count += 1
            return httpx.Response(
                200, json={"access_token": f"fresh-{self.refresh_count}", "expires_in": 3600}
            )

        if path.endswith("/profile"):
            return httpx.Response(
                200, json={"emailAddress": "user@example.com", "messagesTotal": 42}
            )

        if path.endswith("/labels"):
            return httpx.Response(200, json={"labels": LABELS})

        if path.endswith("/messages/send"):
            body = json.loads(request.content)
            self.sent.append(body)
            return httpx.Response(200, json={"id": "sent-1", "threadId": body.get("threadId", "thread-new")})

        if path.endswith("/drafts/send"):
            self.sent_drafts.append(json.loads(request.content)["id"])
            return httpx.Response(200, json={"id": "sent-draft-1", "threadId": "thread-9"})

        if path.endswith("/messages/batchModify"):
            self.modifications.append(json.loads(request.content))
            return httpx.Response(204)

        if path.endswith("/drafts"):
            if request.method == "GET":
                return httpx.Response(
                    200,
                    json={
                        "drafts": [
                            {"id": "draft-1", "message": {"id": "dmsg-1", "threadId": "t9"}}
                        ]
                    },
                )
            body = json.loads(request.content)
            self.drafts.append(body)
            return httpx.Response(
                200,
                json={"id": "draft-1", "message": {"id": "dmsg-1", "threadId": "thread-9"}},
            )

        if path.endswith("/modify"):
            body = json.loads(request.content)
            message_id = path.split("/messages/")[1].split("/")[0]
            self.modifications.append({"id": message_id, **body})
            labels = ["INBOX"] + list(body.get("addLabelIds", []))
            return httpx.Response(200, json={"id": message_id, "labelIds": labels})

        if "/threads/" in path:
            return httpx.Response(
                200,
                json={
                    "id": "thread-1",
                    "messages": [PLAIN_MESSAGE, MULTIPART_MESSAGE],
                },
            )

        if "/messages/" in path:
            message_id = path.split("/messages/")[1]
            message = ALL_MESSAGES.get(message_id)
            if message is None:
                return httpx.Response(
                    404, json={"error": {"message": "Requested entity was not found."}}
                )
            return httpx.Response(200, json=message)

        if path.endswith("/messages"):
            query = request.url.params.get("q", "")
            limit = int(request.url.params.get("maxResults", 20))
            ids = list(ALL_MESSAGES)
            if "is:unread" in query:
                ids = [
                    mid
                    for mid, m in ALL_MESSAGES.items()
                    if "UNREAD" in m["labelIds"]
                ]
            payload: dict[str, Any] = {
                "messages": [{"id": mid} for mid in ids[:limit]]
            }
            if len(ids) > limit:
                payload["nextPageToken"] = "page-2"
            return httpx.Response(200, json=payload)

        return httpx.Response(404, json={"error": {"message": f"unhandled {path}"}})


class MemoryTokenStore:
    """In-memory token store for tests."""

    def __init__(self, tokens: dict[str, Token] | None = None) -> None:
        self.tokens: dict[str, Token] = dict(tokens or {})

    def load(self, account: str) -> Token | None:
        return self.tokens.get(account)

    def save(self, account: str, token: Token) -> None:
        self.tokens[account] = token

    def delete(self, account: str) -> None:
        self.tokens.pop(account, None)


@pytest.fixture
def fake_gmail() -> FakeGmail:
    return FakeGmail()


@pytest.fixture
def http_client(fake_gmail: FakeGmail) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(fake_gmail.handler))


@pytest.fixture
def token_store() -> MemoryTokenStore:
    future = time.time() + 3600
    return MemoryTokenStore(
        {
            "personal": Token("tok-personal", "refresh-personal", future),
            "work-main": Token("tok-work", "refresh-work", future),
        }
    )


@pytest.fixture
def gmail_client(
    http_client: httpx.AsyncClient, token_store: MemoryTokenStore
) -> GmailClient:
    from gmail_mcp.auth import OAuthClient

    manager = TokenManager(
        OAuthClient("client-id", "client-secret"), token_store, http=http_client
    )

    async def no_sleep(_seconds: float) -> None:
        """Collapse retry backoff so the suite stays fast and deterministic."""
        return None

    return GmailClient(
        token_manager=manager,
        http=http_client,
        _sleep=no_sleep,
        _random=lambda: 0.5,
    )


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """A valid accounts.json plus a matching client_secret.json."""
    (tmp_path / "client_secret.json").write_text(
        json.dumps(
            {"installed": {"client_id": "cid.apps.googleusercontent.com", "client_secret": "csecret"}}
        ),
        encoding="utf-8",
    )
    path = tmp_path / "accounts.json"
    path.write_text(
        json.dumps(
            {
                "client_secrets_file": "client_secret.json",
                "accounts": [
                    {"name": "personal", "email": "me@gmail.com", "description": "Personal"},
                    {"name": "work-main", "email": "me@corp.com", "description": "Work"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def config(config_file: Path):
    return load_config(config_file)
