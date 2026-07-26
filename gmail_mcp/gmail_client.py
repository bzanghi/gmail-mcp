"""Async Gmail REST API client.

Wraps the handful of Gmail v1 endpoints this server needs. Everything is
async and every request carries a per-account bearer token supplied by
:class:`~gmail_mcp.auth.TokenManager`.

``messages.list`` returns bare IDs, so listing is inherently N+1: one list call
plus one metadata fetch per message. Those fetches run concurrently under a
semaphore, which keeps a 50-message page well inside Gmail's per-user rate
limit while staying an order of magnitude faster than sequential fetching.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import httpx

API_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"

# Gmail allows 250 quota units/second/user; a metadata get costs 5. Sixteen
# concurrent fetches leaves ample headroom even with several accounts in
# flight at once.
MAX_CONCURRENT_FETCHES = 16

DEFAULT_TIMEOUT = 30.0

# Retry policy for transient failures (429, 5xx, timeouts, connection drops).
# Four attempts with jittered exponential backoff covers the rate-limit blips
# a five-account fan-out produces without making a genuinely broken call hang
# for a minute before reporting.
MAX_ATTEMPTS = 4
BASE_BACKOFF_SECONDS = 0.5
MAX_BACKOFF_SECONDS = 8.0

# messages.batchModify accepts up to 1000 IDs per call.
MAX_BATCH_MODIFY_IDS = 1000


def _path_segment(value: str) -> str:
    """Percent-encode a value for safe use as a single URL path segment.

    Message, thread, and label IDs reach this client from model-supplied tool
    arguments. Interpolating them into a URL path unescaped would let a value
    like ``../../settings/forwarding`` retarget the request to a different
    Gmail endpoint than the tool intends -- same host and same bearer token,
    different operation. Encoding ``/`` and ``.`` away removes that entirely.
    """
    return quote(str(value), safe="")


class GmailApiError(RuntimeError):
    """A Gmail API call failed. The message is written for the model to act on."""


def _describe_http_error(exc: httpx.HTTPStatusError, account: str) -> str:
    status = exc.response.status_code
    try:
        detail = exc.response.json().get("error", {}).get("message", "")
    except Exception:  # noqa: BLE001 - error bodies are not always JSON
        detail = ""
    suffix = f" Google said: {detail}" if detail else ""

    if status == 401:
        return (
            f"Authentication failed for account {account!r}. The token was "
            f"likely revoked; re-run 'gmail-mcp-setup login {account}'.{suffix}"
        )
    if status == 403:
        return (
            f"Permission denied for account {account!r}. Confirm the Gmail API "
            f"is enabled in the Google Cloud project and that this account "
            f"granted all requested scopes.{suffix}"
        )
    if status == 404:
        return (
            f"Not found in account {account!r}. The message, thread, or label "
            f"ID may be wrong, or it may belong to a different account.{suffix}"
        )
    if status == 429:
        return (
            f"Gmail rate limit hit for account {account!r}. Wait a few seconds "
            f"and retry, or request fewer messages.{suffix}"
        )
    if 500 <= status < 600:
        return (
            f"Gmail returned a server error ({status}) for account "
            f"{account!r}. This is transient; retry shortly.{suffix}"
        )
    return f"Gmail API request failed for account {account!r} ({status}).{suffix}"


@dataclass
class GmailClient:
    """Per-account Gmail API access.

    The same instance serves every account; the account alias is passed to each
    call so the correct bearer token is fetched and so errors name the account.

    ``sleep`` and ``random`` are injectable purely so tests can exercise the
    retry schedule without real delays or nondeterminism.
    """

    token_manager: Any  # gmail_mcp.auth.TokenManager
    http: httpx.AsyncClient | None = None
    max_attempts: int = MAX_ATTEMPTS
    _sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    _random: Callable[[], float] = random.random

    def _backoff_delay(self, attempt: int, response: httpx.Response | None) -> float:
        """Seconds to wait before retry ``attempt`` (1-based).

        Honours Gmail's ``Retry-After`` when present; otherwise exponential
        backoff with full jitter. Jitter matters here because a fan-out across
        five accounts fails in lockstep -- without it every account would
        retry at the same instant and re-trigger the same rate limit.
        """
        if response is not None:
            retry_after = response.headers.get("Retry-After", "")
            if retry_after.strip().isdigit():
                return min(float(retry_after), MAX_BACKOFF_SECONDS)
        window = min(BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)
        return self._random() * window

    async def _request(
        self,
        account: str,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make one Gmail API call, retrying transient failures.

        Retries on 429, 5xx, timeouts, and connection errors. Everything else
        -- 401, 403, 404, 400 -- fails immediately, because retrying an
        authorization or not-found error only wastes the user's time.

        The access token is fetched inside the loop so that a 401 arriving
        mid-retry picks up a freshly refreshed token on the next attempt.
        """
        url = f"{API_BASE}{path}"
        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            token = await self.token_manager.access_token(account)
            headers = {"Authorization": f"Bearer {token}"}
            response: httpx.Response | None = None

            try:
                if self.http is not None:
                    response = await self.http.request(
                        method, url, params=params, json=json_body, headers=headers
                    )
                else:
                    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                        response = await client.request(
                            method, url, params=params, json=json_body, headers=headers
                        )
                response.raise_for_status()
                return response.json() if response.content else {}

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status != 429 and not (500 <= status < 600):
                    raise GmailApiError(_describe_http_error(exc, account)) from exc
                last_error = exc
                response = exc.response

            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                response = None

            except httpx.HTTPError as exc:
                raise GmailApiError(
                    f"Network error talking to Gmail for account {account!r}: {exc}"
                ) from exc

            if attempt < self.max_attempts:
                await self._sleep(self._backoff_delay(attempt, response))

        # Exhausted every attempt; report the last failure with its own guidance.
        if isinstance(last_error, httpx.HTTPStatusError):
            raise GmailApiError(
                _describe_http_error(last_error, account)
                + f" (gave up after {self.max_attempts} attempts)"
            ) from last_error
        raise GmailApiError(
            f"Gmail was unreachable for account {account!r} after "
            f"{self.max_attempts} attempts: {last_error}. Check your network, "
            "then retry."
        ) from last_error

    async def get_profile(self, account: str) -> dict[str, Any]:
        """Return the account profile (email address, total message count)."""
        return await self._request(account, "GET", "/profile")

    async def list_message_ids(
        self,
        account: str,
        query: str = "",
        limit: int = 20,
        page_token: str | None = None,
        label_ids: list[str] | None = None,
    ) -> tuple[list[str], str | None]:
        """Run a Gmail search and return ``(message_ids, next_page_token)``."""
        params: dict[str, Any] = {"maxResults": limit}
        if query:
            params["q"] = query
        if page_token:
            params["pageToken"] = page_token
        if label_ids:
            params["labelIds"] = label_ids

        data = await self._request(account, "GET", "/messages", params=params)
        ids = [m["id"] for m in data.get("messages", [])]
        return ids, data.get("nextPageToken")

    async def get_message(
        self, account: str, message_id: str, fmt: str = "metadata"
    ) -> dict[str, Any]:
        """Fetch one message.

        Args:
            fmt: ``metadata`` for headers only (cheap, used for listings),
                ``full`` for the parsed payload including the body.
        """
        params: dict[str, Any] = {"format": fmt}
        if fmt == "metadata":
            # Ask only for the headers we actually render, which keeps
            # responses small on threads with long header chains.
            params["metadataHeaders"] = ["From", "To", "Cc", "Subject", "Date"]
        return await self._request(
            account, "GET", f"/messages/{_path_segment(message_id)}", params=params
        )

    async def get_messages(
        self, account: str, message_ids: list[str], fmt: str = "metadata"
    ) -> list[dict[str, Any]]:
        """Fetch many messages concurrently, preserving input order.

        Individual failures propagate: a partially-filled listing that silently
        drops messages would be worse than an explicit error.
        """
        if not message_ids:
            return []
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)

        async def fetch(mid: str) -> dict[str, Any]:
            async with semaphore:
                return await self.get_message(account, mid, fmt=fmt)

        return await asyncio.gather(*(fetch(mid) for mid in message_ids))

    async def get_thread(
        self, account: str, thread_id: str, fmt: str = "full"
    ) -> dict[str, Any]:
        """Fetch a full thread with all of its messages."""
        return await self._request(
            account,
            "GET",
            f"/threads/{_path_segment(thread_id)}",
            params={"format": fmt},
        )

    async def send_message(
        self, account: str, raw: str, thread_id: str | None = None
    ) -> dict[str, Any]:
        """Send a base64url-encoded RFC 2822 message."""
        body: dict[str, Any] = {"raw": raw}
        if thread_id:
            body["threadId"] = thread_id
        return await self._request(
            account, "POST", "/messages/send", json_body=body
        )

    async def create_draft(
        self, account: str, raw: str, thread_id: str | None = None
    ) -> dict[str, Any]:
        """Create a draft from a base64url-encoded RFC 2822 message."""
        message: dict[str, Any] = {"raw": raw}
        if thread_id:
            message["threadId"] = thread_id
        return await self._request(
            account, "POST", "/drafts", json_body={"message": message}
        )

    async def modify_message(
        self,
        account: str,
        message_id: str,
        add_label_ids: list[str] | None = None,
        remove_label_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Add and/or remove labels on a message."""
        body: dict[str, Any] = {}
        if add_label_ids:
            body["addLabelIds"] = add_label_ids
        if remove_label_ids:
            body["removeLabelIds"] = remove_label_ids
        return await self._request(
            account,
            "POST",
            f"/messages/{_path_segment(message_id)}/modify",
            json_body=body,
        )

    async def batch_modify_messages(
        self,
        account: str,
        message_ids: list[str],
        add_label_ids: list[str] | None = None,
        remove_label_ids: list[str] | None = None,
    ) -> int:
        """Apply the same label change to many messages in one call.

        ``messages.batchModify`` returns no body and costs a fraction of what
        N individual modify calls would. Chunked at the API's 1000-ID limit.

        Returns:
            The number of message IDs submitted.
        """
        if not message_ids:
            return 0
        body_base: dict[str, Any] = {}
        if add_label_ids:
            body_base["addLabelIds"] = add_label_ids
        if remove_label_ids:
            body_base["removeLabelIds"] = remove_label_ids

        for start in range(0, len(message_ids), MAX_BATCH_MODIFY_IDS):
            chunk = message_ids[start : start + MAX_BATCH_MODIFY_IDS]
            await self._request(
                account,
                "POST",
                "/messages/batchModify",
                json_body={**body_base, "ids": chunk},
            )
        return len(message_ids)

    async def list_drafts(
        self, account: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """List draft IDs and their underlying message IDs."""
        data = await self._request(
            account, "GET", "/drafts", params={"maxResults": limit}
        )
        return data.get("drafts", [])

    async def send_draft(self, account: str, draft_id: str) -> dict[str, Any]:
        """Send an existing draft by ID."""
        return await self._request(
            account, "POST", "/drafts/send", json_body={"id": draft_id}
        )

    async def list_labels(self, account: str) -> list[dict[str, Any]]:
        """List all labels, system and user-created."""
        data = await self._request(account, "GET", "/labels")
        return data.get("labels", [])
