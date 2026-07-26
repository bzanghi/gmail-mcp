#!/usr/bin/env python3
"""Exercise every tool against a real Gmail account and report pass/fail.

DEVELOPMENT / VALIDATION ONLY -- not part of the shipped server.

Read-only by default: it searches, reads, and summarises, but does not send
mail or change labels. Pass --write to additionally create a draft (in the
account's own Drafts folder) and round-trip a label change on a message the
script itself just drafted. Nothing is ever deleted.

    python scripts/live_check.py personal
    python scripts/live_check.py personal --write
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gmail_mcp import server  # noqa: E402

PASS = "PASS"
FAIL = "FAIL"

results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    results.append((PASS if ok else FAIL, name, detail))
    marker = "✓" if ok else "✗"
    print(f"  {marker} {name}: {detail}")


async def call(tool: str, **arguments: Any) -> str:
    """Invoke a tool the way Claude Desktop would."""
    from mcp.server.fastmcp.exceptions import ToolError

    try:
        result = await server.mcp.call_tool(tool, {"params": arguments})
    except ToolError as exc:
        return f"Error: {exc}"
    content = result[0] if isinstance(result, tuple) else result
    blocks = content if isinstance(content, list) else [content]
    return "\n".join(getattr(b, "text", str(b)) for b in blocks)


def is_error(output: str) -> bool:
    return output.lstrip().startswith("Error:")


async def main(account: str, do_writes: bool) -> int:
    print(f"\nLive check against account '{account}'\n" + "=" * 50)

    print("\n[1] Accounts and authentication")
    out = await call("gmail_list_accounts")
    if is_error(out):
        record("list_accounts", False, out.strip()[:200])
        print("\nCannot continue without a working config.")
        return 1
    accounts = json.loads(out)["accounts"]
    authed = [a for a in accounts if a["authenticated"]]
    record(
        "list_accounts",
        bool(authed),
        f"{len(accounts)} configured, {len(authed)} authenticated",
    )

    print("\n[2] Search (this is the first real Gmail API call)")
    out = await call(
        "gmail_search_messages",
        account=account,
        query="in:inbox",
        limit=5,
        response_format="json",
    )
    if is_error(out):
        record("search_messages", False, out.strip()[:300])
        print("\nGmail API is not reachable; stopping.")
        return 1
    data = json.loads(out)
    messages = data["messages"]
    record("search_messages", True, f"{data['count']} message(s) returned")

    if not messages:
        print("\nInbox is empty — read/thread checks need at least one message.")
        return 0

    first = messages[0]
    for field in ("id", "thread_id", "from", "subject", "date"):
        record(
            f"search field '{field}'",
            bool(str(first.get(field, "")).strip()),
            repr(str(first.get(field, ""))[:60]),
        )

    print("\n[3] Read a single message")
    out = await call(
        "gmail_read_message",
        account=account,
        message_id=first["id"],
        body_chars=400,
        response_format="json",
    )
    if is_error(out):
        record("read_message", False, out.strip()[:300])
    else:
        msg = json.loads(out)["messages"][0]
        body = msg.get("body", "")
        record("read_message", True, f"subject={msg['subject'][:40]!r}")
        record(
            "body extracted",
            bool(body.strip()),
            f"{len(body)} chars, truncated={msg.get('body_truncated')}",
        )
        record("no raw HTML in body", "<html" not in body.lower(), "checked for tags")

    print("\n[4] Read the thread")
    out = await call(
        "gmail_read_thread",
        account=account,
        thread_id=first["thread_id"],
        response_format="json",
    )
    if is_error(out):
        record("read_thread", False, out.strip()[:300])
    else:
        record("read_thread", True, f"{json.loads(out)['count']} message(s) in thread")

    print("\n[5] Labels")
    out = await call("gmail_list_labels", account=account)
    if is_error(out):
        record("list_labels", False, out.strip()[:300])
    else:
        labels = json.loads(out)["labels"]
        names = {label["name"] for label in labels}
        record("list_labels", True, f"{len(labels)} labels")
        record("system labels present", "INBOX" in names, "INBOX found" if "INBOX" in names else "INBOX missing")

    print("\n[6] Multi-inbox sweep (the scheduled tool)")
    out = await call("gmail_check_inboxes", response_format="json")
    if is_error(out):
        record("check_inboxes", False, out.strip()[:300])
    else:
        sweep = json.loads(out)
        record(
            "check_inboxes",
            True,
            f"{sweep['total_matching']} unread, {sweep['flagged_count']} flagged, "
            f"{len(sweep['accounts'])} account(s)",
        )
        errors = [a for a in sweep["accounts"] if a["error"]]
        record(
            "no per-account errors",
            not errors,
            "all clean" if not errors else f"{len(errors)} failed: {errors[0]['error'][:120]}",
        )

    markdown = await call("gmail_check_inboxes")
    record("sweep output stays small", len(markdown) < 4000, f"{len(markdown)} chars")

    print("\n[7] Error handling against the live API")
    out = await call("gmail_read_message", account=account, message_id="definitely-not-a-real-id")
    record("bad message id is handled", is_error(out), out.strip()[:120])

    out = await call("gmail_search_messages", account="no-such-account")
    record("unknown account is handled", is_error(out), out.strip()[:120])

    if do_writes:
        print("\n[8] Write operations")
        out = await call(
            "gmail_create_draft",
            account=account,
            to=["nobody@example.invalid"],
            subject="gmail-mcp live check (safe to delete)",
            body="Draft created by the gmail-mcp live check. Not sent.",
        )
        if is_error(out):
            record("create_draft", False, out.strip()[:300])
        else:
            draft = json.loads(out)
            record("create_draft", True, f"draft_id={draft['draft_id']}")

            if draft.get("message_id"):
                out = await call(
                    "gmail_modify_labels",
                    account=account,
                    message_id=draft["message_id"],
                    add_labels=["STARRED"],
                )
                starred_ok = not is_error(out) and "STARRED" in out
                record("modify_labels add", starred_ok, out.strip()[:120])

                out = await call(
                    "gmail_modify_labels",
                    account=account,
                    message_id=draft["message_id"],
                    remove_labels=["STARRED"],
                )
                record("modify_labels remove", not is_error(out), out.strip()[:120])
    else:
        print("\n[8] Write operations skipped (pass --write to include them)")

    passed = sum(1 for status, _, _ in results if status == PASS)
    failed = sum(1 for status, _, _ in results if status == FAIL)
    print("\n" + "=" * 50)
    print(f"{passed} passed, {failed} failed")
    if failed:
        print("\nFailures:")
        for status, name, detail in results:
            if status == FAIL:
                print(f"  - {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    raise SystemExit(
        asyncio.run(main(args[0] if args else "personal", "--write" in sys.argv))
    )
