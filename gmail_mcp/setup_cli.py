"""Setup CLI: create the config and authenticate each Gmail account.

Run once per account. Each ``login`` opens a browser, the user picks the Google
account and grants access, and the resulting refresh token is stored. After
that the server runs hands-off -- access tokens are renewed automatically.

    gmail-mcp-setup init                 # write a starter accounts.json
    gmail-mcp-setup login work-main      # browser sign-in for one account
    gmail-mcp-setup login --all          # walk through every unauthenticated one
    gmail-mcp-setup status               # who is authenticated
    gmail-mcp-setup logout work-main     # forget one account's token
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .auth import Token, default_token_store
from .config import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_CONFIG_PATH,
    SCOPES,
    ConfigError,
    load_config,
)

STARTER_CONFIG = {
    "client_secrets_file": "client_secret.json",
    "accounts": [
        {"name": "personal", "email": "", "description": "Personal mail"},
        {"name": "work-main", "email": "", "description": "Primary business account"},
        {"name": "work-sales", "email": "", "description": "Sales and inbound leads"},
        {"name": "work-billing", "email": "", "description": "Invoices and billing"},
        {"name": "work-support", "email": "", "description": "Customer support"},
    ],
}


def cmd_init(args: argparse.Namespace) -> int:
    """Write a starter accounts.json if one does not already exist."""
    path = Path(args.config).expanduser() if args.config else DEFAULT_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not args.force:
        print(f"Config already exists at {path}. Use --force to overwrite.")
        return 1
    path.write_text(json.dumps(STARTER_CONFIG, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote starter config to {path}")
    print()
    print("Next steps:")
    print(f"  1. Edit {path} and set each account's name/email/description.")
    print(
        f"  2. Save your Google Cloud OAuth client file (Desktop app) to "
        f"{path.parent / 'client_secret.json'}"
    )
    print("  3. Run: gmail-mcp-setup login --all")
    return 0


def _authenticate(account_name: str, secrets_file: Path, config_dir: Path) -> bool:
    """Run the browser OAuth flow for one account and store the token."""
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print(
            "google-auth-oauthlib is not installed. Run: "
            "pip install google-auth-oauthlib",
            file=sys.stderr,
        )
        return False

    if not secrets_file.exists():
        print(f"OAuth client file not found: {secrets_file}", file=sys.stderr)
        print(
            "Download it from Google Cloud Console > APIs & Services > "
            "Credentials (create an OAuth client of type 'Desktop app').",
            file=sys.stderr,
        )
        return False

    print(f"\n--- Authenticating '{account_name}' ---")
    print("A browser window will open. Sign in as the Gmail account you want")
    print(f"mapped to the alias '{account_name}', then grant access.")

    flow = InstalledAppFlow.from_client_secrets_file(str(secrets_file), SCOPES)
    # access_type=offline + prompt=consent is what makes Google return a
    # refresh token every time, including on re-authentication.
    credentials = flow.run_local_server(
        port=0, access_type="offline", prompt="consent"
    )

    if not credentials.refresh_token:
        print(
            "Google did not return a refresh token. Revoke this app's access at "
            "https://myaccount.google.com/permissions and try again.",
            file=sys.stderr,
        )
        return False

    expires_at = (
        credentials.expiry.timestamp() if credentials.expiry else time.time() + 3600
    )
    store = default_token_store(config_dir)
    store.save(
        account_name,
        Token(
            access_token=credentials.token or "",
            refresh_token=credentials.refresh_token,
            expires_at=expires_at,
            scopes=tuple(credentials.scopes or SCOPES),
        ),
    )
    print(f"Stored credentials for '{account_name}'.")
    return True


def cmd_login(args: argparse.Namespace) -> int:
    """Authenticate one account, or every unauthenticated account."""
    try:
        config = load_config(Path(args.config).expanduser() if args.config else None)
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    store = default_token_store(config.config_dir)

    if args.all:
        targets = [
            a.name
            for a in config.accounts
            if args.reauth or store.load(a.name) is None
        ]
        if not targets:
            print("All accounts are already authenticated. Use --reauth to redo them.")
            return 0
    else:
        if not args.account:
            print("Specify an account name, or use --all.", file=sys.stderr)
            return 1
        try:
            targets = [config.get(args.account).name]
        except ConfigError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    failures = 0
    for name in targets:
        if not _authenticate(name, config.client_secrets_file, config.config_dir):
            failures += 1

    print()
    print(f"Done. {len(targets) - failures}/{len(targets)} account(s) authenticated.")
    return 1 if failures else 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show which accounts have stored credentials."""
    try:
        config = load_config(Path(args.config).expanduser() if args.config else None)
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    store = default_token_store(config.config_dir)
    backend = type(store).__name__
    print(f"Config:        {config.config_dir}")
    print(f"OAuth client:  {config.client_secrets_file}")
    print(f"Token storage: {backend}")
    print()
    for account in config.accounts:
        token = store.load(account.name)
        state = "authenticated" if token else "NOT authenticated"
        email = f" <{account.email}>" if account.email else ""
        print(f"  {account.name:<16}{email:<32} {state}")
    return 0


def cmd_logout(args: argparse.Namespace) -> int:
    """Delete an account's stored token."""
    try:
        config = load_config(Path(args.config).expanduser() if args.config else None)
        account = config.get(args.account)
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    default_token_store(config.config_dir).delete(account.name)
    print(f"Removed stored credentials for '{account.name}'.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gmail-mcp-setup",
        description="Set up and authenticate accounts for the Gmail MCP server.",
    )
    parser.add_argument(
        "--config",
        help=f"Path to accounts.json (default: {DEFAULT_CONFIG_PATH})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Write a starter accounts.json")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing config")
    p_init.set_defaults(func=cmd_init)

    p_login = sub.add_parser("login", help="Authenticate an account in the browser")
    p_login.add_argument("account", nargs="?", help="Account alias from the config")
    p_login.add_argument("--all", action="store_true", help="Authenticate every account")
    p_login.add_argument(
        "--reauth",
        action="store_true",
        help="With --all, redo accounts that are already authenticated",
    )
    p_login.set_defaults(func=cmd_login)

    p_status = sub.add_parser("status", help="Show authentication status")
    p_status.set_defaults(func=cmd_status)

    p_logout = sub.add_parser("logout", help="Forget one account's stored token")
    p_logout.add_argument("account", help="Account alias from the config")
    p_logout.set_defaults(func=cmd_logout)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
