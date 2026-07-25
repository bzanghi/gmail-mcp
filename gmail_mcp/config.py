"""Account configuration for the Gmail MCP server.

The server is driven by a single JSON config file that names each Gmail account
with a stable, human-friendly alias (``personal``, ``work-main``, ...). Every
tool call takes one of those aliases, so the model never has to remember email
addresses and the user can rename an account without re-authenticating.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_DIR = Path(
    os.environ.get("GMAIL_MCP_HOME", Path.home() / ".gmail-mcp")
).expanduser()
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "accounts.json"

# Gmail scopes. ``gmail.modify`` covers read, label changes, and draft
# creation; ``gmail.send`` covers sending. Neither permits permanent deletion,
# which is deliberate -- the server has no way to destroy mail.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
]


class ConfigError(RuntimeError):
    """Raised when the accounts config is missing or malformed."""


@dataclass(frozen=True)
class Account:
    """A single Gmail account the server can act on.

    Attributes:
        name: Stable alias used in every tool call (e.g. ``work-sales``).
        email: The Gmail address, recorded at setup time for display only.
        description: Free-text hint shown to the model so it can pick the
            right account without asking (e.g. "invoices and vendor mail").
    """

    name: str
    email: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "email": self.email, "description": self.description}


@dataclass(frozen=True)
class Config:
    """Parsed server configuration."""

    accounts: tuple[Account, ...]
    client_secrets_file: Path
    config_dir: Path

    def account_names(self) -> list[str]:
        return [a.name for a in self.accounts]

    def get(self, name: str) -> Account:
        """Look up an account by alias.

        Raises:
            ConfigError: If no account matches, listing the valid aliases so
                the caller can correct itself without a second round trip.
        """
        for account in self.accounts:
            if account.name == name:
                return account
        raise ConfigError(
            f"Unknown account {name!r}. Configured accounts: "
            f"{', '.join(self.account_names()) or '(none)'}."
        )

    def resolve(self, names: list[str] | None) -> list[Account]:
        """Resolve a possibly-empty list of aliases to accounts.

        An empty or omitted list means "every configured account", which is
        what the periodic inbox check relies on.
        """
        if not names:
            return list(self.accounts)
        return [self.get(name) for name in names]


def _parse(raw: dict[str, Any], config_dir: Path) -> Config:
    accounts_raw = raw.get("accounts")
    if not isinstance(accounts_raw, list) or not accounts_raw:
        raise ConfigError(
            "Config must contain a non-empty 'accounts' array. "
            "See accounts.example.json for the expected shape."
        )

    accounts: list[Account] = []
    seen: set[str] = set()
    for entry in accounts_raw:
        if not isinstance(entry, dict) or not entry.get("name"):
            raise ConfigError(
                f"Each account needs a 'name' field; got: {entry!r}"
            )
        name = str(entry["name"]).strip()
        if name in seen:
            raise ConfigError(f"Duplicate account name {name!r} in config.")
        seen.add(name)
        accounts.append(
            Account(
                name=name,
                email=str(entry.get("email", "")).strip(),
                description=str(entry.get("description", "")).strip(),
            )
        )

    secrets = raw.get("client_secrets_file") or str(config_dir / "client_secret.json")
    secrets_path = Path(str(secrets)).expanduser()
    if not secrets_path.is_absolute():
        secrets_path = (config_dir / secrets_path).resolve()

    return Config(
        accounts=tuple(accounts),
        client_secrets_file=secrets_path,
        config_dir=config_dir,
    )


def load_config(path: Path | None = None) -> Config:
    """Load and validate the accounts config.

    Args:
        path: Optional explicit config path. Defaults to
            ``$GMAIL_MCP_HOME/accounts.json`` (``~/.gmail-mcp/accounts.json``).

    Returns:
        The parsed :class:`Config`.

    Raises:
        ConfigError: If the file is missing or malformed. The message names the
            path so setup problems are self-explanatory in Claude Desktop's log.
    """
    config_path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise ConfigError(
            f"No config found at {config_path}. Run 'gmail-mcp-setup init' to "
            "create one, then 'gmail-mcp-setup login' for each account."
        )
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Config at {config_path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Config at {config_path} must be a JSON object.")
    return _parse(raw, config_path.parent)
