from __future__ import annotations

import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE_VAR = "SANE_DS_ENV_FILE"
_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DOUBLE_QUOTE_ESCAPES = {
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "\\": "\\",
    '"': '"',
}


def _resolve_env_path(env_path: str | os.PathLike[str] | None = None) -> Path:
    raw_path = env_path or os.environ.get(_ENV_FILE_VAR) or ".env"
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _strip_inline_comment(value: str) -> str:
    for idx, char in enumerate(value):
        if char == "#" and (idx == 0 or value[idx - 1].isspace()):
            return value[:idx].rstrip()
    return value.strip()


def _parse_quoted_value(value: str) -> str:
    quote = value[0]
    chars: list[str] = []
    escaped = False

    for char in value[1:]:
        if quote == '"' and escaped:
            chars.append(_DOUBLE_QUOTE_ESCAPES.get(char, char))
            escaped = False
            continue
        if quote == '"' and char == "\\":
            escaped = True
            continue
        if char == quote:
            return "".join(chars)
        chars.append(char)

    if escaped:
        chars.append("\\")
    return "".join(chars)


def _parse_dotenv_line(line: str, path: Path, line_no: int) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    if stripped.startswith("export "):
        stripped = stripped[7:].lstrip()

    key, separator, value = stripped.partition("=")
    if not separator:
        return None

    key = key.strip()
    if not _KEY_RE.match(key):
        raise ValueError(f"Invalid .env key at {path}:{line_no}: {key!r}")

    value = value.strip()
    if value.startswith(("'", '"')):
        return key, _parse_quoted_value(value)
    return key, _strip_inline_comment(value)


def load_dotenv(
    env_path: str | os.PathLike[str] | None = None,
    *,
    override: bool = False,
) -> dict[str, str]:
    """Load environment variables from a project .env file.

    Existing process environment values win by default. Relative paths are
    resolved from the project root so shell scripts and Python imports behave
    consistently.
    """
    path = _resolve_env_path(env_path)
    if not path.exists():
        return {}

    loaded: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parsed = _parse_dotenv_line(line, path, line_no)
            if parsed is None:
                continue
            key, value = parsed
            if override or key not in os.environ:
                os.environ[key] = value
                loaded[key] = value

    return loaded
