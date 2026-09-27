"""Strip agent-inserted trailers from a commit message file.

Removes ``Co-authored-by:`` and ``Made-with:`` trailer lines (case
insensitive) and trailing blank lines. Run from the pre-commit ``commit-msg``
hook so local agent tools cannot add noisy metadata to project history.
"""

import re
import sys
from pathlib import Path


TRAILER = re.compile(r"^(co-authored-by|made-with):\s", re.IGNORECASE)


def strip_trailers(message: str) -> str:
    """Return ``message`` without agent trailers or trailing blank lines.

    Args:
        message: Full commit message text.

    Returns:
        The cleaned message, ending with a single newline when non-empty.
    """
    lines = [line for line in message.splitlines() if not TRAILER.match(line)]
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines) + "\n" if lines else ""


def main() -> None:
    """Rewrite the commit message file given as the first argument."""
    path = Path(sys.argv[1])
    if path.is_file():
        path.write_text(strip_trailers(path.read_text()))


if __name__ == "__main__":
    main()
