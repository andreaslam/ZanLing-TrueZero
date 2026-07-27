"""Path helpers anchoring data folders (nets/, games/, hidden/, ...) to the project root.

The project root is three levels above this file:

    <root>/
        src/
            python/
                paths.py   <- here
        nets/
        games/
        hidden/
        python_client_games/
        ...

Use ``data_path("nets/tz_0.pt")`` anywhere so scripts work regardless of the
current working directory.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def data_path(rel: str) -> str:
    """Return an absolute path for ``rel`` relative to the project root."""
    return str(ROOT / rel)


def data_dir(rel: str) -> str:
    """Return an absolute path for a directory relative to the project root,
    creating it if it does not exist."""
    p = ROOT / rel
    p.mkdir(parents=True, exist_ok=True)
    return str(p)
