"""Entry point for running the pipeline as a script."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pain_intelligence.cli import app  # noqa: E402

if __name__ == "__main__":
    app()
