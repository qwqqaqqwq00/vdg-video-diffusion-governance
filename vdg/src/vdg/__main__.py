"""Enable ``python -m vdg`` as an alias for the ``vdg`` console script.

The canonical entry point is declared in ``pyproject.toml`` as
``vdg = "vdg.cli:main"``. This shim mirrors it so the package is also runnable
via ``python -m vdg <command>`` without depending on the install having put
``vdg`` on PATH -- useful in CI, fresh checkouts, and the docs' quickstart.
"""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
