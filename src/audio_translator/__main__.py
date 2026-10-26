"""Entry point: ``audio-translator`` launches the GUI, ``--cli`` the CLI."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--cli" in argv:
        argv.remove("--cli")
        from .cli import main as cli_main

        return cli_main(argv)
    from .gui import main as gui_main

    return gui_main(argv)


if __name__ == "__main__":
    sys.exit(main())
