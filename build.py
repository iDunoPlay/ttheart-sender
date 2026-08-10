"""Build the Windows .exe with PyInstaller.

    python build.py                 one folder in dist/ttheart-sender
    python build.py --onefile       one self-contained .exe, no folder needed
    python build.py --console       keep a console window (for debugging)

``config.yaml``, ``flows/`` and ``templates/`` are always baked into the bundle,
so a lone .exe runs with no files beside it. The folder build also drops
editable copies next to the .exe (``--with-data`` does the same for --onefile);
those copies win when present. See ``default_app_root()`` in
ttheart_sender/config.py for the exact rule.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent
NAME = "ttheart-sender"
ENTRY = ROOT / "ttheart_tray.py"
ICON = ROOT / "ttheart_sender" / "tray" / "assets" / "tray-running.ico"

#: Copied beside the .exe after the build, and bundled inside it. These are the
#: files a user is expected to tweak.
DATA_FILES = ("config.yaml",)
DATA_DIRS = ("flows", "templates")

#: pywin32 pulls this in lazily, so PyInstaller's analysis never sees it.
HIDDEN_IMPORTS = ("win32timezone",)

#: Dev-only weight that would otherwise be dragged in by transitive imports.
EXCLUDES = ("pytest", "_pytest", "setuptools", "pip", "numpy.f2py")


def _add_data(source: Path, destination: str) -> List[str]:
    # PyInstaller's source/destination separator is ';' on Windows.
    return ["--add-data", f"{source};{destination}"]


def build(*, onefile: bool, console: bool, clean: bool, with_data: bool = False) -> int:
    if not ENTRY.exists():
        raise SystemExit(f"Entry point missing: {ENTRY}")

    command: List[str] = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--name",
        NAME,
        "--onefile" if onefile else "--onedir",
        "--console" if console else "--windowed",
        "--distpath",
        str(ROOT / "dist"),
        "--workpath",
        str(ROOT / "build"),
        "--specpath",
        str(ROOT / "build"),
    ]
    if clean:
        command.append("--clean")
    if ICON.exists():
        command += ["--icon", str(ICON)]

    # Tray artwork is a package resource: it must live inside the bundle.
    assets = ROOT / "ttheart_sender" / "tray" / "assets"
    if assets.exists():
        command += _add_data(assets, "ttheart_sender/tray/assets")

    for name in DATA_FILES:
        path = ROOT / name
        if path.exists():
            command += _add_data(path, ".")
    for name in DATA_DIRS:
        path = ROOT / name
        if path.exists():
            command += _add_data(path, name)

    for module in HIDDEN_IMPORTS:
        command += ["--hidden-import", module]
    for module in EXCLUDES:
        command += ["--exclude-module", module]

    command.append(str(ENTRY))

    print("Running:", " ".join(command), "\n", flush=True)
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        return result.returncode

    target = (ROOT / "dist" / f"{NAME}-portable") if onefile else (ROOT / "dist" / NAME)
    target.mkdir(parents=True, exist_ok=True)
    if onefile:
        shutil.move(str(ROOT / "dist" / f"{NAME}.exe"), str(target / f"{NAME}.exe"))
    # A one-file build is meant to be one file, so the editable copies are
    # opt-in there -- emitting them by default would recreate the very folder
    # the user asked to get rid of.
    if with_data or not onefile:
        _copy_editable_data(target)
    _report(target, onefile, with_data)
    return 0


def _copy_editable_data(target: Path) -> None:
    """Put the editable copies of config/flows/templates beside the .exe."""
    target.mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        source = ROOT / name
        if source.exists():
            shutil.copy2(source, target / name)
    for name in DATA_DIRS:
        source = ROOT / name
        if not source.exists():
            continue
        shutil.copytree(source, target / name, dirs_exist_ok=True)


def _report(target: Path, onefile: bool, with_data: bool) -> None:
    exe = target / f"{NAME}.exe"
    size = exe.stat().st_size / (1024 * 1024) if exe.exists() else 0.0
    print("\n" + "=" * 66)
    print(f"Built {exe}  ({size:.1f} MB)")
    if onefile and not with_data:
        print("Self-contained: copy that one file anywhere and run it.")
        print("Drop a config.yaml (plus flows\\ and templates\\) beside it to")
        print("override the built-in copies, or rebuild with --with-data to")
        print("get editable copies emitted for you.")
    elif onefile:
        print("Self-contained, but the config.yaml / flows\\ / templates\\ next")
        print("to it now take priority over the copies inside the .exe.")
    else:
        print(f"Ship the whole {target.name}\\ folder -- the .exe needs _internal\\.")
    print("Double-click it: the icon appears in the system tray (next to the")
    print("clock; you may need to expand the overflow arrow the first time).")
    print("=" * 66)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--onefile",
        action="store_true",
        help="Produce a single .exe (slower to start; it unpacks itself each launch)",
    )
    parser.add_argument(
        "--console", action="store_true", help="Keep a console window for log output"
    )
    parser.add_argument(
        "--with-data",
        action="store_true",
        help="With --onefile, also emit editable config/flows/templates beside the .exe",
    )
    parser.add_argument(
        "--no-clean", action="store_true", help="Reuse the previous build cache"
    )
    args = parser.parse_args()

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        raise SystemExit(
            "PyInstaller is not installed. Run:\n"
            f"    {sys.executable} -m pip install pyinstaller"
        )

    return build(
        onefile=args.onefile,
        console=args.console,
        clean=not args.no_clean,
        with_data=args.with_data,
    )


if __name__ == "__main__":
    raise SystemExit(main())
