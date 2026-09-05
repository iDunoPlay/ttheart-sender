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
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent
NAME = "ttheart-sender"


def _version() -> str:
    """The version, read as text rather than imported.

    ``ttheart_sender/version.py`` is deliberately import-free so setuptools can
    read the literal without executing it, and this reads it the same way --
    importing the package here would pull in its dependencies just to build.
    """
    src = (ROOT / "ttheart_sender" / "version.py").read_text(encoding="utf-8")
    m = re.search(r"""^__version__\s*=\s*['"]([^'"]+)['"]""", src, re.M)
    return m.group(1) if m else "unknown"


#: What a one-file build is called. The version is in the FILE NAME rather than
#: only inside the binary, because a lone .exe is the thing that gets copied to
#: another machine -- and two of them side by side in a downloads folder are
#: otherwise indistinguishable. That is not hypothetical here: rounds were
#: collected on a build that predated a fix, and nothing about the file said so.
ONEFILE_NAME = f"{NAME}_v{_version()}"
ENTRY = ROOT / "ttheart_tray.py"
ICON = ROOT / "ttheart_sender" / "tray" / "assets" / "tray-running.ico"

#: Copied beside the .exe after the build, and bundled inside it. These are the
#: files a user is expected to tweak.
DATA_FILES = ("config.yaml",)

#: ...and the directories. `models` is here because a flow can name one:
#: `reject_model: models/reject.onnx` is a path the app has to be able to
#: FIND, and a build without it produced the worst failure this project has
#: had -- the tray died the moment a round started, silently, because the
#: loader could not open the file and said so by raising SystemExit in a
#: windowed process with no stderr. Both halves are fixed; this is the half
#: that means the file is actually there.
DATA_DIRS = ("flows", "templates", "models")

#: What of `models/` the RUNTIME actually reads: an ONNX graph through
#: `cv2.dnn`, and the `.json` beside it carrying the class order and the
#: normalisation. `.pt` files are torch checkpoints kept so a model can be
#: retrained or re-exported, and torch is not a runtime dependency at all --
#: shipping them put 12MB of dead weight in a file that gets copied to another
#: machine by hand.
MODEL_SUFFIXES = (".onnx", ".json")

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
        ONEFILE_NAME if onefile else NAME,
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
        if not path.exists():
            continue
        if name == "models":
            # File by file, so the training checkpoints stay behind.
            for f in sorted(path.iterdir()):
                if f.is_file() and f.suffix in MODEL_SUFFIXES:
                    command += _add_data(f, name)
        else:
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
        shutil.move(str(ROOT / "dist" / f"{ONEFILE_NAME}.exe"),
                    str(target / f"{ONEFILE_NAME}.exe"))
    # A one-file build is meant to be one file, so the editable copies are
    # opt-in there -- emitting them by default would recreate the very folder
    # the user asked to get rid of.
    #
    # BUT copies that already exist are always refreshed, whether they were
    # asked for this time or not. Editable copies beside the .exe WIN over the
    # bundle (see `default_app_root()`), so a stale `flows/` silently replaces
    # every flow in the build -- and one did, for a month. Three separate
    # experiments were selected in the panel, played, and came back as
    # baseline rounds because the app was reading August's flows.
    if with_data or not onefile or (target / DATA_DIRS[0]).exists():
        _copy_editable_data(target)
    _report(target, onefile, with_data)
    return 0


def _copy_editable_data(target: Path) -> None:
    """Put the editable copies of config/flows/templates/models beside the .exe.

    Directories are REPLACED, not merged. `copytree(dirs_exist_ok=True)` leaves
    behind any file the source no longer has, and a flow that was renamed or
    split would keep running from its old copy -- which is the same failure as
    the stale directory this exists to prevent, one file down.

    `config.yaml` is copied only if it is not already there: it is the one file
    a user is expected to have edited, and its presence is also what makes this
    directory the app root at all.
    """
    target.mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        source, dest = ROOT / name, target / name
        if source.exists() and not dest.exists():
            shutil.copy2(source, dest)
    for name in DATA_DIRS:
        source = ROOT / name
        if not source.exists():
            continue
        dest = target / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest)


def _report(target: Path, onefile: bool, with_data: bool) -> None:
    exe = target / f"{ONEFILE_NAME if onefile else NAME}.exe"
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
