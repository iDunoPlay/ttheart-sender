"""Finding, fetching and swapping in a newer build of the .exe.

The panel's Auto Update tick box is the front-end for this module; everything
here is deliberately GUI-free and thread-free so it can be exercised without a
message loop -- see :mod:`.tray.updater` for the worker that drives it.

Releases are read from the GitHub REST API with :mod:`urllib` rather than
``requests``: one HTTPS GET and a JSON parse is not worth another dependency in
a bundle that is already 76MB. The published asset is a single self-contained
.exe, so an update is "download the new file, put it where the old one is,
restart" -- which on Windows means renaming the running .exe out of the way
rather than overwriting it, because a running image cannot be written to but
can be renamed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import total_ordering
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

from .version import __version__

log = logging.getLogger(__name__)

#: Where releases are published, as ``owner/name``. Overridable from
#: config.yaml so a fork can point the check at its own repository.
DEFAULT_REPO = "iDunoPlay/ttheart-sender"
API_ROOT = "https://api.github.com"
#: How many releases to look at. The list comes back newest-first and only the
#: highest version in it matters -- ten leaves room for a run of pre-releases
#: sitting on top of the last stable one.
RELEASE_PAGE_SIZE = 10
#: GitHub rejects API calls without one, and a named agent makes the traffic
#: identifiable in the repository's own logs.
USER_AGENT = f"ttheart-sender/{__version__}"
#: The release asset an update is applied from.
ASSET_NAME = "ttheart-sender.exe"

#: Suffixes used while swapping the .exe. The ``.new`` file is a build waiting
#: to be installed; the ``.old`` one is the build being replaced -- it cannot
#: be deleted until the process using it exits, so the helper script does it.
NEW_SUFFIX = ".new"
OLD_SUFFIX = ".old"
#: Read this much at a time: big enough that a 76MB download is not 76,000
#: progress callbacks, small enough to notice a cancel promptly.
CHUNK_BYTES = 256 * 1024


# --------------------------------------------------------------------------
# Versions
# --------------------------------------------------------------------------
_VERSION_RE = re.compile(r"^\s*v?(\d+(?:\.\d+)*)[-+._]?(.*)$", re.IGNORECASE)
#: Numbers are compared per position, padded, so 1.6 and 1.6.0 are one build.
_VERSION_PARTS = 4


@total_ordering
@dataclass(frozen=True)
class Version:
    """A dotted release number, plus whatever was tacked on after it.

    Tolerant on purpose: this parses tag names typed by a human ("v1.6.0",
    "1.6", "1.7.0-beta.2"). Anything unreadable comes back from :meth:`parse`
    as ``None`` rather than raising -- a mistyped tag on the server must not
    break the panel of an install that is working perfectly well.
    """

    numbers: Tuple[int, ...]
    #: "beta.2" in "1.7.0-beta.2". Present means pre-release, which sorts
    #: *below* the same numbers with nothing after them.
    suffix: str = ""

    @classmethod
    def parse(cls, text: Any) -> Optional["Version"]:
        if isinstance(text, Version):
            return text
        if not isinstance(text, str):
            return None
        match = _VERSION_RE.match(text.strip())
        if not match:
            return None
        numbers = tuple(int(part) for part in match.group(1).split("."))
        return cls(numbers, (match.group(2) or "").strip().lower())

    @property
    def is_prerelease(self) -> bool:
        return bool(self.suffix)

    def _key(self):
        padded = (self.numbers + (0,) * _VERSION_PARTS)[:_VERSION_PARTS]
        # No suffix wins, so 1.7.0 beats 1.7.0-beta.2; between two
        # pre-releases the text decides, which reads beta before rc.
        return padded, 1 if not self.suffix else 0, self.suffix

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self._key() == other._key()

    def __lt__(self, other: "Version") -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self._key() < other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __str__(self) -> str:
        text = ".".join(str(number) for number in self.numbers)
        return f"{text}-{self.suffix}" if self.suffix else text


#: What this build calls itself, for comparing against what is published.
CURRENT_VERSION = Version.parse(__version__) or Version((0,))


class UpdateError(Exception):
    """Anything that stopped an update, phrased for the panel's status line."""


# --------------------------------------------------------------------------
# Releases
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Release:
    """One published release, reduced to the parts an update needs."""

    version: Version
    tag: str
    #: The release page, opened in a browser when we cannot self-update.
    url: str = ""
    #: The .exe to download, when the release published one. A release without
    #: an asset is still worth reporting -- it just cannot be applied here.
    asset_name: Optional[str] = None
    asset_url: Optional[str] = None
    asset_size: int = 0
    notes: str = ""

    @property
    def has_asset(self) -> bool:
        return bool(self.asset_url)

    def is_newer_than(self, current: Version = CURRENT_VERSION) -> bool:
        return self.version > current


def releases_url(repo: str = DEFAULT_REPO) -> str:
    return f"{API_ROOT}/repos/{repo or DEFAULT_REPO}/releases?per_page={RELEASE_PAGE_SIZE}"


def fetch_releases(repo: str = DEFAULT_REPO, *, timeout: float = 10.0) -> List[Dict[str, Any]]:
    """The raw release list from GitHub.

    Everything that can go wrong on the way -- offline, rate-limited, repo
    renamed -- arrives as :class:`UpdateError`, so callers have one thing to
    catch rather than the several exception families :mod:`urllib` raises.
    """
    request = Request(
        releases_url(repo),
        headers={"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT},
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https URL
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, OSError, ValueError) as exc:
        raise UpdateError(f"Could not reach GitHub ({exc})") from exc
    if not isinstance(payload, list):
        raise UpdateError("Unexpected reply from GitHub (no release list)")
    return payload


def pick_release(
    payload: Sequence[Dict[str, Any]],
    *,
    include_prereleases: bool = False,
    asset_name: str = ASSET_NAME,
) -> Optional[Release]:
    """The highest usable release in a GitHub payload, or ``None``.

    Highest by version rather than first in the list: releases come back in
    publication order, and a patch to an older line can be published after a
    newer one. Drafts are invisible to an unauthenticated caller anyway, but
    are skipped explicitly in case a token is ever added.
    """
    best: Optional[Release] = None
    for entry in payload:
        if not isinstance(entry, dict) or entry.get("draft"):
            continue
        if entry.get("prerelease") and not include_prereleases:
            continue
        version = Version.parse(entry.get("tag_name") or entry.get("name"))
        if version is None:
            log.debug("Ignoring a release with an unreadable tag: %r", entry.get("tag_name"))
            continue
        release = _release_from(entry, version, asset_name)
        if best is None or release.version > best.version:
            best = release
    return best


def _release_from(entry: Dict[str, Any], version: Version, asset_name: str) -> Release:
    asset = _pick_asset(entry.get("assets"), asset_name)
    return Release(
        version=version,
        tag=str(entry.get("tag_name") or version),
        url=str(entry.get("html_url") or ""),
        asset_name=str(asset.get("name")) if asset else None,
        asset_url=str(asset.get("browser_download_url")) if asset else None,
        asset_size=int(asset.get("size") or 0) if asset else 0,
        notes=str(entry.get("body") or "").strip(),
    )


def _pick_asset(assets: Any, preferred: str) -> Optional[Dict[str, Any]]:
    """The .exe to download: the one named like ours, else any .exe."""
    if not isinstance(assets, list):
        return None
    executables = [
        asset
        for asset in assets
        if isinstance(asset, dict)
        and str(asset.get("name", "")).lower().endswith(".exe")
        and asset.get("browser_download_url")
    ]
    for asset in executables:
        if str(asset.get("name", "")).lower() == preferred.lower():
            return asset
    return executables[0] if executables else None


def latest_release(
    repo: str = DEFAULT_REPO,
    *,
    include_prereleases: bool = False,
    timeout: float = 10.0,
    asset_name: str = ASSET_NAME,
) -> Optional[Release]:
    """Fetch the release list and reduce it to the one that matters."""
    return pick_release(
        fetch_releases(repo, timeout=timeout),
        include_prereleases=include_prereleases,
        asset_name=asset_name,
    )


# --------------------------------------------------------------------------
# Downloading
# --------------------------------------------------------------------------
def download(
    url: str,
    destination: Path,
    *,
    expected_size: int = 0,
    timeout: float = 30.0,
    on_progress: Optional[Callable[[int, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Path:
    """Fetch ``url`` to ``destination``, reporting progress as it goes.

    Written under a temporary name in the same directory and renamed at the
    end, so a download killed half way through can never be mistaken for a
    complete build waiting to be installed.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".part", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    written = 0
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - https from the API
            total = int(response.headers.get("Content-Length") or expected_size or 0)
            with os.fdopen(descriptor, "wb") as sink:
                descriptor = -1  # the file object owns it now
                while True:
                    if should_stop is not None and should_stop():
                        raise UpdateError("Download cancelled")
                    chunk = response.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    sink.write(chunk)
                    written += len(chunk)
                    if on_progress is not None:
                        on_progress(written, total)
    except UpdateError:
        _discard(temporary)
        raise
    except (URLError, OSError, ValueError) as exc:
        _discard(temporary)
        raise UpdateError(f"Download failed ({exc})") from exc
    finally:
        if descriptor != -1:
            os.close(descriptor)

    # The size is all the published metadata gives us to check against, but a
    # truncated download is exactly the failure that would otherwise be
    # installed over a working build.
    if expected_size and written != expected_size:
        _discard(temporary)
        raise UpdateError(f"Download is {written} bytes, expected {expected_size}")

    _discard(destination)
    try:
        temporary.replace(destination)
    except OSError as exc:
        _discard(temporary)
        raise UpdateError(f"Could not save the download ({exc})") from exc
    return destination


def _discard(path: Path) -> bool:
    """Delete a file if it is there and lets go. Never raises."""
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return True
    except OSError as exc:
        log.debug("Could not delete %s: %s", path, exc)
        return False


# --------------------------------------------------------------------------
# Installing
# --------------------------------------------------------------------------
def current_exe() -> Optional[Path]:
    """The .exe to replace, or ``None`` when running from a source checkout."""
    if not getattr(sys, "frozen", False):
        return None
    return Path(sys.executable).resolve()


def is_onedir_build(exe: Path) -> bool:
    """Whether this is the folder build, whose .exe needs ``_internal`` beside it.

    Only the one-file .exe is published as a release asset, so swapping the
    executable of a folder install would pair a new program with the old
    build's libraries. That user is better off on the release page.
    """
    return (exe.parent / "_internal").is_dir()


def can_self_update(exe: Optional[Path] = None) -> bool:
    """Whether :func:`install` can actually replace what is running."""
    exe = exe or current_exe()
    return bool(exe is not None and os.name == "nt" and not is_onedir_build(exe))


def why_not_self_update(exe: Optional[Path] = None) -> str:
    """A few words for the panel when :func:`can_self_update` says no.

    Kept short because it is shown inside the status line, next to the version
    that was found; the log carries the longer story.
    """
    exe = exe or current_exe()
    if exe is None:
        return "source checkout"
    if os.name != "nt":
        return "not a Windows build"
    if is_onedir_build(exe):
        return "folder install"
    return ""


def staged_path(exe: Path) -> Path:
    """Where a downloaded build waits beside the .exe it will replace."""
    return exe.with_name(exe.name + NEW_SUFFIX)


def clean_leftovers(directory: Path) -> None:
    """Delete the previous build and any abandoned downloads.

    Called at startup: the helper script deletes the ``.old`` file itself, but
    only if it stays alive long enough to win the race with a user who
    double-clicks the new .exe the moment it appears.
    """
    try:
        entries = list(directory.glob("*" + OLD_SUFFIX)) + list(directory.glob("*.part"))
    except OSError:
        return
    for entry in entries:
        if _discard(entry):
            log.debug("Cleaned up %s", entry.name)


def install(staged: Path, exe: Optional[Path] = None) -> None:
    """Put ``staged`` in place of the running .exe and restart into it.

    A running image cannot be overwritten, so it is renamed aside first --
    Windows allows that -- and the new file takes its name. A small .cmd script
    then waits for this process to let go of the old file, deletes it and
    starts the new build; the caller's job is to exit promptly afterwards.
    """
    exe = exe or current_exe()
    if exe is None:
        raise UpdateError("Not a packaged build, so there is nothing to replace")
    if is_onedir_build(exe):
        raise UpdateError("Folder installs have to be updated by hand")
    if not staged.exists():
        raise UpdateError("The downloaded build has gone missing")

    retired = exe.with_name(exe.name + OLD_SUFFIX)
    _discard(retired)
    try:
        exe.rename(retired)
    except OSError as exc:
        raise UpdateError(f"Could not move the running build aside ({exc})") from exc
    try:
        staged.replace(exe)
    except OSError as exc:
        # Put the working build back rather than leaving the folder with no
        # .exe in it at all.
        try:
            retired.rename(exe)
        except OSError:
            log.error("Could not restore %s from %s", exe.name, retired.name)
        raise UpdateError(f"Could not install the new build ({exc})") from exc

    _launch_restart_script(exe, retired)


def _launch_restart_script(exe: Path, retired: Path) -> None:
    """Start the detached .cmd that waits for us to exit, then relaunches."""
    script = exe.with_name(f"{exe.stem}-update.cmd")
    try:
        script.write_text(_restart_script(exe, retired), encoding="ascii")
        creation = 0
        for name in ("CREATE_NO_WINDOW", "DETACHED_PROCESS"):
            creation |= getattr(subprocess, name, 0)
        subprocess.Popen(  # noqa: S603 - our own script, by absolute path
            ["cmd.exe", "/c", str(script)],
            cwd=str(exe.parent),
            creationflags=creation,
            close_fds=True,
        )
    except OSError as exc:
        # The new build is already in place, so this is only a failure to
        # restart: say so, and the user starts it themselves.
        raise UpdateError(f"Installed, but could not restart automatically ({exc})") from exc


def _restart_script(exe: Path, retired: Path) -> str:
    """The wait-delete-start-vanish script.

    Deleting the retired build is both the cleanup and the wait: the delete
    keeps failing while this process still has that image open, so the loop
    ends exactly when we are gone. It gives up after roughly a minute and
    starts the new build anyway -- a leftover ``.old`` file is tidied at the
    next startup, and a second instance is refused by the tray's own mutex.
    """
    # Plain newlines: write_text turns them into CRLF on the way out, and
    # spelling the carriage return out here too would hand cmd.exe a doubled
    # one on every line.
    return (
        "@echo off\n"
        "setlocal\n"
        "set TRIES=0\n"
        ":wait\n"
        'del "{retired}" >nul 2>&1\n'
        'if not exist "{retired}" goto start\n'
        "set /a TRIES+=1\n"
        "if %TRIES% GEQ 60 goto start\n"
        "ping -n 2 127.0.0.1 >nul\n"
        "goto wait\n"
        ":start\n"
        'start "" "{exe}"\n'
        'del "%~f0" >nul 2>&1\n'
    ).format(retired=retired, exe=exe)
