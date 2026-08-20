"""Update tests: reading releases, fetching one, and swapping it in.

No network and no real .exe -- ``urlopen`` is stubbed and the "running build"
is a text file in ``tmp_path``, which is enough to prove the rename dance the
installer does. The restart script is written but never run: launching it is
the one thing these tests stub out.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest import mock

import pytest

from ttheart_sender import update
from ttheart_sender.tray.updater import UpdateService, UpdateState
from ttheart_sender.update import Release, UpdateError, Version


# --------------------------------------------------------------------------
# Versions
# --------------------------------------------------------------------------
def test_a_tag_is_read_with_or_without_its_v():
    assert Version.parse("v1.6.0") == Version.parse("1.6.0") == Version((1, 6, 0))


def test_missing_positions_are_zeroes_not_a_different_build():
    assert Version.parse("1.6") == Version.parse("1.6.0.0")


def test_versions_compare_by_number_not_by_text():
    assert Version.parse("1.10.0") > Version.parse("1.9.0")
    assert Version.parse("2.0") > Version.parse("1.99.99")


def test_a_prerelease_sorts_below_the_release_it_leads_to():
    beta = Version.parse("1.7.0-beta.2")
    assert beta.is_prerelease
    assert beta < Version.parse("1.7.0")
    assert beta > Version.parse("1.6.0")
    assert Version.parse("1.7.0-beta.2") > Version.parse("1.7.0-beta.1")


@pytest.mark.parametrize("text", ["", "latest", None, 17, "v"])
def test_an_unreadable_tag_is_ignored_rather_than_fatal(text):
    assert Version.parse(text) is None


def test_the_running_build_parses():
    """A version.py this module cannot read would break every comparison."""
    from ttheart_sender import __version__

    assert Version.parse(__version__) == update.CURRENT_VERSION


# --------------------------------------------------------------------------
# Picking a release
# --------------------------------------------------------------------------
def release_entry(tag, *, assets=("ttheart-sender.exe",), prerelease=False, draft=False):
    return {
        "tag_name": tag,
        "html_url": f"https://github.com/x/y/releases/tag/{tag}",
        "prerelease": prerelease,
        "draft": draft,
        "body": f"notes for {tag}",
        "assets": [
            {
                "name": name,
                "size": 1234,
                "browser_download_url": f"https://example.invalid/{tag}/{name}",
            }
            for name in assets
        ],
    }


def test_the_highest_version_wins_not_the_most_recently_published():
    payload = [release_entry("v1.5.1"), release_entry("v1.7.0"), release_entry("v1.6.0")]
    assert update.pick_release(payload).tag == "v1.7.0"


def test_prereleases_and_drafts_are_skipped_unless_asked_for():
    payload = [
        release_entry("v1.8.0-beta.1", prerelease=True),
        release_entry("v1.9.0", draft=True),
        release_entry("v1.7.0"),
    ]
    assert update.pick_release(payload).tag == "v1.7.0"
    assert update.pick_release(payload, include_prereleases=True).tag == "v1.8.0-beta.1"
    # A draft stays invisible either way.
    assert update.pick_release(payload, include_prereleases=True).tag != "v1.9.0"


def test_the_exe_asset_is_found_among_the_others():
    entry = release_entry("v1.7.0", assets=("notes.txt", "ttheart-sender.exe"))
    release = update.pick_release([entry])
    assert release.asset_name == "ttheart-sender.exe"
    assert release.asset_size == 1234
    assert release.has_asset


def test_a_release_without_an_exe_is_still_reported():
    """It cannot be installed here, but the panel should still say it exists."""
    release = update.pick_release([release_entry("v1.7.0", assets=("source.zip",))])
    assert release.tag == "v1.7.0"
    assert release.has_asset is False


def test_an_empty_or_unreadable_payload_yields_nothing():
    assert update.pick_release([]) is None
    assert update.pick_release([{"tag_name": "nightly"}, "junk"]) is None


def test_the_repo_defaults_to_this_project():
    assert update.releases_url("") == update.releases_url(update.DEFAULT_REPO)
    assert "iDunoPlay/ttheart-sender" in update.releases_url()


def test_being_offline_is_reported_not_raised_as_a_url_error():
    from urllib.error import URLError

    with mock.patch.object(update, "urlopen", side_effect=URLError("no route")):
        with pytest.raises(UpdateError):
            update.fetch_releases()


# --------------------------------------------------------------------------
# Downloading
# --------------------------------------------------------------------------
class FakeResponse(io.BytesIO):
    """Just enough of an HTTP response for :func:`update.download`."""

    def __init__(self, payload: bytes, *, length=None):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload) if length is None else length)}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def test_a_download_lands_under_its_real_name_with_no_scraps_left(tmp_path):
    target = tmp_path / "ttheart-sender.exe.new"
    with mock.patch.object(update, "urlopen", return_value=FakeResponse(b"payload" * 100)):
        update.download("https://example.invalid/x.exe", target, expected_size=700)

    assert target.read_bytes() == b"payload" * 100
    assert list(tmp_path.glob("*.part")) == []


def test_a_short_download_is_thrown_away_rather_than_installed(tmp_path):
    target = tmp_path / "ttheart-sender.exe.new"
    with mock.patch.object(update, "urlopen", return_value=FakeResponse(b"half")):
        with pytest.raises(UpdateError, match="expected"):
            update.download("https://example.invalid/x.exe", target, expected_size=999)

    assert not target.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_a_cancelled_download_leaves_nothing_behind(tmp_path):
    target = tmp_path / "ttheart-sender.exe.new"
    with mock.patch.object(update, "urlopen", return_value=FakeResponse(b"x" * 10)):
        with pytest.raises(UpdateError, match="cancelled"):
            update.download("https://example.invalid/x.exe", target, should_stop=lambda: True)

    assert not target.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_progress_is_reported_against_the_advertised_size(tmp_path):
    seen = []
    with mock.patch.object(update, "urlopen", return_value=FakeResponse(b"z" * 4096)):
        update.download(
            "https://example.invalid/x.exe",
            tmp_path / "out.new",
            on_progress=lambda written, total: seen.append((written, total)),
        )
    assert seen[-1] == (4096, 4096)


# --------------------------------------------------------------------------
# Installing
# --------------------------------------------------------------------------
@pytest.fixture()
def install_dir(tmp_path):
    """A folder that looks like a one-file install: the .exe and a download."""
    exe = tmp_path / "ttheart-sender.exe"
    exe.write_text("old build", encoding="utf-8")
    update.staged_path(exe).write_text("new build", encoding="utf-8")
    return exe


def test_installing_moves_the_running_build_aside_and_takes_its_name(install_dir):
    exe = install_dir
    with mock.patch.object(update.subprocess, "Popen") as popen:
        update.install(update.staged_path(exe), exe)

    assert exe.read_text(encoding="utf-8") == "new build"
    # The old image is still open by the running process, so it is renamed
    # rather than deleted; the script does the deleting.
    assert exe.with_name(exe.name + update.OLD_SUFFIX).read_text(encoding="utf-8") == "old build"
    assert not update.staged_path(exe).exists()
    assert popen.call_count == 1


def test_the_restart_script_waits_deletes_and_relaunches(install_dir):
    exe = install_dir
    with mock.patch.object(update.subprocess, "Popen"):
        update.install(update.staged_path(exe), exe)

    script = exe.with_name(f"{exe.stem}-update.cmd")
    body = script.read_text(encoding="ascii")
    assert str(exe.with_name(exe.name + update.OLD_SUFFIX)) in body
    assert f'start "" "{exe}"' in body
    assert 'del "%~f0"' in body
    # One carriage return per line: the text is written with plain newlines
    # precisely because the file object adds the CR itself.
    assert b"\r\r\n" not in script.read_bytes()


def test_a_failed_swap_puts_the_working_build_back(install_dir):
    exe = install_dir
    with mock.patch.object(Path, "replace", side_effect=OSError("in use")):
        with pytest.raises(UpdateError):
            update.install(update.staged_path(exe), exe)

    assert exe.read_text(encoding="utf-8") == "old build"
    assert not exe.with_name(exe.name + update.OLD_SUFFIX).exists()


def test_a_missing_download_is_refused_before_anything_moves(tmp_path):
    exe = tmp_path / "ttheart-sender.exe"
    exe.write_text("old build", encoding="utf-8")
    with pytest.raises(UpdateError, match="missing"):
        update.install(update.staged_path(exe), exe)
    assert exe.read_text(encoding="utf-8") == "old build"


def test_a_folder_install_is_not_updated_in_place(tmp_path):
    """Only the one-file .exe is published, so swapping it alone would break."""
    exe = tmp_path / "ttheart-sender.exe"
    exe.write_text("old build", encoding="utf-8")
    (tmp_path / "_internal").mkdir()
    update.staged_path(exe).write_text("new build", encoding="utf-8")

    assert update.is_onedir_build(exe)
    assert update.can_self_update(exe) is False
    assert "folder install" in update.why_not_self_update(exe)
    with pytest.raises(UpdateError):
        update.install(update.staged_path(exe), exe)


def test_a_source_checkout_knows_it_cannot_update_itself():
    assert update.current_exe() is None  # the tests are not frozen
    assert update.can_self_update() is False
    assert "source checkout" in update.why_not_self_update()


def test_leftovers_from_the_last_swap_are_tidied(tmp_path):
    (tmp_path / ("ttheart-sender.exe" + update.OLD_SUFFIX)).write_text("x", encoding="utf-8")
    (tmp_path / "ttheart-sender.exe.new.abc.part").write_text("x", encoding="utf-8")
    keep = tmp_path / "ttheart-sender.exe"
    keep.write_text("x", encoding="utf-8")

    update.clean_leftovers(tmp_path)

    assert keep.exists()
    assert list(tmp_path.glob("*" + update.OLD_SUFFIX)) == []
    assert list(tmp_path.glob("*.part")) == []


# --------------------------------------------------------------------------
# The service behind the panel
# --------------------------------------------------------------------------
class FakeUpdateConfig:
    enabled = True
    repo = "someone/ttheart-sender"
    check_interval_hours = 6.0
    include_prereleases = False
    timeout = 10.0


def service(**kwargs):
    """An update service with its callbacks recorded rather than wired up."""
    changes = []
    notices = []
    updater = UpdateService(
        FakeUpdateConfig(),
        on_change=lambda: changes.append(True),
        on_notify=lambda title, message, error: notices.append((title, message, error)),
        **kwargs,
    )
    return updater, changes, notices


def newer(tag="v99.0.0", **kwargs):
    return update.pick_release([release_entry(tag, **kwargs)])


def test_being_up_to_date_says_so_without_offering_anything(monkeypatch):
    updater, _changes, notices = service()
    monkeypatch.setattr(
        "ttheart_sender.tray.updater.latest_release", lambda *a, **k: newer("v0.0.1")
    )

    updater._check()

    assert updater.state is UpdateState.IDLE
    assert updater.update_available is False
    assert updater.status_text().endswith("up to date")
    assert updater.button_label() == "Check"
    assert notices == []


def test_a_newer_release_is_announced_and_offered(monkeypatch):
    updater, changes, notices = service()
    monkeypatch.setattr("ttheart_sender.tray.updater.latest_release", lambda *a, **k: newer())
    monkeypatch.setattr("ttheart_sender.tray.updater.can_self_update", lambda *a: True)

    updater._check()

    assert updater.state is UpdateState.AVAILABLE
    assert updater.update_available is True
    assert updater.status_text() == "v99.0.0 available"
    assert updater.button_label() == "Update"
    assert notices[0][0] == "Update available"
    assert changes  # the panel was asked to repaint


def test_a_failed_check_is_shown_on_the_status_line_not_raised(monkeypatch):
    updater, _changes, _notices = service()

    def explode(*args, **kwargs):
        raise UpdateError("Could not reach GitHub (offline)")

    monkeypatch.setattr("ttheart_sender.tray.updater.latest_release", explode)
    updater._check()

    assert updater.state is UpdateState.FAILED
    assert "offline" in updater.status_text()
    assert updater.button_label() == "Retry"


def test_checks_can_be_switched_off_entirely(monkeypatch):
    config = FakeUpdateConfig()
    config.enabled = False
    updater = UpdateService(config)
    monkeypatch.setattr(
        "ttheart_sender.tray.updater.latest_release",
        lambda *a, **k: pytest.fail("checked with update.enabled false"),
    )

    updater._check()
    assert updater.state is UpdateState.IDLE
    assert updater._check_due() is False


def test_auto_update_installs_by_itself_once_a_build_is_found(monkeypatch):
    installed = []
    updater, _changes, _notices = service(auto=True, on_restart=lambda: installed.append("exit"))
    monkeypatch.setattr("ttheart_sender.tray.updater.latest_release", lambda *a, **k: newer())
    monkeypatch.setattr("ttheart_sender.tray.updater.can_self_update", lambda *a: True)
    monkeypatch.setattr(
        "ttheart_sender.tray.updater.current_exe", lambda: Path("C:/x/ttheart-sender.exe")
    )
    monkeypatch.setattr(
        "ttheart_sender.tray.updater.download", lambda *a, **k: installed.append("download")
    )
    monkeypatch.setattr(
        "ttheart_sender.tray.updater.install", lambda *a, **k: installed.append("install")
    )

    updater._check()
    updater._maybe_auto_install()

    assert installed == ["download", "install", "exit"]
    assert updater.state is UpdateState.READY


def test_an_automatic_install_waits_for_the_run_to_finish(monkeypatch):
    """Restarting mid-flow would abandon the round being played."""
    busy = {"running": True}
    installed = []
    updater, _changes, _notices = service(auto=True, can_apply=lambda: not busy["running"])
    monkeypatch.setattr("ttheart_sender.tray.updater.latest_release", lambda *a, **k: newer())
    monkeypatch.setattr("ttheart_sender.tray.updater.can_self_update", lambda *a: True)
    monkeypatch.setattr(
        "ttheart_sender.tray.updater.current_exe", lambda: Path("C:/x/ttheart-sender.exe")
    )
    monkeypatch.setattr(
        "ttheart_sender.tray.updater.download", lambda *a, **k: installed.append("download")
    )
    monkeypatch.setattr(
        "ttheart_sender.tray.updater.install", lambda *a, **k: installed.append("install")
    )

    updater._check()
    updater._maybe_auto_install()
    assert installed == []
    assert updater.state is UpdateState.AVAILABLE

    busy["running"] = False
    updater._maybe_auto_install()
    assert installed == ["download", "install"]


def test_nothing_is_installed_while_the_box_is_unticked(monkeypatch):
    installed = []
    updater, _changes, _notices = service(auto=False)
    monkeypatch.setattr("ttheart_sender.tray.updater.latest_release", lambda *a, **k: newer())
    monkeypatch.setattr("ttheart_sender.tray.updater.can_self_update", lambda *a: True)
    monkeypatch.setattr(
        "ttheart_sender.tray.updater.install", lambda *a, **k: installed.append("install")
    )

    updater._check()
    updater._maybe_auto_install()

    # Found and reported, but not acted on -- that is what the tick box means.
    assert installed == []
    assert updater.state is UpdateState.AVAILABLE


def test_a_build_that_cannot_replace_itself_offers_the_release_page(monkeypatch):
    opened = []
    updater, _changes, _notices = service()
    monkeypatch.setattr("ttheart_sender.tray.updater.latest_release", lambda *a, **k: newer())
    monkeypatch.setattr("ttheart_sender.tray.updater.can_self_update", lambda *a: False)
    monkeypatch.setattr(
        "ttheart_sender.tray.updater.why_not_self_update", lambda *a: "source checkout"
    )
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))

    updater._check()
    assert updater.button_label() == "Open"
    assert "source checkout" in updater.status_text()

    updater.activate()
    assert opened == [newer().url]


def test_the_button_checks_when_there_is_nothing_to_install(monkeypatch):
    updater, _changes, _notices = service()
    updater.activate()
    assert updater._command == "check"


def test_the_button_does_nothing_while_a_download_is_in_flight():
    updater, _changes, _notices = service()
    updater._set(UpdateState.DOWNLOADING)
    updater.activate()
    assert updater._command is None
    assert updater.busy is True


def test_a_release_with_no_exe_cannot_be_installed(monkeypatch):
    updater, _changes, _notices = service(auto=True)
    monkeypatch.setattr(
        "ttheart_sender.tray.updater.latest_release",
        lambda *a, **k: newer(assets=("source.zip",)),
    )
    monkeypatch.setattr("ttheart_sender.tray.updater.can_self_update", lambda *a: True)

    updater._check()
    updater._install()

    assert updater.state is UpdateState.FAILED
    assert "no .exe" in updater.status_text()


def test_download_progress_reaches_the_status_line():
    updater, changes, _notices = service()
    with updater._lock:
        updater._latest = Release(Version((99, 0, 0)), "v99.0.0")
    updater._set(UpdateState.DOWNLOADING)
    updater._progress(50, 200)

    assert updater.status_text() == "Downloading v99.0.0 - 25%"
    # Same percentage twice must not repaint the panel again.
    before = len(changes)
    updater._progress(51, 200)
    assert len(changes) == before


def test_ticking_the_box_is_remembered_and_wakes_the_worker():
    updater, changes, _notices = service(auto=False)
    assert updater.set_auto(True) is True
    assert updater.auto is True
    assert updater._wake.is_set()
    assert updater.set_auto(True) is False  # no change, no repaint
    assert len(changes) == 1
