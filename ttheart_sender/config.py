"""Typed configuration loaded from ``config.yaml``.

Every setting lives in a dataclass with a default, so the YAML file only needs
to contain what you want to override. Unknown keys raise :class:`ConfigError`
rather than being silently ignored -- a typo'd setting that does nothing is far
harder to debug than an error at startup.
"""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_args, get_origin, get_type_hints

import yaml

from .exceptions import ConfigError
from .geometry import Insets

DEFAULT_CONFIG_NAME = "config.yaml"
LOCAL_CONFIG_NAME = "config.local.yaml"


def default_app_root() -> Path:
    """Where config.yaml / templates / flows / logs live when not overridden.

    Two cases:

    * **Dev / source checkout**: two levels up from this file, i.e. the repo
      root next to ``main.py``.
    * **Frozen (PyInstaller) .exe**: the directory the .exe was launched from,
      *not* the PyInstaller temp extraction dir that ``__file__`` would point
      into. This is what lets someone drop ``config.yaml`` / ``templates/`` /
      ``flows/`` beside the .exe and edit them without rebuilding it.

      A one-file build has no such directory until the user creates one, so if
      there is no ``config.yaml`` next to the .exe we fall back to the copies
      baked into the bundle. Dropping a ``config.yaml`` beside the .exe is
      therefore the switch that takes over the whole data directory.
    """
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        if (exe_dir / DEFAULT_CONFIG_NAME).exists():
            return exe_dir
        return bundle_dir() or exe_dir
    return Path(__file__).resolve().parent.parent


def bundle_dir() -> Optional[Path]:
    """The PyInstaller one-file extraction directory, if we are running in one."""
    meipass = getattr(sys, "_MEIPASS", None)
    return Path(meipass).resolve() if meipass else None


def default_output_root(root: Path) -> Path:
    """Where logs and debug screenshots go.

    Normally alongside everything else, but a one-file bundle is unpacked into
    a temp directory that is deleted on exit -- logs written there would be
    gone before anyone could read them. Send those next to the .exe instead.
    """
    bundle = bundle_dir()
    if bundle is not None and root == bundle:
        return Path(sys.executable).resolve().parent
    return root


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------
@dataclass
class AppConfig:
    log_level: str = "INFO"
    log_dir: str = "logs"


@dataclass
class WindowTargetConfig:
    """How to recognise the emulator window among all open windows."""

    # LDPlayer's player window class across versions 3/4/9.
    class_names: List[str] = field(
        default_factory=lambda: ["LDPlayerMainFrame", "LDPlayerMainFrame2"]
    )
    title_contains: List[str] = field(default_factory=lambda: ["LDPlayer"])
    # The multi-instance manager looks similar but is not a player window.
    exclude_class_names: List[str] = field(
        default_factory=lambda: ["LDMultiPlayerMainFrame", "LDMultiPlayerMainFrame2"]
    )
    exclude_title_contains: List[str] = field(default_factory=lambda: ["Multi-Instance", "多开器"])
    # When True a window matching *either* class or title is accepted; when
    # False it must match both. Keep True -- LDPlayer instances get renamed.
    match_any: bool = True


@dataclass
class WindowPositionConfig:
    x: int = 0
    y: int = 0


@dataclass
class WindowSizeConfig:
    width: int
    height: int


@dataclass
class WindowConfig:
    target: WindowTargetConfig = field(default_factory=WindowTargetConfig)
    #: Which detected instance to drive (0-based) when several are running.
    instance: int = 0
    #: Match a specific instance by (case-insensitive substring of) its title.
    instance_title: Optional[str] = None
    #: Where to park the window before automating. Top-left keeps the maths easy.
    position: WindowPositionConfig = field(default_factory=WindowPositionConfig)
    #: Force a window size on startup, or null to leave it as the user set it.
    size: Optional[WindowSizeConfig] = None
    #: Detect the emulator and park it at ``position`` as the first thing every
    #: session does, before any command runs. Turn off to leave it where it is.
    prepare_on_launch: bool = True
    move_on_prepare: bool = True
    focus_on_prepare: bool = True
    #: Search inside the client area (excludes title bar) rather than the frame.
    use_client_area: bool = True
    #: Trim LDPlayer's side toolbar / bottom bar out of the search area.
    insets: Insets = field(default_factory=Insets)
    #: Re-read the window rectangle before every screen capture.
    refresh_rect_each_capture: bool = False


@dataclass
class MatchingConfig:
    templates_dir: str = "templates"
    confidence: float = 0.85
    grayscale: bool = True
    #: Scale factors tried for each template; add e.g. 0.9/1.1 if the emulator
    #: resolution varies between runs.
    scales: List[float] = field(default_factory=lambda: [1.0])
    #: Ignore alpha channel instead of using it as a match mask.
    ignore_alpha: bool = False
    debug_dir: str = "logs/debug"
    #: Save an annotated screenshot whenever a lookup fails.
    save_failures: bool = False


@dataclass
class InputConfig:
    move_duration: float = 0.12
    #: Random pixel jitter added to every click so taps are not pixel-identical.
    click_jitter: int = 2
    #: Pause after pressing before releasing the button.
    click_hold: float = 0.05
    post_click_delay: float = 0.15
    #: pyautogui's fail-safe: slam the cursor into a screen corner to abort.
    failsafe: bool = True


@dataclass
class RunnerConfig:
    flows_dir: str = "flows"
    step_delay: float = 0.05
    default_timeout: float = 10.0
    default_poll_interval: float = 0.25
    #: Global panic key polled between steps and during waits.
    stop_key: Optional[str] = "f12"
    #: Stop the run when the physical cursor leaves the emulator window.
    #: Arms on the first checkpoint that sees the cursor inside it.
    stop_on_cursor_exit: bool = True
    #: Slack in pixels around the window before the guard trips, so a click
    #: on the very edge (plus its jitter) is not treated as an escape.
    cursor_exit_margin: int = 16
    #: Safety net for ``repeat: {forever: true}``.
    max_iterations: int = 100_000


@dataclass
class DatasetConfig:
    """Training samples captured while playing -- see :mod:`..game.dataset`.

    Off by default: it costs one extra capture on the drags it samples, and
    nobody should be writing images to disk without having asked for it.
    """

    enabled: bool = False
    #: Relative paths land next to the .exe, not inside a one-file bundle's
    #: temp directory -- the same rule logs follow, and for the same reason.
    dir: str = "dataset"
    #: Cap per round. Boards inside one round are highly alike, so a handful
    #: from each of many rounds beats hundreds from one.
    per_round: int = 20
    #: Sample every Nth drag rather than the first N.
    every: int = 4
    quality: int = 85
    #: Seconds to wait after the press before photographing the marks. NOT
    #: ``--hold-delay``: that one is paid on every drag and sits below the
    #: 0.15s floor where the game has finished drawing the highlight, which is
    #: why the first collection carried no label at all. Clamped up to the
    #: floor by :class:`~..game.dataset.DatasetWriter`.
    delay: float = 0.25
    #: Frames to read the marks from -- only what changed in all of them
    #: counts, which is what keeps a settling board out of the reading.
    frames: int = 3
    #: Seconds between those frames.
    gap: float = 0.05
    #: A mark has to beat the board's own noise floor by this much. Measured
    #: with ``hold``, real marks sit 8x-25x above it; the fixed 8.0 threshold
    #: sits inside that floor on a live board. 0 = use the fixed threshold.
    floor_mult: float = 5.0
    #: Refuse a sample whose board was already jiggling this hard at press
    #: time; the reading would be motion rather than marks. 0 = keep all.
    max_motion: float = 12.0
    #: Budgets across the whole dataset directory, not one session. There is
    #: no per-day limit anywhere else: unattended, the first collection wrote
    #: 803MB in one night. 0 = no cap.
    max_mb: float = 2048.0
    max_total: int = 0


@dataclass
class UpdateConfig:
    """Where new builds come from -- see :mod:`..update`.

    Only the checking is configured here. Whether a newer build is *installed*
    by itself is the panel's "Auto Update" tick box, which lives with the other
    per-user switches in ``ttheart-settings.json``.
    """

    #: Look for new releases at all. Off means the panel never asks GitHub.
    enabled: bool = True
    #: ``owner/name`` on GitHub. Empty falls back to the project's own repo,
    #: so a fork only has to set this line.
    repo: str = ""
    #: Hours between checks. Clamped by the updater, so a typo here cannot
    #: turn the panel into a polling loop.
    check_interval_hours: float = 6.0
    #: Offer betas and release candidates as well as finished releases.
    include_prereleases: bool = False
    #: Seconds to wait on the API call. The download is given more.
    timeout: float = 10.0


@dataclass
class Config:
    app: AppConfig = field(default_factory=AppConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    input: InputConfig = field(default_factory=InputConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    update: UpdateConfig = field(default_factory=UpdateConfig)

    #: Directory the config was loaded from; all relative paths resolve here.
    root: Path = field(default_factory=Path.cwd)
    #: Where runtime output is written. Same as ``root`` except inside a
    #: one-file .exe, whose ``root`` is a temp directory wiped on exit.
    output_root: Path = field(default_factory=Path.cwd)

    # -- path helpers ----------------------------------------------------
    def resolve(self, value: Union[str, Path]) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.root / path)

    def resolve_output(self, value: Union[str, Path]) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.output_root / path)

    @property
    def templates_dir(self) -> Path:
        return self.resolve(self.matching.templates_dir)

    @property
    def flows_dir(self) -> Path:
        return self.resolve(self.runner.flows_dir)

    @property
    def log_dir(self) -> Path:
        return self.resolve_output(self.app.log_dir)

    @property
    def debug_dir(self) -> Path:
        return self.resolve_output(self.matching.debug_dir)

    @property
    def dataset_dir(self) -> Path:
        return self.resolve_output(self.dataset.dir)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_config(path: Optional[Union[str, Path]] = None, *, root: Optional[Path] = None) -> Config:
    """Load ``config.yaml`` (plus an optional ``config.local.yaml`` overlay).

    ``path`` may point at a file or a directory. When omitted the package's
    parent directory (the project root) is used.
    """
    project_root = Path(root) if root else default_app_root()

    if path is None:
        config_path: Optional[Path] = project_root / DEFAULT_CONFIG_NAME
    else:
        candidate = Path(path)
        if candidate.is_dir():
            project_root = candidate.resolve()
            config_path = candidate / DEFAULT_CONFIG_NAME
        else:
            project_root = candidate.resolve().parent
            config_path = candidate

    data: Dict[str, Any] = {}
    if config_path is not None and config_path.exists():
        data = _read_yaml_mapping(config_path)

    local_path = project_root / LOCAL_CONFIG_NAME
    if local_path.exists():
        data = deep_merge(data, _read_yaml_mapping(local_path))

    config = from_dict(Config, data, exclude={"root", "output_root"})
    config.root = project_root
    config.output_root = default_output_root(project_root)
    return config


def _read_yaml_mapping(path: Path) -> Dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML -- {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level, got {type(raw).__name__}")
    return raw


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overlay`` onto ``base`` without mutating either."""
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


# --------------------------------------------------------------------------
# Generic dataclass <- dict coercion
# --------------------------------------------------------------------------
def from_dict(cls: type, data: Any, *, path: str = "", exclude: Optional[set] = None) -> Any:
    """Build a dataclass instance from a plain mapping, validating as we go."""
    if data is None:
        data = {}
    where = path or "config"
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")

    exclude = exclude or set()
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls) if f.name not in exclude}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(
            f"{where}: unknown setting(s) {unknown}. Valid keys: {sorted(known)}"
        )

    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name in exclude or f.name not in data:
            continue
        child_path = f"{path}.{f.name}" if path else f.name
        kwargs[f.name] = _coerce(hints[f.name], data[f.name], child_path)
    return cls(**kwargs)


def _coerce(target: Any, value: Any, path: str) -> Any:
    origin = get_origin(target)

    # Optional[X] / Union[...]
    if origin is Union:
        args = [a for a in get_args(target) if a is not type(None)]  # noqa: E721
        if value is None:
            return None
        if len(args) == 1:
            return _coerce(args[0], value, path)
        for arg in args:  # best effort for genuine unions
            try:
                return _coerce(arg, value, path)
            except ConfigError:
                continue
        raise ConfigError(f"{path}: value {value!r} does not match {target}")

    if is_dataclass(target):
        return from_dict(target, value, path=path)

    if origin in (list, List):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        (item_type,) = get_args(target) or (Any,)
        return [_coerce(item_type, item, f"{path}[{i}]") for i, item in enumerate(value)]

    if origin in (dict, Dict):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        return dict(value)

    if target is Any:
        return value

    if target is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{path}: expected true/false, got {value!r}")

    if target in (int, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return target(value)

    if target is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value

    if target is Path:
        return Path(value)

    return value


def to_dict(config: Any) -> Dict[str, Any]:
    """Serialise a config dataclass back to plain data (handy for debugging)."""
    return dataclasses.asdict(config)
