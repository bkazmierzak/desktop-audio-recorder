"""Main application window: device line, transport, level meter, waveform, playback, status bar.

Threading: every pyaudiowpatch/PortAudio call (device refresh, ``recorder.start()``,
``recorder.stop()``) runs on one background thread through ``SerialTaskRunner``; results come
back on the GUI thread. ``pause()``/``resume()`` only flip the recorder's clock, so they run
directly. After Stop, the WAV is loaded (and its waveform envelope built) on a second runner.
Playback uses sounddevice's own PortAudio; ``ClipPlayer`` re-initialises it and opens/closes
streams on its own stream thread (about 90 ms on some drivers), so ``play()``/``pause()`` return
at once; the stream callback only copies samples.

A 30 ms timer polls the recorder's levels and elapsed time while recording (its ``on_level``
callback fires on the writer thread, so it isn't used here) and the player's position while
playing.

Editing goes through an ``EditSession`` (clip + waveform pyramid + undo history). An edit made
while playing pauses playback first. Edits that touch at most ``sync_edit_frames`` frames run
directly on the GUI thread (a few ms); bigger ones (any length-changing edit on a long clip, or a
gain over a long selection) run on the loader thread in the ``EDITING`` state, so the GUI stays
responsive. Either way the waveform, player, selection and cursor are updated in place
(``replace_clip``): the zoom and scroll position are kept.

Noise reduction (seconds per minute of audio) asks for its options in ``DenoiseDialog``, then
runs on the loader thread in the ``PROCESSING`` state, with a progress bar and a Cancel button
in the status bar. It computes the result without touching the clip (Cancel just discards it),
then applies it as one undoable ``ReplaceRegion`` edit.

Export (Ctrl+S, ``ExportDialog``) writes the whole clip to WAV or MP3 on the loader thread in the
``EXPORTING`` state, with the same progress bar and Cancel button; the file appears only once it
is complete. The clip counts as saved until the next edit (``EditHistory.position`` is noted at
export; undoing back to it counts as saved again). Starting a new recording or closing the window
with an unsaved clip asks Save / Discard / Cancel first; Save exports and then carries on.

Device changes (``DeviceWatch``, Windows Core Audio notifications). While recording, a change of
the Windows default playback device moves the recording to the new device
(``LoopbackRecorder.switch_device`` on the audio thread, ``SWITCH_DELAY_MS`` after the changes
settle), when Recording > Follow the default playback device is on (the default); otherwise a
warning banner says the recording carries on from the device it started on. If the recorded
device is unplugged or disabled, the recording moves to the new default (waiting up to
``LOST_GRACE_MS`` for Windows to pick one); without one, or with following off, it stops and
what was captured is kept. Outside a recording, a default change refreshes the device line at
once. The 30 ms timer also watches the recorder's health: a writer failure (disk full) stops the
recording (keeping the audio written so far), and a long stretch without audio while the
keep-alive is running shows a warning.

Crash recovery: at startup (``offer_recovery_later``) a recording left in the temp folder by a
crash is offered for recovery (Recover / Delete / Not now); recovering repairs its WAV header
and loads it like a finished recording (IDLE -> LOADING -> STOPPED).

File > Open (Ctrl+O, a file dropped on the window, or a file named on the command line) loads an
existing WAV, MP3, FLAC, OGG or AIFF file as the clip (IDLE/STOPPED -> LOADING -> STOPPED), off
the GUI thread, after the usual Save / Discard / Cancel for an unsaved clip. Any channel count
is mixed to stereo and the file's own rate is kept. An opened file is never deleted or
overwritten by the app: it counts as saved until edited, and Export suggests a new name next to
it. A failed open goes back to the previous clip.

Session end (``commit_session_data``, connected to ``QGuiApplication.commitDataRequest``): when
Windows signs out, restarts or shuts down with an unsaved clip or a recording in progress, the
window asks as it does on close; Save or Cancel cancels the sign-out. If Windows doesn't allow
asking, the temp recording is kept so it is offered for recovery at the next start.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from desktop_audio_recorder import APP_NAME, __version__
from desktop_audio_recorder.audio import recovery
from desktop_audio_recorder.audio.clip import AudioClip
from desktop_audio_recorder.audio.devices import DeviceError, DeviceManager, LoopbackDevice
from desktop_audio_recorder.audio.endpoints import (
    DEVICE_STATE_ACTIVE,
    Endpoint,
    EndpointEvent,
    EventKind,
    describe_state,
)
from desktop_audio_recorder.audio.envelope import EnvelopePyramid
from desktop_audio_recorder.audio.player import ClipPlayer, PlayerError
from desktop_audio_recorder.audio.recorder import (
    LoopbackRecorder,
    RecorderError,
    RecorderState,
    RecordingResult,
    SwitchResult,
    default_temp_dir,
)
from desktop_audio_recorder.editing import commands
from desktop_audio_recorder.editing.history import DEFAULT_MAX_BYTES, Command, NoChange
from desktop_audio_recorder.editing.session import APPLY, REDO, UNDO, EditResult, EditSession
from desktop_audio_recorder.editing.spill import DEFAULT_MAX_DISK_BYTES, SpillStore
from desktop_audio_recorder.export.writers import Cancelled as ExportCancelled
from desktop_audio_recorder.export.writers import ExportResult, export
from desktop_audio_recorder.processing.denoise import Cancelled, DenoiseSettings, denoise
from desktop_audio_recorder.settings import Settings
from desktop_audio_recorder.ui.denoise_dialog import (
    DenoisePlan,
    DenoiseRequest,
    ask_denoise,
    plan_denoise,
)
from desktop_audio_recorder.ui.device_watch import DeviceWatch
from desktop_audio_recorder.ui.edit_toolbar import EditToolbar, ask_gain_db, ask_silence_seconds
from desktop_audio_recorder.ui.export_dialog import (
    OPEN_EXTENSIONS,
    ExportRequest,
    RecoveryChoice,
    RecoveryPrompt,
    UnsavedChoice,
    UnsavedPrompt,
    ask_export,
    ask_open_path,
    ask_recovery,
    ask_unsaved,
    format_size,
    with_extension,
)
from desktop_audio_recorder.ui.level_meter import LevelMeter
from desktop_audio_recorder.ui.state import (
    AppState,
    EditControls,
    StateMachine,
    can_open,
    controls_for,
    edit_controls,
    has_progress,
    is_capturing,
)
from desktop_audio_recorder.ui.transport_bar import PlaybackBar, TransportBar, format_elapsed
from desktop_audio_recorder.ui.waveform_view import WaveformView
from desktop_audio_recorder.ui.workers import ProgressJob, SerialTaskRunner

log = logging.getLogger(__name__)

METER_INTERVAL_MS = 30
PROGRESS_INTERVAL_MS = 100
DENOISE_LABEL = "Noise reduction"
# Edits touching more frames than this run on the loader thread (EDITING state). 100 s of 48 kHz
# stereo is 38 MB: a copy plus a waveform rebuild of that takes ~10-20 ms.
SYNC_EDIT_MAX_FRAMES = 4_800_000
ACTIVATE_REFRESH_MIN_INTERVAL_S = 1.0
# With the keep-alive running, loopback delivers audio continuously; this long without any
# means the capture stream has died.
STALL_WARNING_S = 2.0
# Windows sends a burst of notifications when the default device changes (and people click
# through devices): switch the recording once they have settled.
SWITCH_DELAY_MS = 300
# When the recorded device goes away, how long to wait for Windows to choose a new default.
LOST_GRACE_MS = 1500
_ERROR_STYLE = "QStatusBar { color: rgb(230, 70, 60); }"
_WARNING_STYLE = (
    "QLabel { background: rgb(255, 196, 0); color: rgb(30, 30, 30);"
    " border-radius: 4px; padding: 6px 10px; }"
)


def device_summary(device: LoopbackDevice) -> str:
    rate = device.sample_rate / 1000
    rate_text = f"{rate:g} kHz"
    if device.channels == 2:
        layout = "stereo"
    elif device.channels == 1:
        layout = "mono → stereo"
    else:
        layout = f"{device.channels} ch → stereo"
    return f"{rate_text} · {layout}"


def load_clip(path: str | Path) -> tuple[AudioClip, EnvelopePyramid]:
    """Read a recording and precompute its waveform envelope (runs off the GUI thread)."""
    started = time.perf_counter()
    clip = AudioClip.load(path)
    pyramid = EnvelopePyramid(clip.samples)
    log.info(
        "Loaded %s: %.3f s, %d frames, peak %.3f, in %.0f ms",
        Path(path).name,
        clip.duration,
        clip.frames,
        pyramid.peak,
        (time.perf_counter() - started) * 1000,
    )
    return clip, pyramid


@dataclass(frozen=True, slots=True)
class OpenedFile:
    """Where a clip opened with File > Open came from (the app never deletes this file)."""

    path: Path
    duration: float
    sample_rate: int
    channels: int  # in the file (the clip itself is always stereo)
    format: str  # libsndfile's name: "WAV", "MP3", "FLAC", ...

    @property
    def layout(self) -> str:
        """``"stereo"``, ``"stereo (from mono)"``, ``"stereo (from 6 ch)"``."""
        if self.channels == 2:
            return "stereo"
        return f"stereo (from {'mono' if self.channels == 1 else f'{self.channels} ch'})"


def open_audio(path: str | Path) -> tuple[AudioClip, EnvelopePyramid, OpenedFile]:
    """Read an audio file for File > Open (runs off the GUI thread)."""
    path = Path(path)
    if not path.is_file():  # libsndfile reports this as an unknown format
        raise FileNotFoundError(path)
    info = sf.info(str(path))
    clip, pyramid = load_clip(path)
    if clip.frames == 0:
        raise ValueError("the file has no audio")
    opened = OpenedFile(path, clip.duration, clip.sample_rate, int(info.channels), info.format)
    return clip, pyramid, opened


def describe_open_error(exc: BaseException) -> str:
    """Why a file couldn't be opened, for the status bar."""
    if isinstance(exc, FileNotFoundError):
        return "the file doesn't exist"
    if isinstance(exc, PermissionError):
        return "access denied (is it open in another program?)"
    if isinstance(exc, MemoryError):
        return "not enough memory"
    if isinstance(exc, sf.SoundFileError):
        return "it isn't an audio file this app can read, or it is damaged"
    return describe_error(exc)


def describe_error(exc: BaseException) -> str:
    """One line for the status bar."""
    text = str(exc) or type(exc).__name__
    cause = exc.__cause__
    if isinstance(exc, RecorderError) and cause is not None:
        text = f"{text}: {cause or type(cause).__name__}"
    return text


class MainWindow(QMainWindow):
    def __init__(
        self,
        *,
        recorder: Any | None = None,
        device_manager: Any | None = None,
        runner: SerialTaskRunner | None = None,
        loader: SerialTaskRunner | None = None,
        player: ClipPlayer | None = None,
        clock: Callable[[], float] = time.monotonic,
        sync_edit_frames: int = SYNC_EDIT_MAX_FRAMES,
        max_undo_bytes: int = DEFAULT_MAX_BYTES,
        max_undo_disk_bytes: int = DEFAULT_MAX_DISK_BYTES,
        preview_player: Callable[[], ClipPlayer] = ClipPlayer,
        settings: Settings | None = None,
        device_watch: DeviceWatch | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.resize(960, 620)
        self.setMinimumSize(600, 440)

        self._dm = device_manager or DeviceManager()
        self._recorder = recorder or LoopbackRecorder(self._dm)
        self._runner = runner or SerialTaskRunner(self)
        self._loader = loader or SerialTaskRunner(self, name="loader")
        self._player = player or ClipPlayer()
        self._clip: AudioClip | None = None
        self._session: EditSession | None = None
        self._sync_edit_frames = sync_edit_frames
        self._max_undo_bytes = max_undo_bytes
        self._max_undo_disk_bytes = max_undo_disk_bytes
        self._spill: SpillStore | None = None  # created with the first clip
        self._last_gain_db = 3.0
        self._last_silence_s = 1.0
        # Modal prompts (replaced in tests); ``None`` means cancelled.
        self.prompt_gain_db: Callable[[], float | None] = lambda: ask_gain_db(
            self, self._last_gain_db
        )
        self.prompt_silence_seconds: Callable[[], float | None] = lambda: ask_silence_seconds(
            self, self._last_silence_s
        )
        self._preview_player = preview_player  # makes the dialog's preview player
        self._settings = settings if settings is not None else Settings()
        # The dialog starts from the options used last (persisted in the settings).
        self.denoise_settings: DenoiseSettings = self._settings.denoise_settings
        self.denoise_profile_from_selection: bool = self._settings.denoise_profile_from_selection
        self._job: ProgressJob | None = None  # the running noise reduction or export
        self.prompt_denoise: Callable[[], DenoiseRequest | None] = self._ask_denoise
        self.prompt_export: Callable[[], ExportRequest | None] = self._ask_export
        self.prompt_unsaved: Callable[[UnsavedPrompt], UnsavedChoice] = lambda prompt: ask_unsaved(
            self, prompt
        )
        self.prompt_recovery: Callable[[RecoveryPrompt], RecoveryChoice] = lambda prompt: (
            ask_recovery(self, prompt)
        )
        self._recovered = False  # the clip is a recording recovered after a crash
        self._opened: OpenedFile | None = None  # the clip came from File > Open
        self._opening: Path | None = None  # the file being opened (LOADING)
        self._open_previous: AppState = AppState.IDLE
        self.prompt_open_path: Callable[[], Path | None] = lambda: ask_open_path(
            self, self._settings.open_folder
        )
        self._session_ending = False  # Windows is signing out: close without asking again
        self._keep_for_recovery = False  # keep the temp recording (couldn't ask at session end)
        self._saved_position: int | None = None  # history position at the last export
        self._saved_path: Path | None = None  # the clip's last export
        self._recorded_at: datetime | None = None  # names the export file
        self._clock = clock
        self._sm = StateMachine()
        self._start_previous: AppState = AppState.IDLE
        self._device: LoopbackDevice | None = None
        self._result: RecordingResult | None = None
        self._refresh_in_flight = False
        self._refresh_again = False  # the default changed while a refresh was running
        self._refresh_when_idle = False  # the default changed during a recording
        self._capture_endpoint: Endpoint | None = None  # what the recording captures
        self._warnings: dict[str, str] = {}  # banner texts by kind ("default", "stall")
        self._stop_reason: str | None = None  # why the app stopped the recording itself
        self._last_refresh_request = -1e9
        self._shut_down = False
        self._announce_device = False  # say "Playing on <device>" once the stream runs
        self._last_tick = clock()
        self._follow_default = self._settings.follow_default_device
        self._switching = False  # recorder.switch_device() is running on the audio thread
        self._lost_endpoint: Endpoint | None = None  # gone; waiting to switch away from it
        self._lost_state = ""  # how it went ("unplugged", "disabled", "removed")
        self._switch_timer = QTimer(self)
        self._switch_timer.setSingleShot(True)
        self._switch_timer.setInterval(SWITCH_DELAY_MS)
        self._switch_timer.timeout.connect(self._switch_to_default)
        self._lost_timer = QTimer(self)
        self._lost_timer.setSingleShot(True)
        self._lost_timer.setInterval(LOST_GRACE_MS)
        self._lost_timer.timeout.connect(self._on_lost_grace_over)

        self._build_ui()
        self._watch = device_watch if device_watch is not None else DeviceWatch(self)
        self._watch.changed.connect(self._on_endpoint_event)
        self.playback.loop_button.setChecked(self._settings.loop)
        self.follow_action.blockSignals(True)
        self.follow_action.setChecked(self._follow_default)
        self.follow_action.blockSignals(False)
        geometry = self._settings.window_geometry
        if geometry is not None and not self.restoreGeometry(geometry):
            log.info("Ignoring the saved window geometry")
        self._settings_saved = False

        self._meter_timer = QTimer(self)
        self._meter_timer.setInterval(METER_INTERVAL_MS)
        self._meter_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._meter_timer.timeout.connect(self._tick)
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(PROGRESS_INTERVAL_MS)
        self._progress_timer.timeout.connect(self._show_progress)

        self._apply_state()
        self.show_info("Ready")
        self.refresh_device()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 8)
        root.setSpacing(10)

        device_row = QHBoxLayout()
        self.device_label = QLabel(central)
        bold = QFont(self.device_label.font())
        bold.setBold(True)
        self.device_label.setFont(bold)
        self.device_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.device_detail_label = QLabel(central)
        self.device_detail_label.setEnabled(False)  # secondary text colour
        self.refresh_button = QPushButton("Refresh", central)
        self.refresh_button.setToolTip("Re-detect the Windows default playback device")
        self.refresh_button.clicked.connect(lambda: self.refresh_device(force=True))
        device_row.addWidget(self.device_label)
        device_row.addWidget(self.device_detail_label)
        device_row.addStretch(1)
        device_row.addWidget(self.refresh_button)
        root.addLayout(device_row)

        self.warning_label = QLabel(central)
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet(_WARNING_STYLE)
        self.warning_label.hide()
        root.addWidget(self.warning_label)

        self.transport = TransportBar(central)
        self.transport.record_clicked.connect(self.start_recording)
        self.transport.pause_clicked.connect(self.pause_recording)
        self.transport.resume_clicked.connect(self.resume_recording)
        self.transport.stop_clicked.connect(self.stop_recording)
        root.addWidget(self.transport)

        self.meter = LevelMeter(central)
        root.addWidget(self.meter)

        self.clip_panel = QFrame(central)
        self.clip_panel.setFrameShape(QFrame.Shape.StyledPanel)
        panel_layout = QVBoxLayout(self.clip_panel)
        panel_layout.setContentsMargins(8, 6, 8, 8)
        panel_layout.setSpacing(6)
        self.clip_header = QWidget(self.clip_panel)
        header = QHBoxLayout(self.clip_header)
        header.setContentsMargins(0, 0, 0, 0)
        self.clip_info = QLabel(self.clip_header)
        self.clip_info.setWordWrap(True)
        header.addWidget(self.clip_info, 1)
        self.export_button = QPushButton("Export…", self.clip_header)
        self.export_button.setToolTip("Save the clip as a WAV or MP3 file (Ctrl+S)")
        self.export_button.setMinimumHeight(28)
        self.export_button.clicked.connect(lambda: self.export_clip())
        self.export_button.hide()
        header.addWidget(self.export_button, 0, Qt.AlignmentFlag.AlignTop)
        panel_layout.addWidget(self.clip_header, 1)
        self.edit_toolbar = EditToolbar(self.clip_panel)
        self._connect_edit_actions()
        self.edit_toolbar.hide()
        panel_layout.addWidget(self.edit_toolbar)
        self.waveform = WaveformView(self.clip_panel)
        self.waveform.seek_requested.connect(self.seek)
        self.waveform.selection_changed.connect(self._on_selection_changed)
        self.waveform.hide()
        panel_layout.addWidget(self.waveform, 1)
        self.playback = PlaybackBar(self.clip_panel)
        self.playback.play_clicked.connect(self.play)
        self.playback.pause_clicked.connect(self.pause_playback)
        self.playback.stop_clicked.connect(self.stop_playback)
        self.playback.loop_toggled.connect(self._on_loop_toggled)
        panel_layout.addWidget(self.playback)
        root.addWidget(self.clip_panel, 1)

        self.setCentralWidget(central)
        self._set_device(None, pending=True)

        file_menu = self.menuBar().addMenu("&File")
        self.open_action = QAction("&Open…", self)
        self.open_action.setShortcut(QKeySequence(QKeySequence.StandardKey.Open))
        self.open_action.setStatusTip("Open a WAV, MP3, FLAC, OGG or AIFF file to edit it")
        self.open_action.triggered.connect(lambda: self.open_file())
        file_menu.addAction(self.open_action)
        # Ctrl+S is the window-wide shortcut below; the menu only shows it.
        self.export_action = QAction("&Export…\tCtrl+S", self)
        self.export_action.setStatusTip("Save the clip as a WAV or MP3 file")
        self.export_action.triggered.connect(lambda: self.export_clip())
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        self.exit_action = QAction("E&xit", self)
        self.exit_action.triggered.connect(self.close)
        file_menu.addAction(self.exit_action)
        recording_menu = self.menuBar().addMenu("&Recording")
        self.follow_action = QAction("&Follow the default playback device", self)
        self.follow_action.setCheckable(True)
        self.follow_action.setStatusTip(
            "When Windows switches the default playback device during a recording, carry on"
            " recording from the new device"
        )
        self.follow_action.toggled.connect(self._on_follow_toggled)
        recording_menu.addAction(self.follow_action)
        self.setAcceptDrops(True)

        # Noise reduction / export progress, shown in the PROCESSING and EXPORTING states.
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFixedWidth(180)
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.cancel_processing)
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.statusBar().addPermanentWidget(self.cancel_button)

        # Window-wide shortcuts (they take precedence over a focused button's Space).
        self.play_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self.play_shortcut.activated.connect(self.toggle_playback)
        self.home_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Home), self)
        self.home_shortcut.activated.connect(self.go_to_start)
        self.export_shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.Save), self)
        self.export_shortcut.activated.connect(lambda: self.export_clip())

    def _connect_edit_actions(self) -> None:
        bar = self.edit_toolbar
        bar.delete_action.triggered.connect(self.delete_selection)
        bar.crop_action.triggered.connect(self.crop_to_selection)
        bar.silence_action.triggered.connect(self.silence_selection)
        bar.insert_silence_action.triggered.connect(lambda: self.insert_silence())
        bar.fade_in_action.triggered.connect(self.fade_in)
        bar.fade_out_action.triggered.connect(self.fade_out)
        bar.gain_action.triggered.connect(lambda: self.apply_gain())
        bar.normalize_action.triggered.connect(self.normalize)
        bar.denoise_action.triggered.connect(lambda: self.reduce_noise())
        bar.undo_action.triggered.connect(self.undo)
        bar.redo_action.triggered.connect(self.redo)

    # --------------------------------------------------------- public state

    @property
    def state(self) -> AppState:
        return self._sm.state

    @property
    def result(self) -> RecordingResult | None:
        """The last finished (unsaved) recording, if any."""
        return self._result

    @property
    def device(self) -> LoopbackDevice | None:
        return self._device

    @property
    def clip(self) -> AudioClip | None:
        """The loaded clip (after a recording stops and its WAV has loaded)."""
        return self._clip

    @property
    def player(self) -> ClipPlayer:
        return self._player

    @property
    def session(self) -> EditSession | None:
        """The clip's edit session (undo history), while a clip is loaded."""
        return self._session

    # ------------------------------------------------------------ status bar

    def show_info(self, message: str) -> None:
        self.statusBar().setStyleSheet("")
        self.statusBar().showMessage(message)

    def show_error(self, message: str) -> None:
        log.warning("Status error: %s", message)
        self.statusBar().setStyleSheet(_ERROR_STYLE)
        self.statusBar().showMessage(message)

    @property
    def status_text(self) -> str:
        return self.statusBar().currentMessage()

    @property
    def status_is_error(self) -> bool:
        return bool(self.statusBar().styleSheet())

    @property
    def warning_text(self) -> str:
        """The warning banner's text ("" when hidden)."""
        return self.warning_label.text() if self._warnings else ""

    def set_warning(self, kind: str, text: str) -> None:
        """Show (or update) a warning in the banner above the transport."""
        if self._warnings.get(kind) != text:
            log.warning("Warning (%s): %s", kind, text)
        self._warnings[kind] = text
        self._show_warnings()

    def clear_warning(self, kind: str | None = None) -> None:
        """Remove one warning (or all of them)."""
        if kind is None:
            self._warnings.clear()
        else:
            self._warnings.pop(kind, None)
        self._show_warnings()

    def _show_warnings(self) -> None:
        self.warning_label.setText("\n".join(self._warnings.values()))
        self.warning_label.setVisible(bool(self._warnings))

    # ------------------------------------------------------------ state machine

    def _go(self, target: AppState) -> None:
        previous = self._sm.go(target)
        log.debug("State %s -> %s", previous.value, target.value)
        self._apply_state()
        if self._refresh_when_idle and not is_capturing(target) and controls_for(target).refresh:
            self._refresh_when_idle = False  # the default changed during the recording
            QTimer.singleShot(0, lambda: self.refresh_device(force=True))

    def _apply_state(self) -> None:
        state = self._sm.state
        controls = controls_for(state)
        self.transport.set_state(state)
        self.playback.set_state(state)
        self.refresh_button.setEnabled(controls.refresh)
        self.waveform.setEnabled(controls.clip)
        self.progress_bar.setVisible(controls.cancel)
        self.cancel_button.setVisible(controls.cancel)
        self.cancel_button.setEnabled(controls.cancel)
        self.cancel_button.setToolTip(
            "Stop the export; no file is written"
            if state is AppState.EXPORTING
            else "Stop the noise reduction; the clip stays as it was"
        )
        self.export_button.setEnabled(controls.export)
        self.export_action.setEnabled(controls.export)
        self.open_action.setEnabled(can_open(state))
        self._update_edit_controls()
        prefix = {AppState.RECORDING: "● ", AppState.PAUSED: "Paused - "}.get(state, "")
        self.setWindowTitle(f"{prefix}{APP_NAME} {__version__}")
        self._update_clip_info()

    def _update_clip_info(self) -> None:
        state = self._sm.state
        has_clip = self._clip is not None and state in (
            AppState.STOPPED,
            AppState.PLAYING,
            AppState.EDITING,
            AppState.PROCESSING,
            AppState.EXPORTING,
        )
        if has_clip and (self._result is not None or self._opened is not None):
            history = self._session.history if self._session is not None else None
            if self._saved_path is None:
                unchanged = self._opened is not None and not self.has_unsaved_changes
                saved = "unchanged" if unchanged else "not saved yet"
            elif self.has_unsaved_changes:
                saved = f"changed since saved as {self._saved_path.name}"
            else:
                saved = f"saved as {self._saved_path.name}"
            edited = history is not None and (history.can_undo or history.evicted)
            if self._opened is not None:
                o = self._opened
                rate = f"{o.sample_rate / 1000:g} kHz {o.layout}"
                if edited:
                    text = (
                        f"Edited: {format_elapsed(self._clip.duration)} · {rate}"
                        f" · opened {o.path.name} ({format_elapsed(o.duration)}) · {saved}"
                    )
                else:
                    text = f"Opened {o.path.name} · {format_elapsed(o.duration)} · {rate} · {saved}"
            else:
                r = self._result
                rate = f"{r.sample_rate / 1000:g} kHz stereo"
                if edited:
                    text = (
                        f"Edited: {format_elapsed(self._clip.duration)} · {rate}"
                        f" · recorded {format_elapsed(r.duration)} · {saved}"
                    )
                else:
                    kind = "Recovered recording" if self._recovered else "Recorded"
                    text = f"{kind} {format_elapsed(r.duration)} · {rate} · {saved}"
        elif state is AppState.LOADING and self._opening is not None:
            text = f"Opening {self._opening.name}…"
        elif state is AppState.LOADING:
            text = "Loading the recording…"
        elif state in (AppState.RECORDING, AppState.STARTING):
            text = "Recording… the clip will appear here when you stop."
        elif state is AppState.PAUSED:
            text = "Paused. Resume to continue the same recording, or Stop to finish it."
        elif state is AppState.STOPPING:
            text = "Finishing the recording…"
        else:
            text = "Press Record to capture whatever is playing on the default playback device."
        self.clip_info.setText(text)
        # With a clip the label is a one-line caption above the waveform; otherwise a message
        # centred in the panel.
        loaded = self._clip is not None
        self.waveform.setVisible(loaded)
        self.edit_toolbar.setVisible(loaded)
        self.export_button.setVisible(loaded)
        self.clip_panel.layout().setStretchFactor(self.clip_header, 0 if loaded else 1)
        self.clip_info.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            if loaded
            else Qt.AlignmentFlag.AlignCenter
        )

    # ------------------------------------------------------------ device line

    def _set_device(self, device: LoopbackDevice | None, *, pending: bool = False) -> None:
        self._device = device
        if device is not None:
            self.device_label.setText(f"Recording from: {device.output_name}")
            self.device_label.setStyleSheet("")
            self.device_detail_label.setText(device_summary(device))
            self.device_label.setToolTip(
                f"{device.name}\n{device.sample_rate} Hz, {device.channels} channel(s),"
                " recorded as stereo"
            )
        elif pending:
            self.device_label.setText("Recording from: detecting…")
            self.device_label.setStyleSheet("")
            self.device_detail_label.setText("")
        else:
            self.device_label.setText("Recording from: no playback device found")
            self.device_label.setStyleSheet("color: rgb(230, 70, 60);")
            self.device_detail_label.setText("")
            self.device_label.setToolTip("")

    def refresh_device(self, force: bool = False) -> None:
        """Re-resolve the default playback device on the audio thread (not while recording)."""
        if self._shut_down:
            return
        if self._refresh_in_flight:
            if force:
                self._refresh_again = True  # it may have read the old default
            return
        if not controls_for(self._sm.state).refresh:
            return
        now = self._clock()
        if not force and now - self._last_refresh_request < ACTIVATE_REFRESH_MIN_INTERVAL_S:
            return
        self._last_refresh_request = now
        self._refresh_in_flight = True
        self._runner.submit(
            self._dm.resolve_default_loopback, self._on_device_resolved, self._on_device_error
        )

    def _refresh_finished(self) -> None:
        self._refresh_in_flight = False
        if self._refresh_again:
            self._refresh_again = False
            QTimer.singleShot(0, lambda: self.refresh_device(force=True))

    def _on_device_resolved(self, device: LoopbackDevice) -> None:
        self._refresh_finished()
        if is_capturing(self._sm.state):
            return  # the label shows the device being recorded
        changed = self._device is None or self._device != device
        self._set_device(device)
        if changed:
            log.info("Default playback device: %s", device.output_name)
        if self.status_is_error and self._sm.state is AppState.IDLE:
            self.show_info("Ready")

    def _on_device_error(self, exc: BaseException) -> None:
        self._refresh_finished()
        if is_capturing(self._sm.state):
            return
        self._set_device(None)
        if isinstance(exc, DeviceError):
            self.show_error(f"No playback device: {describe_error(exc)}")
        else:
            log.error("Device refresh failed", exc_info=exc)
            self.show_error(f"Could not query audio devices: {describe_error(exc)}")

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            self.refresh_device()

    # ------------------------------------------------------------ device changes

    @property
    def device_watch(self) -> DeviceWatch:
        return self._watch

    def _recording_live(self) -> bool:
        return self._sm.state in (AppState.RECORDING, AppState.PAUSED)

    def _on_endpoint_event(self, event: EndpointEvent) -> None:
        """A Windows endpoint change (GUI thread, queued from Core Audio's thread)."""
        if self._shut_down:
            return
        log.info("Audio endpoint %s: %s", event.kind.value, event.device_id)
        captured = self._capture_endpoint
        if event.kind is EventKind.DEFAULT_CHANGED:
            if self._recording_live() and captured is not None:
                self._check_default(event.device_id, captured)
            elif is_capturing(self._sm.state):
                self._refresh_when_idle = True  # starting or stopping: catch up afterwards
            else:
                self.refresh_device(force=True)
        elif event.kind in (EventKind.STATE_CHANGED, EventKind.REMOVED):
            gone = event.kind is EventKind.REMOVED or event.state != DEVICE_STATE_ACTIVE
            if (
                gone
                and captured is not None
                and event.device_id == captured.id
                and self._recording_live()
            ):
                state = "removed" if event.state is None else describe_state(event.state)
                self._capture_device_lost(captured, state)

    def _check_default(self, default_id: str, captured: Endpoint) -> None:
        if default_id == captured.id:
            self._switch_timer.stop()  # switched back before the recording followed
            if self._lost_endpoint is None:
                self.clear_warning("default")
            return
        self._refresh_when_idle = True
        new = self._watch.endpoint(default_id) if default_id else None
        if new is not None and self._follow_default:
            self._switch_timer.start()  # (re)start: switch once the changes have settled
            return
        if new is None:
            text = (
                "Windows has no default playback device any more. This recording continues"
                f" from {captured.name}."
            )
        else:
            text = (
                f"Windows switched the default playback device to {new.name}. This recording"
                f" continues from {captured.name}; to record {new.name}, stop and start a new"
                " recording (or turn on Recording > Follow the default playback device)."
            )
        self.set_warning("default", text)
        QApplication.alert(self)

    def _capture_device_lost(self, captured: Endpoint, state: str) -> None:
        if self._stop_reason is not None or self._lost_endpoint is not None:
            return
        if self._follow_default:
            # Move to the new default; Windows may announce it just before or after this.
            self._lost_endpoint = captured
            self._lost_state = state
            self.set_warning(
                "default",
                f"{captured.name} was disconnected ({state}). Moving the recording to the new"
                " default playback device…",
            )
            QApplication.alert(self)
            self._lost_timer.start()
            self._switch_to_default()
            return
        self._stop_reason = f"{captured.name} was disconnected ({state}), so the recording stopped"
        QApplication.alert(self)
        self.stop_recording()

    def _on_lost_grace_over(self) -> None:
        """The recorded device went away and no new default turned up in time: stop."""
        lost, self._lost_endpoint = self._lost_endpoint, None
        if lost is None or self._shut_down or not self._recording_live():
            return
        if self._switching:  # a switch is under way after all: let it finish
            self._lost_endpoint = lost
            self._lost_timer.start()
            return
        if self._stop_reason is None:
            self._stop_reason = (
                f"{lost.name} was disconnected ({self._lost_state}), so the recording stopped"
            )
            self.stop_recording()

    def _on_follow_toggled(self, on: bool) -> None:
        self._follow_default = on
        try:
            self._settings.follow_default_device = on
        except Exception:
            log.warning("Could not save the setting", exc_info=True)
        if not on:
            self._switch_timer.stop()
        captured = self._capture_endpoint
        default = self._watch.default_render()
        if self._recording_live() and captured is not None and default is not None:
            self._check_default(default.id, captured)

    def _switch_to_default(self) -> None:
        """Move the recording to the Windows default playback device (if it differs)."""
        if self._shut_down or not self._recording_live():
            return
        if self._switching:
            self._switch_timer.start()  # look again once the running switch is done
            return
        default = self._watch.default_render()
        captured = self._capture_endpoint
        if default is None or (captured is not None and default.id == captured.id):
            return  # nothing to switch to (yet: a lost device waits for its grace period)
        self._switching = True
        self.show_info(f"Moving the recording to {default.name}…")
        self._runner.submit(self._recorder.switch_device, self._on_switched, self._on_switch_failed)

    def _on_switched(self, result: SwitchResult) -> None:
        self._switching = False
        if self._shut_down or not self._recording_live():
            return
        self._set_device(result.device)
        self._capture_endpoint = self._watch.find_render(result.device.output_name)
        if result.error is not None:
            if self._lost_endpoint is not None:
                return  # the grace timer stops the recording if nothing better turns up
            self.set_warning(
                "default",
                f"Couldn't move the recording to the new default playback device"
                f" ({result.error}). It continues from {result.device.output_name}.",
            )
            return
        self._lost_endpoint = None
        self._lost_timer.stop()
        self.clear_warning("default")
        self.show_info(
            f"Now recording from {result.device.output_name}"
            f" (moved from {result.previous.output_name} at {format_elapsed(result.at)})"
        )
        default = self._watch.default_render()  # changed again in the meantime?
        captured = self._capture_endpoint
        if default is not None and captured is not None and default.id != captured.id:
            self._switch_timer.start()

    def _on_switch_failed(self, exc: BaseException) -> None:
        self._switching = False
        self._lost_endpoint = None
        self._lost_timer.stop()
        if self._shut_down or not self._recording_live():
            return
        log.error("Could not switch the recording", exc_info=exc)
        if self._stop_reason is None:
            self._stop_reason = (
                f"Could not record from the new default playback device"
                f" ({describe_error(exc)}), so the recording stopped"
            )
            QApplication.alert(self)
            self.stop_recording()

    def _check_recorder_health(self) -> None:
        """30 ms tick while recording: writer failure, or no audio arriving."""
        failure = getattr(self._recorder, "failure", None)
        if failure and self._stop_reason is None:
            self._stop_reason = f"Recording stopped: {failure}"
            self.stop_recording()
            return
        if self._sm.state is not AppState.RECORDING:
            return
        gap = float(getattr(self._recorder, "data_gap", 0.0) or 0.0)
        if getattr(self._recorder, "keepalive_active", False) and gap >= STALL_WARNING_S:
            name = self._device.output_name if self._device is not None else "the device"
            self.set_warning(
                "stall",
                f"No audio has arrived from {name} for {gap:.0f} s. Is the device still"
                " connected? The recording is filling the gap with silence.",
            )
        elif gap < 0.5 and "stall" in self._warnings:
            self.clear_warning("stall")

    def _end_capture(self) -> None:
        """The recording is over (stopped or failed): drop its warnings, catch up on changes."""
        self._capture_endpoint = None
        self._lost_endpoint = None
        self._switch_timer.stop()
        self._lost_timer.stop()
        self.clear_warning()  # a pending device refresh runs from _go() once it's allowed

    # ------------------------------------------------------------ transport

    def start_recording(self) -> None:
        """Record (asks to save an unsaved clip first: the new recording replaces it)."""
        if self._sm.state is AppState.PLAYING:
            self.stop_playback()  # the Record button is disabled while playing; be safe anyway
        if not self._sm.can(AppState.STARTING) or self._shut_down:
            return
        if self.has_unsaved_changes:
            choice = self.prompt_unsaved(
                UnsavedPrompt(
                    closing=False,
                    can_save=True,
                    text="Save the current clip before recording a new one?",
                    detail="The new recording replaces it, so unsaved audio and edits are lost.",
                )
            )
            if choice is UnsavedChoice.SAVE:
                self.export_clip(then=self.start_recording)
                return
            if choice is not UnsavedChoice.DISCARD or not self._sm.can(AppState.STARTING):
                return
            # Discard: the clip goes once the new recording has started (a failed start keeps it).
        self._start_previous = self._sm.state
        self._go(AppState.STARTING)
        self.transport.set_elapsed(0.0)
        self.meter.reset()
        self.show_info("Starting…")
        self._runner.submit(self._recorder.start, self._on_started, self._on_start_failed)

    def _on_started(self, device: LoopbackDevice) -> None:
        self._go(AppState.RECORDING)
        self._set_device(device)
        self._stop_reason = None
        self._capture_endpoint = self._watch.find_render(device.output_name)
        if self._capture_endpoint is None and self._watch.available:
            log.warning("No endpoint named %r: device changes aren't watched", device.output_name)
        previous, self._result = self._result, None
        self._recovered = False
        self._opened = None
        self._set_clip(None)
        if previous is not None:
            self._discard(previous)
        self.show_info(f"Recording from {device.output_name}")
        self._last_tick = self._clock()
        self._meter_timer.start()
        default = self._watch.default_render()
        if self._capture_endpoint is not None and default is not None:
            self._check_default(default.id, self._capture_endpoint)  # changed while starting

    def _on_start_failed(self, exc: BaseException) -> None:
        self._go(self._start_previous)
        if isinstance(exc, DeviceError):
            self._set_device(None)
            self.show_error(f"Cannot record: {describe_error(exc)}")
        else:
            log.error("Could not start recording", exc_info=exc)
            self.show_error(f"Could not start recording: {describe_error(exc)}")

    def pause_recording(self) -> None:
        if self._sm.state is not AppState.RECORDING:
            return
        try:
            self._recorder.pause()
        except RecorderError as exc:
            self.show_error(f"Could not pause: {describe_error(exc)}")
            return
        self._go(AppState.PAUSED)
        self.show_info("Paused")

    def resume_recording(self) -> None:
        if self._sm.state is not AppState.PAUSED:
            return
        try:
            self._recorder.resume()
        except RecorderError as exc:
            self.show_error(f"Could not resume: {describe_error(exc)}")
            return
        self._go(AppState.RECORDING)
        self.show_info("Recording")

    def stop_recording(self) -> None:
        if not self._sm.can(AppState.STOPPING):
            return
        self._go(AppState.STOPPING)
        self.show_info("Finishing the recording…")
        self._runner.submit(self._recorder.stop, self._on_stopped, self._on_stop_failed)

    def _on_stopped(self, result: RecordingResult) -> None:
        self._result = result
        self._recorded_at = datetime.now()
        self._go(AppState.LOADING)
        self.transport.set_elapsed(result.duration)
        self._end_capture()
        message = f"Recorded {format_elapsed(result.duration)} from {result.device_names}"
        if result.zero_filled_frames:
            message += f" ({result.zero_filled_frames / result.sample_rate:.1f} s of gaps filled)"
        reason, self._stop_reason = self._stop_reason, None
        if result.error is not None:
            reason = f"Recording stopped early: {result.error}"
        if reason is not None:
            self.show_error(f"{reason}. Kept {format_elapsed(result.duration)}.")
        else:
            self.show_info(message)
        path = result.path
        self._loader.submit(
            lambda: load_clip(path),
            lambda loaded: self._on_loaded(result, loaded),
            lambda exc: self._on_load_failed(result, exc),
        )

    def _on_loaded(
        self, result: RecordingResult, loaded: tuple[AudioClip, EnvelopePyramid]
    ) -> None:
        if self._shut_down or result is not self._result or self.state is not AppState.LOADING:
            return
        clip, pyramid = loaded
        self._set_clip(clip, pyramid)
        self._go(AppState.STOPPED)

    def _on_load_failed(self, result: RecordingResult, exc: BaseException) -> None:
        if self._shut_down or result is not self._result or self.state is not AppState.LOADING:
            return
        log.error("Could not load the recording", exc_info=exc)
        # Keep ``_result`` so the temp file is still cleaned up later.
        self._go(AppState.IDLE)
        self.show_error(f"Could not load the recording: {describe_error(exc)}")

    def _on_stop_failed(self, exc: BaseException) -> None:
        log.error("Recording failed", exc_info=exc)
        self._end_capture()
        self._stop_reason = None
        self._go(AppState.IDLE)
        self.transport.set_elapsed(0.0)
        self.show_error(f"Recording failed: {describe_error(exc)}")

    # ------------------------------------------------------------ playback

    def _set_clip(self, clip: AudioClip | None, pyramid: EnvelopePyramid | None = None) -> None:
        """Show a new clip (resets cursor, selection, zoom, the undo history, saved state)."""
        self._clip = clip
        self._saved_position = None
        self._saved_path = None
        if self._session is not None:
            self._session.close()  # its undo data (and files) go now, not at garbage collection
        if clip is not None and clip.frames:
            self._session = EditSession(
                clip, pyramid, max_bytes=self._max_undo_bytes, spill=self._spill_store()
            )
            pyramid = self._session.pyramid
        else:
            self._session = None
        self._player.set_clip(clip)
        self._player.loop = self.playback.loop
        self.waveform.set_clip(clip, pyramid)
        self.playback.set_selection(None)
        self.playback.set_position(0.0, clip.duration if clip is not None else 0.0)
        self._update_edit_controls()
        self._update_clip_info()

    def play(self) -> None:
        if self._sm.state is not AppState.STOPPED or self._clip is None:
            return
        try:
            self._player.play()
        except PlayerError as exc:
            log.error("Could not start playback", exc_info=exc)
            self.show_error(describe_error(exc))
            return
        self._go(AppState.PLAYING)
        # The stream opens on the stream thread; name the device once it's running.
        self._announce_device = True
        self._announce_playback()
        self._last_tick = self._clock()
        self._meter_timer.start()

    def _announce_playback(self) -> None:
        if not self._announce_device:
            return
        if self._player.streaming:
            self._announce_device = False
            self.show_info(f"Playing on {self._player.device_name or 'the default output'}")
        elif self.status_text != "Playing…":
            self.show_info("Playing…")

    def pause_playback(self) -> None:
        if self._sm.state is not AppState.PLAYING:
            return
        self._player.pause()
        self._go(AppState.STOPPED)
        self._show_position()
        self.show_info("Playback paused")

    def stop_playback(self) -> None:
        if self._sm.state is not AppState.PLAYING:
            return
        self._player.stop()
        self._go(AppState.STOPPED)
        self._show_position(follow=True)
        self.show_info("Playback stopped")

    def toggle_playback(self) -> None:
        """Space: play or pause the clip."""
        if self._sm.state is AppState.PLAYING:
            self.pause_playback()
        elif self._sm.state is AppState.STOPPED:
            self.play()

    def go_to_start(self) -> None:
        """Home: move the cursor (or the playing position) to the start of the clip."""
        self.seek(0.0)

    def seek(self, seconds: float) -> None:
        if self._clip is None or not controls_for(self._sm.state).clip:
            return
        self._player.seek(seconds)
        self._show_position(target=self._clip.clamp_seconds(seconds), follow=True)

    def _on_selection_changed(self, selection: tuple[float, float] | None) -> None:
        if selection is None:
            self._player.set_selection(None)
        else:
            self._player.set_selection(*selection)
        self.playback.set_selection(selection)
        self._update_edit_controls()

    def _on_loop_toggled(self, checked: bool) -> None:
        self._player.loop = checked
        self._settings.loop = checked

    def _show_position(self, target: float | None = None, follow: bool = False) -> None:
        if self._clip is None:
            return
        position = self._player.position if target is None else target
        self.waveform.set_playhead(position, follow=follow)
        self.playback.set_position(position, self._clip.duration)

    def _poll_player(self) -> None:
        self._announce_playback()
        if self._player.update():  # reached the end, the device went away or didn't open
            self._announce_device = False
            self._go(AppState.STOPPED)
            if self._player.last_error:
                self.show_error(self._player.last_error)
            else:
                self.show_info("Playback finished")
        self._show_position(follow=True)

    # ------------------------------------------------------------ editing

    @property
    def edit_controls(self) -> EditControls:
        session = self._session
        return edit_controls(
            self._sm.state if session is not None else AppState.IDLE,
            has_selection=self._selection_frames() is not None,
            can_undo=session is not None and session.history.can_undo,
            can_redo=session is not None and session.history.can_redo,
        )

    def _update_edit_controls(self) -> None:
        session = self._session
        self.edit_toolbar.set_controls(self.edit_controls)
        if session is None:
            self.edit_toolbar.set_history(None, None)
        else:
            self.edit_toolbar.set_history(session.history.undo_label, session.history.redo_label)

    def _selection_frames(self) -> tuple[int, int] | None:
        selection, clip = self.waveform.selection, self._clip
        if selection is None or clip is None:
            return None
        a, b = clip.frame_range(*selection)
        return (a, b) if b > a else None

    def _target_frames(self) -> tuple[int, int, bool]:
        """``(start, stop, selected)``: the selection, or the whole clip without one."""
        selection = self._selection_frames()
        if selection is not None:
            return selection[0], selection[1], True
        return 0, self._clip.frames, False

    def delete_selection(self) -> None:
        """Del: remove the selection."""
        if not self.edit_controls.delete:
            return
        a, b = self._selection_frames()
        if (a, b) == (0, self._clip.frames):
            self.show_info("Can't delete the whole clip (use Record to start over)")
            return
        self._edit(APPLY, commands.Delete(a, b))

    def crop_to_selection(self) -> None:
        if self.edit_controls.crop:
            self._edit(APPLY, commands.Crop(*self._selection_frames()))

    def silence_selection(self) -> None:
        if self.edit_controls.silence:
            self._edit(APPLY, commands.Silence(*self._selection_frames()))

    def fade_in(self) -> None:
        if self.edit_controls.fade_in:
            self._edit(APPLY, commands.FadeIn(*self._selection_frames()))

    def fade_out(self) -> None:
        if self.edit_controls.fade_out:
            self._edit(APPLY, commands.FadeOut(*self._selection_frames()))

    def apply_gain(self, db: float | None = None) -> None:
        """Gain on the selection (or the whole clip); asks for the dB amount if not given."""
        if not self.edit_controls.gain:
            return
        if db is None:
            db = self.prompt_gain_db()
            if db is None or not self.edit_controls.gain:  # cancelled, or the state changed
                return
            self._last_gain_db = db
        a, b, selected = self._target_frames()
        self._edit(APPLY, commands.Gain(a, b, db, selected=selected))

    def normalize(self) -> None:
        """Normalize the selection (or the whole clip) to a -1 dBFS peak."""
        if self.edit_controls.normalize:
            a, b, selected = self._target_frames()
            self._edit(APPLY, commands.Normalize(a, b, selected=selected))

    def insert_silence(self, seconds: float | None = None) -> None:
        """Insert silence at the selection start, or at the cursor without a selection."""
        if not self.edit_controls.insert_silence:
            return
        if seconds is None:
            seconds = self.prompt_silence_seconds()
            if seconds is None or not self.edit_controls.insert_silence:
                return
            self._last_silence_s = seconds
        clip = self._clip
        frames = round(seconds * clip.sample_rate)
        if frames <= 0:
            return
        selection = self._selection_frames()
        if selection is not None:
            at = selection[0]
        else:
            if self._sm.state is AppState.PLAYING:
                self.pause_playback()
            at = min(self._player.position_frame, clip.frames)
        self._edit(APPLY, commands.InsertSilence(at, frames, clip.sample_rate))

    def undo(self) -> None:
        """Ctrl+Z."""
        if self.edit_controls.undo:
            self._edit(UNDO)

    def redo(self) -> None:
        """Ctrl+Y / Ctrl+Shift+Z."""
        if self.edit_controls.redo:
            self._edit(REDO)

    def _edit(self, action: str, command: Command | None = None) -> None:
        session = self._session
        if session is None or self._shut_down:
            return
        if self._sm.state is AppState.PLAYING:
            self.pause_playback()
        if self._sm.state is not AppState.STOPPED:
            return
        if action == APPLY:
            label = command.label
        elif action == UNDO:
            label = f"Undo {session.history.undo_label}"
        else:
            label = f"Redo {session.history.redo_label}"
        if session.work_frames(action, command) <= self._sync_edit_frames:
            try:
                result = session.run(action, command)
            except Exception as exc:
                self._on_edit_failed(session, label, exc)
                return
            self._commit_edit(result)
            return
        self._go(AppState.EDITING)
        self.show_info(f"{label}…")
        self._loader.submit(
            lambda: session.run(action, command),
            lambda result: self._on_edit_done(session, result),
            lambda exc: self._on_edit_failed(session, label, exc),
        )

    def _on_edit_done(self, session: EditSession, result: EditResult) -> None:
        if self._shut_down or session is not self._session:
            return
        if self._sm.state is AppState.EDITING:
            self._go(AppState.STOPPED)
        self._commit_edit(result)

    def _on_edit_failed(self, session: EditSession, label: str, exc: BaseException) -> None:
        if self._shut_down or session is not self._session:
            return
        if self._sm.state is AppState.EDITING:
            self._go(AppState.STOPPED)
        if isinstance(exc, NoChange):
            self.show_info(str(exc))
            return
        log.error("Edit failed: %s", label, exc_info=exc)
        reason = "not enough memory" if isinstance(exc, MemoryError) else describe_error(exc)
        self.show_error(f"Could not {label[0].lower()}{label[1:]}: {reason}")

    def _commit_edit(self, result: EditResult) -> None:
        """Show the edited clip, keeping the view (see ``WaveformView.replace_clip``)."""
        clip, hint = result.clip, result.hint
        rate = clip.sample_rate
        self._clip = clip
        self._player.replace_clip(clip, hint.cursor)
        self.waveform.replace_clip(clip, result.pyramid, shift=hint.shift / rate)
        selection = hint.selection
        self.waveform.set_selection(
            None if selection is None else (selection[0] / rate, selection[1] / rate)
        )
        self._on_selection_changed(self.waveform.selection)
        self._show_position(follow=True)
        self._update_clip_info()
        self.show_info(result.message)

    # ------------------------------------------------------------ noise reduction

    @property
    def job(self) -> ProgressJob | None:
        """The running noise reduction's or export's progress/cancel handle, if any."""
        return self._job

    def _ask_denoise(self) -> DenoiseRequest | None:
        return ask_denoise(
            self,
            self._clip,
            self._selection_frames(),
            self.denoise_settings,
            self.denoise_profile_from_selection,
            runner=self._loader,
            player_factory=self._preview_player,
        )

    def reduce_noise(self, request: DenoiseRequest | None = None) -> None:
        """Noise reduction on the selection or the whole clip (see ``DenoiseDialog``); asks
        for the options if not given. Runs on the loader thread with progress and Cancel."""
        if not self.edit_controls.denoise or self._shut_down:
            return
        if self._sm.state is AppState.PLAYING:
            self.pause_playback()
        if request is None:
            request = self.prompt_denoise()
            if request is None or not self.edit_controls.denoise:  # cancelled / state changed
                return
            self.denoise_settings = request.settings
            self.denoise_profile_from_selection = request.profile_from_selection
            self._settings.denoise_settings = request.settings
            self._settings.denoise_profile_from_selection = request.profile_from_selection
        if self._sm.state is not AppState.STOPPED:
            return
        settings, clip = request.settings, self._clip
        if settings.is_noop:
            self.show_info("Noise reduction: these settings change nothing")
            return
        try:
            plan = plan_denoise(
                clip.frames,
                clip.sample_rate,
                self._selection_frames(),
                settings,
                request.profile_from_selection,
            )
        except ValueError as exc:
            self.show_info(str(exc))
            return
        self._start_processing(plan, settings)

    def _start_processing(self, plan: DenoisePlan, settings: DenoiseSettings) -> None:
        session = self._session
        samples, rate = session.clip.samples, session.clip.sample_rate
        profile = None
        if plan.profile is not None:
            profile = samples[plan.profile[0] : plan.profile[1]].copy()
        job = self._job = ProgressJob()
        started = time.perf_counter()

        def task() -> EditResult:  # loader thread; the GUI leaves the session alone meanwhile
            target = samples[plan.start : plan.stop]
            processed = denoise(target, rate, settings, profile, job.report, job.is_cancelled)
            if job.is_cancelled():
                raise Cancelled("Noise reduction cancelled")
            command = commands.ReplaceRegion(
                plan.start, plan.stop, processed, DENOISE_LABEL, selected=plan.selected
            )
            return session.apply(command)

        self._go(AppState.PROCESSING)
        self.progress_bar.setValue(0)
        self.show_info(f"Reducing noise ({settings.describe()})…")
        self._progress_timer.start()
        self._loader.submit(
            task,
            lambda result: self._on_processing_done(session, job, settings, started, result),
            lambda exc: self._on_processing_failed(session, job, exc),
        )

    def cancel_processing(self) -> None:
        """Cancel: the noise reduction or export stops at its next chunk; the clip is left as it
        was and no file is written."""
        if self._job is None or not has_progress(self._sm.state):
            return
        self._job.cancel()
        self.cancel_button.setEnabled(False)
        self.show_info("Cancelling…")

    def _show_progress(self) -> None:
        job = self._job
        if job is None or not has_progress(self._sm.state):
            self._progress_timer.stop()
            return
        self.progress_bar.setValue(round(job.fraction * 100))

    def _end_processing(self, session: EditSession, job: ProgressJob) -> bool:
        """Leave PROCESSING/EXPORTING; ``False`` if the result is stale (or the window closed)."""
        if job is not self._job:
            return False
        self._job = None
        self._progress_timer.stop()
        if self._shut_down or session is not self._session:
            return False
        if has_progress(self._sm.state):
            self._go(AppState.STOPPED)
        return True

    def _on_processing_done(
        self,
        session: EditSession,
        job: ProgressJob,
        settings: DenoiseSettings,
        started: float,
        result: EditResult,
    ) -> None:
        if not self._end_processing(session, job):
            return
        self._commit_edit(result)
        seconds = time.perf_counter() - started
        start, stop = result.command.region
        audio = format_elapsed((stop - start) / result.clip.sample_rate)
        self.show_info(f"{DENOISE_LABEL} ({settings.describe()}): {audio} in {seconds:.1f} s")

    def _on_processing_failed(
        self, session: EditSession, job: ProgressJob, exc: BaseException
    ) -> None:
        if not self._end_processing(session, job):
            return
        if isinstance(exc, Cancelled):
            self.show_info("Noise reduction cancelled; the clip is unchanged")
            return
        log.error("Noise reduction failed", exc_info=exc)
        reason = "not enough memory" if isinstance(exc, MemoryError) else describe_error(exc)
        self.show_error(f"Could not reduce noise: {reason}")

    # ------------------------------------------------------------ export

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def has_unsaved_changes(self) -> bool:
        """Whether the loaded clip hasn't been exported since it was recorded or last changed."""
        session = self._session
        return session is not None and self._saved_position != session.history.position

    @property
    def saved_path(self) -> Path | None:
        """Where the clip was last exported (``None`` if never)."""
        return self._saved_path

    def suggested_export_path(self) -> Path:
        """The last export of this clip; for an opened file, its name next to it (never the
        file itself: ``<name> (edited)``); else ``<last folder>/Recording <date time>``."""
        settings = self._settings.export_settings
        if self._saved_path is not None:
            return with_extension(self._saved_path, settings.format)
        if self._opened is not None:
            source = self._opened.path
            path = with_extension(source, settings.format)
            if path.name.lower() == source.name.lower():
                path = path.with_name(f"{source.stem} (edited){path.suffix}")
            return path
        stamp = self._recorded_at or datetime.now()
        name = f"Recording {stamp:%Y-%m-%d %H-%M-%S}"
        return with_extension(Path(self._settings.export_folder) / name, settings.format)

    def _ask_export(self) -> ExportRequest | None:
        session = self._session
        return ask_export(
            self,
            frames=session.clip.frames,
            rate=session.clip.sample_rate,
            settings=self._settings.export_settings,
            path=self.suggested_export_path(),
            has_overs=session.pyramid.peak > 1.0,
        )

    def export_clip(
        self, request: ExportRequest | None = None, then: Callable[[], None] | None = None
    ) -> None:
        """Export… (Ctrl+S): write the whole clip to a WAV or MP3 file on the loader thread,
        with progress and Cancel; asks for the format and file if ``request`` isn't given.
        ``then`` runs after a successful export (Save in the unsaved-changes prompt)."""
        if self._shut_down or self._session is None or not controls_for(self._sm.state).export:
            return
        if self._sm.state is AppState.PLAYING:
            self.pause_playback()
        if request is None:
            request = self.prompt_export()
            if request is None:  # cancelled: so is whatever ``then`` would have done
                return
        if self._sm.state is not AppState.STOPPED or self._session is None:
            return
        self._settings.export_settings = request.settings
        self._settings.export_folder = request.path.parent
        self._start_export(request, then)

    def _start_export(self, request: ExportRequest, then: Callable[[], None] | None) -> None:
        session = self._session
        clip = session.clip
        position = session.history.position
        job = self._job = ProgressJob()

        def task() -> ExportResult:  # loader thread; nothing edits the clip meanwhile
            return export(
                clip.samples,
                clip.sample_rate,
                request.path,
                request.settings,
                job.report,
                job.is_cancelled,
            )

        self._go(AppState.EXPORTING)
        self.progress_bar.setValue(0)
        self.show_info(f"Exporting {request.path.name} ({request.settings.describe()})…")
        self._progress_timer.start()
        self._loader.submit(
            task,
            lambda result: self._on_export_done(session, job, position, result, then),
            lambda exc: self._on_export_failed(session, job, request, exc),
        )

    def _on_export_done(
        self,
        session: EditSession,
        job: ProgressJob,
        position: int,
        result: ExportResult,
        then: Callable[[], None] | None,
    ) -> None:
        if not self._end_processing(session, job):
            return
        self._saved_position = position
        self._saved_path = result.path
        self._update_clip_info()
        message = (
            f"Exported {result.path.name} ({result.settings.describe()}):"
            f" {format_elapsed(result.duration)}, {format_size(result.bytes)}"
            f" in {result.seconds:.1f} s"
        )
        if result.clipped:
            message += f"; {result.clipped:,} samples above 0 dBFS were clipped"
        self.show_info(message)
        if then is not None:
            then()

    def _on_export_failed(
        self, session: EditSession, job: ProgressJob, request: ExportRequest, exc: BaseException
    ) -> None:
        if not self._end_processing(session, job):
            return
        if isinstance(exc, ExportCancelled):
            self.show_info("Export cancelled; no file was written")
            return
        log.error("Export to %s failed", request.path, exc_info=exc)
        if isinstance(exc, MemoryError):
            reason = "not enough memory"
        elif isinstance(exc, PermissionError):
            reason = "access denied (is the file open in another program?)"
        elif isinstance(exc, OSError) and exc.strerror:
            reason = exc.strerror
        else:
            reason = describe_error(exc)
        self.show_error(f"Could not export {request.path.name}: {reason}")

    # ------------------------------------------------------------ meter / timer

    def _tick(self) -> None:
        now = self._clock()
        dt, self._last_tick = now - self._last_tick, now
        state = self._sm.state
        if state is AppState.PLAYING:
            self._poll_player()
            state = self._sm.state
        if state in (AppState.RECORDING, AppState.PAUSED):
            self._check_recorder_health()
            state = self._sm.state
        if state in (AppState.RECORDING, AppState.PAUSED):
            left, right = self._recorder.levels
            self.transport.set_elapsed(self._recorder.elapsed)
        elif state is AppState.STOPPING:
            left, right = self._recorder.levels
        elif state is AppState.PLAYING:
            left, right = self._playback_levels(dt)
        else:
            left = right = 0.0
        self.meter.set_levels(left, right, dt)
        if not is_capturing(state) and state is not AppState.PLAYING and self.meter.at_rest:
            self._meter_timer.stop()

    def _playback_levels(self, dt: float) -> tuple[float, float]:
        """Peaks of the audio heard since the last tick (the meter follows playback)."""
        clip = self._clip
        if clip is None or not self._player.streaming:
            return 0.0, 0.0
        end = self._player.position_frame
        frames = max(1, round(min(max(dt, METER_INTERVAL_MS / 1000), 0.2) * clip.sample_rate))
        block = clip.samples[max(0, end - frames) : end]
        if not len(block):
            return 0.0, 0.0
        peaks = np.abs(block).max(axis=0)
        return float(peaks[0]), float(peaks[1])

    @property
    def meter_running(self) -> bool:
        return self._meter_timer.isActive()

    # ------------------------------------------------------------ File > Open

    @property
    def opened_file(self) -> OpenedFile | None:
        """Where the clip came from, if it was opened with File > Open."""
        return self._opened

    def open_file(self, path: str | Path | None = None) -> None:
        """File > Open: load an audio file as the clip (``path``, or ask for one). Asks to save
        an unsaved clip first; the current clip stays until the file has loaded."""
        if self._shut_down or not can_open(self._sm.state):
            return
        if path is None:
            path = self.prompt_open_path()
            if path is None or not can_open(self._sm.state):
                return
        path = Path(path)
        if self._sm.state is AppState.PLAYING:
            self.stop_playback()
        if self.has_unsaved_changes:
            choice = self.prompt_unsaved(
                UnsavedPrompt(
                    closing=False,
                    can_save=True,
                    text=f"Save the current clip before opening {path.name}?",
                    detail="The file replaces it, so unsaved audio and edits are lost.",
                )
            )
            if choice is UnsavedChoice.SAVE:
                self.export_clip(then=lambda: self.open_file(path))
                return
            if choice is not UnsavedChoice.DISCARD:
                return
        if not self._sm.can(AppState.LOADING):
            return
        self._open_previous = self._sm.state
        self._opening = path
        self._go(AppState.LOADING)
        self.show_info(f"Opening {path.name}…")
        self._loader.submit(
            lambda: open_audio(path),
            lambda loaded: self._on_opened(path, loaded),
            lambda exc: self._on_open_failed(path, exc),
        )

    def _on_opened(self, path: Path, loaded: tuple[AudioClip, EnvelopePyramid, OpenedFile]) -> None:
        if self._shut_down or path is not self._opening or self.state is not AppState.LOADING:
            return
        self._opening = None
        clip, pyramid, opened = loaded
        previous, self._result = self._result, None
        self._recovered = False
        self._opened = opened
        self._recorded_at = None
        self._set_clip(clip, pyramid)
        if self._session is not None:
            self._saved_position = self._session.history.position  # unchanged: nothing to save
        self.transport.set_elapsed(clip.duration)
        self._go(AppState.STOPPED)
        if previous is not None:
            self._discard(previous)  # the recording it replaced
        try:
            self._settings.open_folder = opened.path.parent
        except Exception:
            log.warning("Could not remember the folder", exc_info=True)
        log.info(
            "Opened %s (%s, %d ch, %d Hz, %.3f s)",
            path,
            opened.format,
            opened.channels,
            opened.sample_rate,
            opened.duration,
        )
        self.show_info(
            f"Opened {path.name}: {format_elapsed(clip.duration)}"
            f" · {opened.sample_rate / 1000:g} kHz {opened.layout}"
        )

    def _on_open_failed(self, path: Path, exc: BaseException) -> None:
        if self._shut_down or path is not self._opening:
            return
        self._opening = None
        log.error("Could not open %s", path, exc_info=exc)
        if self.state is AppState.LOADING:
            self._go(self._open_previous)  # the previous clip (if any) is still there
        self.show_error(f"Could not open {path.name}: {describe_open_error(exc)}")

    @staticmethod
    def _dropped_file(event: QDragEnterEvent | QDropEvent) -> Path | None:
        mime = event.mimeData()
        if not mime.hasUrls() or len(mime.urls()) != 1:
            return None
        url = mime.urls()[0]
        if not url.isLocalFile():
            return None
        path = Path(url.toLocalFile())
        return path if path.suffix.lower() in OPEN_EXTENSIONS else None

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if can_open(self._sm.state) and self._dropped_file(event) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        path = self._dropped_file(event)
        if path is None or not can_open(self._sm.state):
            event.ignore()
            return
        event.acceptProposedAction()
        # After the drop has finished (a modal prompt inside a drop handler confuses Windows).
        QTimer.singleShot(0, lambda: self.open_file(path))

    # ------------------------------------------------------------ crash recovery

    def _temp_folder(self) -> Path:
        return Path(getattr(self._recorder, "temp_dir", None) or default_temp_dir())

    def _spill_store(self) -> SpillStore | None:
        """Where big undo data goes when memory runs short (``None``: disabled)."""
        if self._spill is None and self._max_undo_disk_bytes > 0:
            self._spill = SpillStore(self._temp_folder(), max_bytes=self._max_undo_disk_bytes)
        return self._spill

    def offer_recovery_later(self, delay_ms: int = 300) -> None:
        """Look for crashed recordings once the window is up (called by ``app.main``), and
        delete undo files a crashed instance left behind."""
        try:
            SpillStore.remove_stale(self._temp_folder())
        except Exception:
            log.warning("Could not clean up old undo files", exc_info=True)
        QTimer.singleShot(delay_ms, self.offer_recovery)

    def offer_recovery(self) -> None:
        """Offer the newest recording left behind by a crash (only with no clip loaded)."""
        if self._shut_down or self._sm.state is not AppState.IDLE:
            return
        exclude = (self._result.path,) if self._result is not None else ()
        leftovers = recovery.find_leftovers(self._temp_folder(), exclude)
        empty = [x.path for x in leftovers if x.empty]
        if empty:
            log.info("Deleting %d empty leftover recording(s)", recovery.delete(empty))
        leftovers = [x for x in leftovers if not x.empty]
        if not leftovers:
            return
        newest, count = leftovers[0], len(leftovers)
        log.info("Found %d unsaved recording(s); newest %s", count, newest.path)
        detail = (
            "The app probably closed unexpectedly while recording, or before the clip was"
            " exported. Recover it to play, edit and export it."
        )
        if count > 1:
            detail += (
                f"\n\n{count - 1} older unsaved recording(s) are also there: they are offered"
                " next time the app starts, and Delete all removes them too."
            )
        choice = self.prompt_recovery(
            RecoveryPrompt(
                text=(
                    f"A recording from {newest.modified:%d %b %Y, %H:%M}"
                    f" ({format_elapsed(newest.duration)}) wasn't saved."
                ),
                detail=detail,
                count=count,
            )
        )
        if choice is RecoveryChoice.RECOVER:
            self.recover_recording(newest)
        elif choice is RecoveryChoice.DELETE:
            deleted = recovery.delete([x.path for x in leftovers])
            self.show_info(f"Deleted {deleted} unsaved recording(s)")

    def recover_recording(self, leftover: recovery.Leftover) -> None:
        """Repair and load a crashed recording as the current clip (from IDLE)."""
        if self._shut_down or self._sm.state is not AppState.IDLE:
            return
        rate = leftover.sample_rate
        source = LoopbackDevice(
            index=-1,
            name="",
            sample_rate=rate,
            channels=2,
            output_index=-1,
            output_name="a recovered recording",
            output_channels=2,
        )
        result = RecordingResult(
            path=leftover.path,
            sample_rate=rate,
            channels=2,
            frames=leftover.frames,
            zero_filled_frames=0,
            input_overflows=0,
            wall_clock_seconds=leftover.duration,
            keepalive_active=False,
            device=source,
        )
        self._result = result
        self._recovered = True
        self._opened = None
        self._recorded_at = leftover.modified
        self._go(AppState.LOADING)
        self.transport.set_elapsed(leftover.duration)
        when = f"{leftover.modified:%d %b %Y, %H:%M}"
        self.show_info(f"Recovering the recording from {when}…")
        path = leftover.path

        def task() -> tuple[AudioClip, EnvelopePyramid]:  # loader thread
            recovery.repair(path)
            return load_clip(path)

        def loaded(clip_and_pyramid: tuple[AudioClip, EnvelopePyramid]) -> None:
            if result is not self._result or self._sm.state is not AppState.LOADING:
                return
            self._on_loaded(result, clip_and_pyramid)
            self.show_info(
                f"Recovered the recording from {when}"
                f" ({format_elapsed(clip_and_pyramid[0].duration)}). Export it to keep it."
            )

        def failed(exc: BaseException) -> None:
            if self._shut_down or result is not self._result:
                return
            log.error("Could not recover %s", path, exc_info=exc)
            self._result = None  # keep the file: it isn't deleted on close
            self._recovered = False
            if self._sm.state is AppState.LOADING:
                self._go(AppState.IDLE)
            self.show_error(
                f"Could not recover {path.name}: {describe_error(exc)}."
                f" The file is still in {path.parent}."
            )

        self._loader.submit(task, loaded, failed)

    # ------------------------------------------------------------ session end

    def commit_session_data(self, manager: Any) -> None:
        """Windows is signing out, restarting or shutting down (``commitDataRequest``)."""
        if self._shut_down:
            return
        busy = self._busy_text()
        if busy is None and not self.has_unsaved_changes:
            self._session_ending = True
            return
        if not manager.allowsInteraction():
            log.warning("Session ending without interaction: keeping the recording for recovery")
            self._keep_for_recovery = True
            self._session_ending = True
            return
        if busy is not None:
            prompt = UnsavedPrompt(
                closing=True,
                can_save=False,
                text=busy,
                detail="Windows is signing out. Close anyway? Unsaved audio will be lost.",
            )
        else:
            prompt = UnsavedPrompt(
                closing=True,
                can_save=True,
                text="Save the clip before Windows signs out?",
                detail="Save… cancels the sign-out so the export can finish.",
            )
        try:
            choice = self.prompt_unsaved(prompt)
        finally:
            manager.release()
        if choice is UnsavedChoice.DISCARD:
            self._session_ending = True
            return
        manager.cancel()
        log.info("Sign-out cancelled (%s)", choice.value)
        if choice is UnsavedChoice.SAVE:
            self.export_clip()

    # ------------------------------------------------------------ cleanup

    @staticmethod
    def _discard(result: RecordingResult) -> None:
        """Delete a recording's temp file (the clip was saved, or the user chose Discard)."""
        try:
            Path(result.path).unlink(missing_ok=True)
            log.info("Discarded unsaved recording %s", result.path)
        except OSError:
            log.warning("Could not delete temp recording %s", result.path, exc_info=True)

    def shutdown(self) -> None:
        """Finish background work, stop any recording and delete unsaved temp files.

        Idempotent: runs from closeEvent and again from ``QApplication.aboutToQuit``.
        """
        self._shut_down = True  # closing fires ActivationChange; don't queue a refresh
        self._save_settings()
        if self._job is not None:
            self._job.cancel()  # don't wait minutes for a noise reduction nobody will see
        self._progress_timer.stop()
        self._watch.close()
        self._player.close()
        self._runner.drain()
        self._loader.drain()
        self._meter_timer.stop()
        if self._session is not None:
            self._session.close()
        if self._spill is not None:
            self._spill.close()
        if self._recorder.state is not RecorderState.IDLE:
            try:
                result = self._runner.run_blocking(self._recorder.stop)
                if self._keep_for_recovery:
                    log.info("Kept the recording %s for recovery", result.path)
                else:
                    self._discard(result)
            except Exception:
                log.exception("Could not stop the recorder on exit")
        if self._result is not None:
            if self._keep_for_recovery:
                log.info("Kept the recording %s for recovery", self._result.path)
            else:
                self._discard(self._result)
            self._result = None

    def _save_settings(self) -> None:
        """Remember the window geometry (once, while the window still has it)."""
        if self._settings_saved:
            return
        self._settings_saved = True
        try:
            self._settings.window_geometry = self.saveGeometry()
            self._settings.sync()
        except Exception:
            log.warning("Could not save the settings", exc_info=True)

    def _busy_text(self) -> str | None:
        """What would be lost by closing now, while audio is being recorded or processed."""
        state = self._sm.state
        if is_capturing(state) or state is AppState.LOADING:
            return "A recording is in progress."
        if state is AppState.EXPORTING:
            return "The clip is being exported."
        if state in (AppState.EDITING, AppState.PROCESSING):
            return "The clip is being edited."
        return None

    def confirm_close(self) -> bool:
        """Ask before closing with unsaved audio; ``False`` keeps the window open (Save starts
        an export and closes the window when it has finished)."""
        busy = self._busy_text()
        if busy is not None:
            choice = self.prompt_unsaved(
                UnsavedPrompt(
                    closing=True,
                    can_save=False,
                    text=busy,
                    detail="Close anyway? Unsaved audio will be lost.",
                )
            )
            return choice is UnsavedChoice.DISCARD
        if not self.has_unsaved_changes:
            return True
        choice = self.prompt_unsaved(
            UnsavedPrompt(
                closing=True,
                can_save=True,
                text="Save the clip before closing?",
                detail="Otherwise the recording and your edits are lost.",
            )
        )
        if choice is UnsavedChoice.SAVE:
            self.export_clip(then=self.close)
            return False
        return choice is UnsavedChoice.DISCARD

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._shut_down and not self._session_ending and not self.confirm_close():
            event.ignore()
            return
        self.shutdown()
        super().closeEvent(event)
