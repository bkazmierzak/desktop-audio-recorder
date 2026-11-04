"""The main window's state machine (no widgets here, so it is easy to test).

::

    IDLE ──record──► STARTING ──ok──► RECORDING ◄──resume/pause──► PAUSED
      ▲                 │ fail            │ stop                    │ stop
      │                 ▼                 ▼                         ▼
      ├──(fail)──── previous ◄──fail── STOPPING ◄────────────────────┘
      │                                   │ ok
      │                                   ▼
      └───────────────fail─────────── LOADING  (WAV → AudioClip off the GUI thread)
                                          │ ok
                                          ▼
                     STARTING ◄──record── STOPPED  (has a clip) ◄──pause/stop/end──► PLAYING
                                          ▲ │
                                     done │ │ edit / undo / redo (long clip)
                                          │ ▼
                                        EDITING

    STOPPED ──noise reduction──► PROCESSING ──done / cancelled / failed──► STOPPED
    STOPPED ──export─────────► EXPORTING  ──done / cancelled / failed──► STOPPED
    IDLE ──recover a crashed recording──► LOADING ──ok──► STOPPED  (or ──fail──► IDLE)
    IDLE / STOPPED ──open a file──► LOADING ──ok──► STOPPED  (or ──fail──► where it came from)

``STARTING``, ``STOPPING``, ``LOADING``, ``EDITING``, ``PROCESSING`` and ``EXPORTING`` are
transient: the work runs on a background thread and the controls are disabled until it finishes.
``PROCESSING`` (noise reduction: seconds per minute of audio) and ``EXPORTING`` (writing a WAV or
MP3: up to a minute per hour of audio) show progress and are the only states with a Cancel
button; ``EDITING`` (at most a second or so per hour of audio) isn't cancellable.
Recording is disabled while ``PLAYING``; pausing or stopping playback returns to ``STOPPED``.

Editing works in ``STOPPED`` and ``PLAYING``; an edit made while playing pauses playback first
(the window goes to ``STOPPED``, keeping the cursor). Small edits finish immediately and never
enter ``EDITING``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class AppState(enum.Enum):
    IDLE = "idle"  # nothing recorded yet
    STARTING = "starting"  # recorder.start() running on the audio thread
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPING = "stopping"  # recorder.stop() running on the audio thread
    LOADING = "loading"  # the finished recording is being loaded into memory
    STOPPED = "stopped"  # a clip is loaded and not playing
    PLAYING = "playing"  # the clip is playing
    EDITING = "editing"  # an edit (or undo/redo) on a long clip runs on the loader thread
    PROCESSING = "processing"  # noise reduction runs on the loader thread (progress, Cancel)
    EXPORTING = "exporting"  # the clip is being written to a file (progress, Cancel)


TRANSITIONS: dict[AppState, frozenset[AppState]] = {
    # IDLE -> LOADING: recovering a recording left behind by a crash, or opening a file.
    AppState.IDLE: frozenset({AppState.STARTING, AppState.LOADING}),
    AppState.STOPPED: frozenset(
        {
            AppState.STARTING,
            AppState.LOADING,  # File > Open (a failed open comes back here, clip unchanged)
            AppState.PLAYING,
            AppState.EDITING,
            AppState.PROCESSING,
            AppState.EXPORTING,
        }
    ),
    # A failed start returns to whichever of IDLE/STOPPED it came from.
    AppState.STARTING: frozenset({AppState.RECORDING, AppState.IDLE, AppState.STOPPED}),
    AppState.RECORDING: frozenset({AppState.PAUSED, AppState.STOPPING}),
    AppState.PAUSED: frozenset({AppState.RECORDING, AppState.STOPPING}),
    # A failed stop leaves no usable clip, so it goes back to IDLE (the previous clip was
    # discarded when the recording started).
    AppState.STOPPING: frozenset({AppState.LOADING, AppState.IDLE}),
    AppState.LOADING: frozenset({AppState.STOPPED, AppState.IDLE}),
    AppState.PLAYING: frozenset({AppState.STOPPED}),
    AppState.EDITING: frozenset({AppState.STOPPED}),  # done or failed: the clip is still there
    AppState.PROCESSING: frozenset({AppState.STOPPED}),  # done, cancelled or failed
    AppState.EXPORTING: frozenset({AppState.STOPPED}),  # done, cancelled or failed
}


class InvalidTransition(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Controls:
    """Which controls are usable in a state."""

    record: bool
    pause: bool  # the Pause/Resume (recording) button
    stop: bool  # Stop recording
    refresh: bool
    pause_label: str  # "Pause" or "Resume"
    play: bool  # the Play/Pause (playback) button and the Space shortcut
    play_label: str  # "Play" or "Pause"
    stop_playback: bool
    clip: bool  # the waveform (seek, select), Loop and Home are usable
    edit: bool  # the edit toolbar and its shortcuts (see ``edit_controls``)
    cancel: bool  # the Cancel button next to the progress bar
    export: bool  # Export… (Ctrl+S)


def controls_for(state: AppState) -> Controls:
    has_clip = state in (AppState.STOPPED, AppState.PLAYING)
    return Controls(
        record=state in (AppState.IDLE, AppState.STOPPED),
        pause=state in (AppState.RECORDING, AppState.PAUSED),
        stop=state in (AppState.RECORDING, AppState.PAUSED),
        refresh=state in (AppState.IDLE, AppState.STOPPED, AppState.PLAYING),
        pause_label="Resume" if state is AppState.PAUSED else "Pause",
        play=has_clip,
        play_label="Pause" if state is AppState.PLAYING else "Play",
        stop_playback=state is AppState.PLAYING,
        clip=has_clip,
        edit=has_clip,
        cancel=has_progress(state),
        export=has_clip,
    )


def can_open(state: AppState) -> bool:
    """Whether File > Open is usable (playback is stopped first)."""
    return state in (AppState.IDLE, AppState.STOPPED, AppState.PLAYING)


def has_progress(state: AppState) -> bool:
    """Whether the state shows a progress bar and a Cancel button."""
    return state in (AppState.PROCESSING, AppState.EXPORTING)


@dataclass(frozen=True, slots=True)
class EditControls:
    """Which edit actions are usable."""

    delete: bool  # needs a selection
    crop: bool
    silence: bool
    fade_in: bool
    fade_out: bool
    gain: bool  # the selection, or the whole clip without one
    normalize: bool
    insert_silence: bool  # at the selection start or the cursor
    undo: bool
    redo: bool
    denoise: bool  # noise reduction on the selection, or the whole clip without one


def edit_controls(
    state: AppState, *, has_selection: bool, can_undo: bool, can_redo: bool
) -> EditControls:
    edit = controls_for(state).edit
    selected = edit and has_selection
    return EditControls(
        delete=selected,
        crop=selected,
        silence=selected,
        fade_in=selected,
        fade_out=selected,
        gain=edit,
        normalize=edit,
        insert_silence=edit,
        undo=edit and can_undo,
        redo=edit and can_redo,
        denoise=edit,
    )


def is_capturing(state: AppState) -> bool:
    """Whether the recorder has (or is about to have) an open session."""
    return state in (AppState.STARTING, AppState.RECORDING, AppState.PAUSED, AppState.STOPPING)


class StateMachine:
    def __init__(self, initial: AppState = AppState.IDLE) -> None:
        self._state = initial

    @property
    def state(self) -> AppState:
        return self._state

    def can(self, target: AppState) -> bool:
        return target in TRANSITIONS.get(self._state, frozenset())

    def go(self, target: AppState) -> AppState:
        """Move to ``target``; return the previous state. Raises ``InvalidTransition``."""
        if not self.can(target):
            raise InvalidTransition(f"{self._state.value} -> {target.value}")
        previous, self._state = self._state, target
        return previous
