"""Edit toolbar: Delete, Crop, Silence, Insert silence, Fade in/out, Gain, Normalize, Noise
reduction, Undo/Redo.

The toolbar only owns the ``QAction``s (with their shortcuts) and their enabled state; the main
window connects ``action.triggered`` and runs the edits. Shortcuts are window-wide
(``Qt.WindowShortcut``, the ``QAction`` default) and, like the buttons, do nothing while an
action is disabled: Ctrl+Z undo, Ctrl+Y or Ctrl+Shift+Z redo, Del delete the selection.

Labels are plain text: glyphs outside Segoe UI (e.g. arrows for undo/redo) stall the first paint
while Qt searches the system fonts.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QInputDialog, QSizePolicy, QToolBar, QWidget

from desktop_audio_recorder.ui.state import EditControls

GAIN_RANGE_DB = (-48.0, 48.0)
INSERT_RANGE_S = (0.01, 600.0)


def _set_tip(action: QAction, tip: str) -> None:
    """Tooltip with the (first) shortcut, e.g. ``"Undo: Delete (Ctrl+Z)"``."""
    keys = action.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
    action.setToolTip(f"{tip} ({keys})" if keys else tip)


class EditToolbar(QToolBar):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Edit", parent)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.setMovable(False)
        self.setFloatable(False)

        self.delete_action = self._action(
            "Delete", "Remove the selection; the audio after it moves up", ["Del"]
        )
        self.crop_action = self._action("Crop", "Keep only the selection")
        self.silence_action = self._action("Silence", "Replace the selection with silence")
        self.insert_silence_action = self._action(
            "Insert silence…", "Insert silence at the start of the selection (or at the cursor)"
        )
        self.addSeparator()
        self.fade_in_action = self._action("Fade in", "Fade the selection in from silence")
        self.fade_out_action = self._action("Fade out", "Fade the selection out to silence")
        self.addSeparator()
        self.gain_action = self._action(
            "Gain…", "Change the volume of the selection (or the whole clip) in dB"
        )
        self.normalize_action = self._action(
            "Normalize", "Scale the selection (or the whole clip) so its peak is at -1 dBFS"
        )
        self.addSeparator()
        self.denoise_action = self._action(
            "Noise reduction…",
            "Reduce steady background noise (hiss, hum, fan) in the selection or the whole clip",
        )

        spacer = QWidget(self)
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.addWidget(spacer)

        self.undo_action = self._action("Undo", "Undo the last edit", ["Ctrl+Z"])
        self.redo_action = self._action(
            "Redo", "Redo the last undone edit", ["Ctrl+Y", "Ctrl+Shift+Z"]
        )
        self.set_history(None, None)
        self.set_controls(None)

    def _action(self, text: str, tip: str, shortcuts: list[str] | None = None) -> QAction:
        action = QAction(text, self)
        if shortcuts:
            action.setShortcuts([QKeySequence(s) for s in shortcuts])
            action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        _set_tip(action, tip)
        self.addAction(action)
        return action

    @property
    def edit_actions(self) -> dict[str, QAction]:
        """Actions by their ``EditControls`` field name."""
        return {
            "delete": self.delete_action,
            "crop": self.crop_action,
            "silence": self.silence_action,
            "fade_in": self.fade_in_action,
            "fade_out": self.fade_out_action,
            "gain": self.gain_action,
            "normalize": self.normalize_action,
            "insert_silence": self.insert_silence_action,
            "undo": self.undo_action,
            "redo": self.redo_action,
            "denoise": self.denoise_action,
        }

    def set_controls(self, controls: EditControls | None) -> None:
        """Enable the actions (``None`` disables all of them)."""
        for name, action in self.edit_actions.items():
            action.setEnabled(bool(controls is not None and getattr(controls, name)))

    def set_history(self, undo_label: str | None, redo_label: str | None) -> None:
        """Name the edit that Undo/Redo would act on in their tooltips."""
        _set_tip(self.undo_action, f"Undo: {undo_label}" if undo_label else "Nothing to undo")
        _set_tip(self.redo_action, f"Redo: {redo_label}" if redo_label else "Nothing to redo")


# --------------------------------------------------------------------------- dialogs


def ask_gain_db(parent: QWidget | None, initial: float = 3.0) -> float | None:
    """Modal dB prompt; ``None`` if cancelled."""
    dialog = QInputDialog(parent)
    dialog.setWindowTitle("Gain")
    dialog.setLabelText("Gain in dB (negative makes it quieter):")
    dialog.setInputMode(QInputDialog.InputMode.DoubleInput)
    dialog.setDoubleRange(*GAIN_RANGE_DB)
    dialog.setDoubleDecimals(1)
    dialog.setDoubleStep(0.5)
    dialog.setDoubleValue(initial)
    if dialog.exec() != QInputDialog.DialogCode.Accepted:
        return None
    return dialog.doubleValue()


def ask_silence_seconds(parent: QWidget | None, initial: float = 1.0) -> float | None:
    """Modal seconds prompt for Insert silence; ``None`` if cancelled."""
    dialog = QInputDialog(parent)
    dialog.setWindowTitle("Insert silence")
    dialog.setLabelText("Length of silence to insert (seconds):")
    dialog.setInputMode(QInputDialog.InputMode.DoubleInput)
    dialog.setDoubleRange(*INSERT_RANGE_S)
    dialog.setDoubleDecimals(2)
    dialog.setDoubleStep(0.5)
    dialog.setDoubleValue(initial)
    if dialog.exec() != QInputDialog.DialogCode.Accepted:
        return None
    return dialog.doubleValue()
