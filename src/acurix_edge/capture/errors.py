"""Capture errors (no OpenCV import: the composition root can catch them without loading cv2)."""

from __future__ import annotations

from acurix_edge.contracts.errors import ErrorCode


class CaptureError(RuntimeError):
    code = ErrorCode.CAP_001


class CaptureUnavailable(CaptureError):
    """Source could not be opened or did not deliver a valid frame in time."""


class CaptureLost(CaptureError):
    """The session's source was lost; the session can never receive frames again."""


class StaleSession(CaptureError):
    """The session id is not the one bound to the capture (ended, lost or never begun)."""


class EndOfStream(Exception):
    """Finite source exhausted and all its frames delivered (explicit end, not an error)."""
