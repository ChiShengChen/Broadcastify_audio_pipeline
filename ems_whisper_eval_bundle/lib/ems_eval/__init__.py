"""EMS evaluation package for speech-to-text and medical terminology assessment."""

from .preprocessing import normalize_ems_text
from .metrics import STTMetrics

__all__ = ["normalize_ems_text", "STTMetrics"]
