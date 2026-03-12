"""Video thumbnail demo — high-performance timeline strip generator.

Modules
-------
config      – UI constants, stylesheet, tuning knobs.
probe       – Video probing via ffprobe / PyAV.
hwaccel     – Hardware-acceleration detection (cached per process).
extraction  – Frame and contact-sheet extraction helpers.
worker      – ``ThumbnailWorker`` QThread orchestrator.
ui          – Qt widgets (ThumbnailBar, VideoEditor, …).
"""
