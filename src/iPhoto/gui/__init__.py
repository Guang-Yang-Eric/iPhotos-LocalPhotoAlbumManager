"""GUI package for the iPhoto application."""


def __getattr__(name):
    if name == "AppFacade":
        from .facade import AppFacade
        return AppFacade
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AppFacade"]
