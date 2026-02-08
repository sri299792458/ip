from .config import GenerationConfig, CameraConfig

__all__ = ["GenerationConfig", "CameraConfig", "PseudoDemoGenerator"]


def __getattr__(name):
    if name == "PseudoDemoGenerator":
        from .pseudo_demo_generator import PseudoDemoGenerator
        return PseudoDemoGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
