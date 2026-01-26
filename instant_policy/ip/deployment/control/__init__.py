from ip.deployment.control.action_executor import ActionExecutor, SafetyLimits

__all__ = ["ActionExecutor", "SafetyLimits", "URRTDEControl"]


def __getattr__(name: str):
    if name == "URRTDEControl":
        # Lazy import to avoid circular imports during package initialization.
        from ip.deployment.control.ur_rtde_control import URRTDEControl

        return URRTDEControl
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
