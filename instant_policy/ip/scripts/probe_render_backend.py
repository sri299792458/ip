"""Probe pyrender/OpenGL backend availability.

Usage examples:
  python -m ip.scripts.probe_render_backend --platform egl --strict
  python -m ip.scripts.probe_render_backend --platform osmesa --strict
"""

from __future__ import annotations

import argparse
import os
import sys


EXPECTED_PLATFORM = {
    "egl": "EGLPlatform",
    "osmesa": "OSMesaPlatform",
    "pyglet": "PygletPlatform",
    "glx": "PygletPlatform",
}


def _dec(value) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe pyrender backend.")
    parser.add_argument(
        "--platform",
        type=str,
        default="auto",
        choices=["auto", "egl", "osmesa", "pyglet", "glx"],
        help="Requested PYOPENGL_PLATFORM (auto keeps current env).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if actual backend does not match requested platform.",
    )
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    args = parser.parse_args()

    if args.platform != "auto":
        os.environ["PYOPENGL_PLATFORM"] = args.platform

    requested = args.platform
    env_platform = os.environ.get("PYOPENGL_PLATFORM", "")
    print(f"[render_probe] requested_platform={requested}")
    print(f"[render_probe] PYOPENGL_PLATFORM={env_platform}")

    try:
        import pyrender
        from OpenGL import GL

        renderer = pyrender.OffscreenRenderer(args.width, args.height)
        platform = type(getattr(renderer, "_platform", None)).__name__
        gl_vendor = _dec(GL.glGetString(GL.GL_VENDOR))
        gl_renderer = _dec(GL.glGetString(GL.GL_RENDERER))
        gl_version = _dec(GL.glGetString(GL.GL_VERSION))
        renderer.delete()

        print(f"[render_probe] platform={platform}")
        print(f"[render_probe] vendor={gl_vendor}")
        print(f"[render_probe] renderer={gl_renderer}")
        print(f"[render_probe] version={gl_version}")

        if args.strict and requested != "auto":
            expected = EXPECTED_PLATFORM.get(requested)
            if expected and platform != expected:
                print(f"[render_probe] status=mismatch expected={expected} actual={platform}")
                return 2

        print("[render_probe] status=ok")
        return 0
    except Exception as exc:
        print(f"[render_probe] status=error")
        print(f"[render_probe] error={exc!r}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

