import os
import sys


def ensure_zeus_on_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    zeus_path = os.path.join(repo_root, "zeus-master")
    if zeus_path not in sys.path:
        sys.path.insert(0, zeus_path)
    return zeus_path
