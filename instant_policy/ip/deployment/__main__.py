import importlib.util
from pathlib import Path


def _load_entrypoint():
    entrypoint_path = Path(__file__).resolve().parents[1] / "deployment.py"
    spec = importlib.util.spec_from_file_location("ip._deployment_entrypoint", entrypoint_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load deployment entrypoint from {entrypoint_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = _load_entrypoint()
    if not hasattr(module, "main"):
        raise RuntimeError("Deployment entrypoint does not define main()")
    module.main()


if __name__ == "__main__":
    main()
