
"""
Model registry: automatically discovers any model in src/models/*/model.py
Each model.py must define:
    NAME  -> string identifier
    build -> function returning a torch.nn.Module
"""

import importlib
import pkgutil
from pathlib import Path

def discover_models(pkg: str = "src.models"):
    """
    Returns a dict {model_name: build_function}
    Scans all subfolders under src/models/ that contain model.py
    """
    models = {}
    pkg_path = Path(__file__).parent / "models"

    for mod in pkgutil.iter_modules([str(pkg_path)]):
        # We only care about subpackages (each model in its own folder)
        if not mod.ispkg:
            continue
        try:
            m = importlib.import_module(f"{pkg}.{mod.name}.model")
        except ModuleNotFoundError:
            continue

        name = getattr(m, "NAME", None)
        build = getattr(m, "build", None)

        if isinstance(name, str) and callable(build):
            models[name] = build

    return models
