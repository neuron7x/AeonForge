import importlib, pkgutil, sys
import pathlib

def import_all(package_name: str):
    pkg = importlib.import_module(package_name)
    pkg_path = pathlib.Path(pkg.__file__).parent
    for m in pkgutil.walk_packages([str(pkg_path)], prefix=package_name + "."):
        importlib.import_module(m.name)

def test_import_all_modules():
    # Imports all submodules to catch import-time errors
    import_all("agi_core")
