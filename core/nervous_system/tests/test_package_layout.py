from importlib import import_module
from pathlib import Path


def test_nervous_system_package_and_required_subpackages_exist():
    root = Path("core/nervous_system")
    names = (
        "contracts", "config", "persistence", "data_registry", "context",
        "policy", "portfolio", "execution", "orchestration", "replay",
    )
    assert root.is_dir()
    for name in names:
        import_module(f"core.nervous_system.{name}")
