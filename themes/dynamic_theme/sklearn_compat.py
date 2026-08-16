"""Import UMAP / HDBSCAN under the scikit-learn version this repo pins.

scikit-learn renamed ``check_array(force_all_finite=...)`` to
``ensure_all_finite=`` in 1.6. Current umap-learn and hdbscan pass the new name
unconditionally, but ``requirements.txt`` pins ``scikit-learn>=1.5,<1.6`` — so
both libraries die with ``TypeError: check_array() got an unexpected keyword
argument 'ensure_all_finite'``. UMAP fails first (it broke the 2026-08-03 weekly
theme rebuild); HDBSCAN fails on the very next line of step03.

Translate the kwarg on the reference each library holds, rather than moving
scikit-learn off its pin. The shim removes itself once the pin is lifted: under
scikit-learn >= 1.6 ``check_array`` already accepts the new name and the loaders
return their modules untouched.
"""
from __future__ import annotations

import functools
import inspect
from types import ModuleType

_PATCH_FLAG = "_cynolycus_ensure_all_finite_shim"


def _needs_shim() -> bool:
    from sklearn.utils import check_array

    return "ensure_all_finite" not in inspect.signature(check_array).parameters


def _patch(module: ModuleType) -> None:
    """Point ``module.check_array`` at a kwarg-translating wrapper, once."""
    original = getattr(module, "check_array", None)
    if original is None or getattr(original, _PATCH_FLAG, False):
        return

    @functools.wraps(original)
    def check_array_compat(*args, **kwargs):
        if "ensure_all_finite" in kwargs:
            kwargs["force_all_finite"] = kwargs.pop("ensure_all_finite")
        return original(*args, **kwargs)

    setattr(check_array_compat, _PATCH_FLAG, True)
    module.check_array = check_array_compat


def load_umap() -> ModuleType:
    """Return the ``umap`` module, kwarg-compatible with the pinned sklearn."""
    try:
        import umap
        import umap.umap_ as umap_module
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError("Install umap-learn: pip install umap-learn") from exc

    if _needs_shim():
        _patch(umap_module)
    return umap


def load_hdbscan() -> ModuleType:
    """Return the ``hdbscan`` module, kwarg-compatible with the pinned sklearn."""
    try:
        import hdbscan
        import hdbscan.hdbscan_ as hdbscan_module
        import hdbscan.robust_single_linkage_ as rsl_module
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError("Install hdbscan: pip install hdbscan") from exc

    if _needs_shim():
        _patch(hdbscan_module)
        _patch(rsl_module)
    return hdbscan
