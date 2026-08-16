"""umap-learn and hdbscan must keep working under the pinned scikit-learn <1.6.

Regression cover for the 2026-08-03 weekly theme rebuild, which died in
step03 with ``TypeError: check_array() got an unexpected keyword argument
'ensure_all_finite'``. UMAP raised it first; hdbscan raises the same error on
the next statement, so both loaders are covered here.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from themes.dynamic_theme.sklearn_compat import (
    _PATCH_FLAG,
    _needs_shim,
    load_hdbscan,
    load_umap,
)


def test_shim_is_needed_only_below_sklearn_1_6():
    from sklearn.utils import check_array

    accepts_new_name = "ensure_all_finite" in inspect.signature(check_array).parameters
    assert _needs_shim() is not accepts_new_name


def test_umap_fit_transform_survives_the_renamed_kwarg():
    umap = load_umap()
    matrix = np.random.default_rng(0).normal(size=(120, 32)).astype(np.float32)

    reduced = umap.UMAP(
        n_components=5,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=False,
    ).fit_transform(matrix)

    assert reduced.shape == (120, 5)
    assert np.isfinite(reduced).all()


def test_hdbscan_fit_survives_the_renamed_kwarg():
    """The failure one line after UMAP's, which 2026-08-03 never reached."""
    hdbscan = load_hdbscan()
    rng = np.random.default_rng(0)
    blobs = np.vstack([rng.normal(loc, 0.2, size=(40, 5)) for loc in (-4.0, 0.0, 4.0)])

    model = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=2,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    model.fit(blobs.astype(np.float32))

    assert model.labels_.shape == (120,)
    assert int(model.labels_.max()) + 1 >= 2


@pytest.mark.parametrize(
    "loader, module_name",
    [(load_umap, "umap.umap_"), (load_hdbscan, "hdbscan.hdbscan_")],
)
def test_repeated_loads_do_not_stack_wrappers(loader, module_name):
    import importlib

    loader()
    module = importlib.import_module(module_name)

    if not _needs_shim():
        pytest.skip("scikit-learn >= 1.6 needs no shim")

    first = module.check_array
    loader()
    assert module.check_array is first
    assert getattr(first, _PATCH_FLAG, False) is True


def test_translated_kwarg_keeps_its_meaning():
    """It must still gate on non-finite values, not just swallow the kwarg."""
    load_umap()
    import umap.umap_ as umap_module

    with_nan = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)

    with pytest.raises(ValueError):
        umap_module.check_array(with_nan, ensure_all_finite=True)

    passed = umap_module.check_array(with_nan, ensure_all_finite=False)
    assert np.isnan(passed).any()
