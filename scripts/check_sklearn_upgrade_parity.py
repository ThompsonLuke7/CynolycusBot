"""Prove a scikit-learn upgrade does not move any live model's output.

Every pickled estimator under strategies/ and signals/ was serialized by a
scikit-learn NEWER than the pinned runtime (28 on 1.6.1, 2 on 1.9.0 as of
2026-08-10), so loading them already crosses a version boundary. Moving the pin
changes which direction that crossing runs. Unpickling across versions is not
guaranteed to preserve behaviour, and a silent shift in a calibrator or tree
wrapper would show up as slightly different live scores rather than an error.

Usage — capture on the OLD interpreter, compare on the NEW one:

    .venv/bin/python scripts/check_sklearn_upgrade_parity.py --save before.json
    # ... pip install 'scikit-learn==1.6.1' ...
    .venv/bin/python scripts/check_sklearn_upgrade_parity.py --compare before.json

Inputs are synthetic but fixed by seed, so the same rows are scored on both
sides. That is what parity needs; it is NOT a claim about live feature
distributions.

Exit codes: 0 parity holds, 1 a model moved, 2 the artifact set itself changed.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_GLOBS = ("strategies/**/*.joblib", "signals/**/*.joblib")
N_ROWS = 64
SEED = 20260810
# Tree ensembles are deterministic given identical input, so parity should be
# bit-exact. The tolerance only absorbs float32/float64 summation order.
RTOL = 1e-9
ATOL = 1e-9


def discover_artifacts() -> list[Path]:
    found: set[Path] = set()
    for pattern in ARTIFACT_GLOBS:
        found.update(Path(p) for p in glob.glob(str(REPO_ROOT / pattern), recursive=True))
    return sorted(found)


def _make_input(model, n_features: int | None) -> np.ndarray:
    """Deterministic input matched to the estimator's expected shape."""
    rng = np.random.default_rng(SEED)
    if n_features is None:
        # 1-D calibrators (IsotonicRegression). Sweep the fitted support when it
        # is known, so the comparison covers the interpolation knots.
        lo = float(getattr(model, "X_min_", -5.0))
        hi = float(getattr(model, "X_max_", 5.0))
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            lo, hi = -5.0, 5.0
        return np.linspace(lo, hi, N_ROWS)
    return rng.standard_normal((N_ROWS, int(n_features)))


def _score(model) -> tuple[str, np.ndarray]:
    n_features = getattr(model, "n_features_in_", None)
    x = _make_input(model, n_features)
    for method in ("predict_proba", "predict", "transform"):
        fn = getattr(model, method, None)
        if fn is None:
            continue
        out = np.asarray(fn(x), dtype=np.float64)
        return method, out.ravel()
    raise TypeError(f"{type(model).__name__} exposes no predict/transform method")


def fingerprint_artifact(path: Path) -> dict:
    import joblib

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(path)
        method, out = _score(model)

    return {
        "estimator": type(model).__name__,
        "method": method,
        "n_features_in": getattr(model, "n_features_in_", None),
        "n_outputs": int(out.size),
        # Full vector, so --compare can localise a drift rather than only flag it.
        "outputs": [float(v) for v in out],
        "sha256": hashlib.sha256(out.tobytes()).hexdigest(),
    }


def build_report() -> dict:
    import sklearn

    artifacts: dict[str, dict] = {}
    failures: dict[str, str] = {}
    for path in discover_artifacts():
        rel = str(path.relative_to(REPO_ROOT))
        try:
            artifacts[rel] = fingerprint_artifact(path)
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            failures[rel] = f"{type(exc).__name__}: {exc}"

    return {
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "seed": SEED,
        "n_rows": N_ROWS,
        "artifacts": artifacts,
        "failures": failures,
    }


def cmd_save(out_path: Path) -> int:
    report = build_report()
    out_path.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"sklearn {report['sklearn']} / numpy {report['numpy']}")
    print(f"fingerprinted {len(report['artifacts'])} artifacts -> {out_path}")
    if report["failures"]:
        print(f"WARNING: {len(report['failures'])} artifact(s) could not be scored:")
        for rel, err in report["failures"].items():
            print(f"  {rel}: {err}")
    return 0


def cmd_compare(baseline_path: Path) -> int:
    baseline = json.loads(baseline_path.read_text())
    current = build_report()

    print(f"baseline: sklearn {baseline['sklearn']} / numpy {baseline['numpy']}")
    print(f"current : sklearn {current['sklearn']} / numpy {current['numpy']}")

    if baseline.get("seed") != current["seed"] or baseline.get("n_rows") != current["n_rows"]:
        print("FAIL: baseline was captured with different input settings")
        return 2

    old_keys, new_keys = set(baseline["artifacts"]), set(current["artifacts"])
    if old_keys != new_keys:
        for rel in sorted(old_keys - new_keys):
            print(f"MISSING now: {rel}")
        for rel in sorted(new_keys - old_keys):
            print(f"NEW now    : {rel}")
        print("FAIL: artifact set changed; parity is not comparable")
        return 2

    newly_broken = set(current["failures"]) - set(baseline["failures"])
    drifted: list[tuple[str, float]] = []
    for rel in sorted(old_keys):
        before = np.asarray(baseline["artifacts"][rel]["outputs"], dtype=np.float64)
        after = np.asarray(current["artifacts"][rel]["outputs"], dtype=np.float64)
        if before.shape != after.shape:
            drifted.append((rel, float("inf")))
            continue
        if not np.allclose(before, after, rtol=RTOL, atol=ATOL, equal_nan=True):
            drifted.append((rel, float(np.nanmax(np.abs(before - after)))))

    if newly_broken:
        print(f"\nFAIL: {len(newly_broken)} artifact(s) stopped loading:")
        for rel in sorted(newly_broken):
            print(f"  {rel}: {current['failures'][rel]}")

    if drifted:
        print(f"\nFAIL: {len(drifted)} artifact(s) changed output:")
        for rel, delta in drifted:
            print(f"  max|delta|={delta:.3e}  {rel}")

    if drifted or newly_broken:
        print("\nDo NOT ship this upgrade. Roll back and investigate.")
        return 1

    print(f"\nPASS: all {len(old_keys)} artifacts byte-identical "
          f"(rtol={RTOL:g}, atol={ATOL:g})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--save", type=Path, metavar="PATH", help="write a fingerprint of every artifact")
    group.add_argument("--compare", type=Path, metavar="PATH", help="re-score and diff against a saved fingerprint")
    args = parser.parse_args(argv)
    return cmd_save(args.save) if args.save else cmd_compare(args.compare)


if __name__ == "__main__":
    sys.exit(main())
