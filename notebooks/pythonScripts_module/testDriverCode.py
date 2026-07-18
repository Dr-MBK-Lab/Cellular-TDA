#!/usr/bin/env python3
"""
5_3_2026_Test_Driver.py

Test driver for tda_modulev3.py.

Purpose
-------
This script generates deterministic fake embryo centroid CSV files and tests:

1. Fake data generation.
2. Loading through CloudDataManager.
3. Reproducibility of persistence diagram creation.
4. Landscape matrix row-order alignment.
5. Plotting functions, limited to two total plot calls.
6. Full-sample stress testing.
7. Runtime, memory, and CPU logging where feasible.

Important compatibility notes
-----------------------------
The requested filename sketch was:

    {group}_t{timestep}_fake_centroid_name={embryo_name}

However, tda_modulev3.py currently parses timesteps using the pattern:

    t=<integer>

and load_groups_from_directory only loads .csv files.

Therefore this driver writes filenames like:

    control_t=0_fake_centroid_name=control_rep01_embryo01.csv

This preserves the requested naming intent while remaining compatible with the
current loader.

Fake point clouds are 3D by default because tda_modulev3.py expects these columns:

    Center of the object_0
    Center of the object_1
    Center of the object_2
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import psutil
except ImportError:  # psutil is recommended but not strictly required.
    psutil = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Replace this import if your module/class name changes.
# ---------------------------------------------------------------------
from tda_modulev3 import CloudDataManager


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

FAKE_DATA_DIR = Path("/workspaces/TDA-workspace/data/fake_data")
OUTPUT_DIR = Path("test_driver_outputs")

GROUPS = ("control", "epha2a")
N_REPLICATES = 3
N_EMBRYOS = 10
N_TIMEPOINTS = 10
EXPECTED_FILES_PER_GROUP = N_REPLICATES * N_EMBRYOS * N_TIMEPOINTS
EXPECTED_TOTAL_FILES = len(GROUPS) * EXPECTED_FILES_PER_GROUP

POINT_DIMENSION = 3
MIN_POINTS = 80
MAX_POINTS = 120
RANDOM_SEED = 20260503

SUBSET_FRACTION = 0.20

# Keep these moderate for test-driver practicality.
# Increase resolution/n_layers or max_edge_length if you want a heavier stress test.
TDA_PARAMS = {
    "diagram": {
        "max_dimension": 2,
        "max_edge_length": np.inf,
        "homology_coeff_field": 2,
        "min_persistence": 0.0,
    },
    "landscape": {
        "max_dimension": 2,
        "min_persistence": 0.0,
        "resolution": 50,
        "n_layers": 3,
    },
    "matrix_properties": [0, 1],
}


# ---------------------------------------------------------------------
# Performance helpers
# ---------------------------------------------------------------------

@dataclass
class PerformanceRecord:
    name: str
    elapsed_seconds: float
    rss_start_mb: Optional[float]
    rss_end_mb: Optional[float]
    rss_delta_mb: Optional[float]
    cpu_percent: Optional[float]
    notes: str = ""


def _process():
    if psutil is None:
        return None
    return psutil.Process(os.getpid())


def current_rss_mb() -> Optional[float]:
    proc = _process()
    if proc is None:
        return None
    return proc.memory_info().rss / (1024 ** 2)


@contextmanager
def performance_logger(name: str, records: List[PerformanceRecord], notes: str = ""):
    proc = _process()

    if proc is not None:
        proc.cpu_percent(interval=None)

    rss_start = current_rss_mb()
    start = time.perf_counter()

    yield

    elapsed = time.perf_counter() - start
    rss_end = current_rss_mb()

    if proc is not None:
        cpu_percent = proc.cpu_percent(interval=None)
    else:
        cpu_percent = None

    rss_delta = None
    if rss_start is not None and rss_end is not None:
        rss_delta = rss_end - rss_start

    records.append(
        PerformanceRecord(
            name=name,
            elapsed_seconds=elapsed,
            rss_start_mb=rss_start,
            rss_end_mb=rss_end,
            rss_delta_mb=rss_delta,
            cpu_percent=cpu_percent,
            notes=notes,
        )
    )


def print_performance_summary(records: Sequence[PerformanceRecord]) -> None:
    print("\n=== Performance Summary ===")
    for rec in records:
        print(f"\n[{rec.name}]")
        print(f"  elapsed_seconds: {rec.elapsed_seconds:.3f}")

        if rec.rss_start_mb is not None:
            print(f"  rss_start_mb:    {rec.rss_start_mb:.2f}")
            print(f"  rss_end_mb:      {rec.rss_end_mb:.2f}")
            print(f"  rss_delta_mb:    {rec.rss_delta_mb:.2f}")
        else:
            print("  memory:          psutil not available")

        if rec.cpu_percent is not None:
            print(f"  cpu_percent:     {rec.cpu_percent:.2f}")
        else:
            print("  cpu_percent:     psutil not available")

        if rec.notes:
            print(f"  notes:           {rec.notes}")


def write_performance_json(records: Sequence[PerformanceRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2)


# ---------------------------------------------------------------------
# Fake data generation
# ---------------------------------------------------------------------

def clean_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def make_fake_centroids(
    rng: np.random.Generator,
    group: str,
    replicate: int,
    embryo: int,
    timestep: int,
    n_points: int,
) -> np.ndarray:
    """
    Generate one deterministic synthetic 3D point cloud.

    The cloud has a mild group effect and timestep drift so the fake data are
    not completely exchangeable noise. This is useful for exercising trajectory
    and matrix-building behavior.
    """
    assert POINT_DIMENSION == 3, "tda_modulev3.py expects 3 centroid columns."

    group_shift = 0.0 if group == "control" else 2.0
    replicate_shift = 0.15 * replicate
    embryo_shift = 0.05 * embryo
    time_shift = 0.10 * timestep

    center = np.array(
        [
            group_shift + replicate_shift,
            embryo_shift + time_shift,
            0.5 * group_shift + 0.03 * timestep,
        ],
        dtype=float,
    )

    scale = np.array(
        [
            1.0 + 0.02 * timestep,
            1.1 + 0.03 * replicate,
            0.9 + 0.01 * embryo,
        ],
        dtype=float,
    )

    points = rng.normal(loc=center, scale=scale, size=(n_points, POINT_DIMENSION))
    return points


def fake_filename(group: str, replicate: int, embryo: int, timestep: int) -> str:
    """
    Filename compatible with tda_modulev3.py.

    Uses:
      - starts with group
      - contains t=<integer>
      - contains name=<sample_name>
      - ends with .csv

    Embryo names include group and replicate to avoid overwriting manager.groups,
    because tda_modulev3.py keys groups by (name, timestep), not by label.
    """
    embryo_name = f"{group}_rep{replicate:02d}_embryo{embryo:02d}"
    return f"{group}_t={timestep}_fake_centroid_name={embryo_name}.csv"


def generate_fake_data(
    fake_data_dir: Path = FAKE_DATA_DIR,
    seed: int = RANDOM_SEED,
) -> List[Path]:
    """
    Create a clean fake_data folder and generate deterministic centroid CSVs.

    Returns
    -------
    List[Path]
        Sorted paths to all generated CSV files.
    """
    clean_directory(fake_data_dir)

    rng = np.random.default_rng(seed)
    generated: List[Path] = []

    columns = [
        "Center of the object_0",
        "Center of the object_1",
        "Center of the object_2",
    ]

    for group in GROUPS:
        for replicate in range(1, N_REPLICATES + 1):
            for embryo in range(1, N_EMBRYOS + 1):
                for timestep in range(N_TIMEPOINTS):
                    n_points = int(rng.integers(MIN_POINTS, MAX_POINTS + 1))
                    points = make_fake_centroids(
                        rng=rng,
                        group=group,
                        replicate=replicate,
                        embryo=embryo,
                        timestep=timestep,
                        n_points=n_points,
                    )

                    path = fake_data_dir / fake_filename(
                        group=group,
                        replicate=replicate,
                        embryo=embryo,
                        timestep=timestep,
                    )

                    with path.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(columns)
                        writer.writerows(points)

                    generated.append(path)

    generated = sorted(generated)
    assert len(generated) == EXPECTED_TOTAL_FILES, (
        f"Expected {EXPECTED_TOTAL_FILES} files, got {len(generated)}"
    )

    for group in GROUPS:
        count = sum(1 for p in generated if p.name.startswith(group))
        assert count == EXPECTED_FILES_PER_GROUP, (
            f"Expected {EXPECTED_FILES_PER_GROUP} {group} files, got {count}"
        )

    return generated


# ---------------------------------------------------------------------
# Loading and subset selection
# ---------------------------------------------------------------------

def load_sample_paths(fake_data_dir: Path = FAKE_DATA_DIR) -> List[Path]:
    paths = sorted(fake_data_dir.glob("*.csv"))
    assert paths, f"No CSV files found in {fake_data_dir}"
    return paths


def select_subset(
    paths: Sequence[Path],
    fraction: float = SUBSET_FRACTION,
    seed: int = RANDOM_SEED,
) -> List[Path]:
    """
    Select a deterministic 20% subset of generated files.
    """
    assert 0.0 < fraction <= 1.0
    n_subset = max(1, int(round(len(paths) * fraction)))

    rng = np.random.default_rng(seed)
    indices = np.arange(len(paths))
    rng.shuffle(indices)

    selected = sorted(paths[i] for i in indices[:n_subset])
    assert len(selected) == n_subset
    return selected


def copy_subset_to_temp_dir(paths: Sequence[Path], subset_dir: Path) -> Path:
    clean_directory(subset_dir)
    for path in paths:
        shutil.copy2(path, subset_dir / path.name)
    return subset_dir


def load_manager_from_directory(directory: Path) -> CloudDataManager:
    manager = CloudDataManager()
    loaded_ids = manager.load_groups_from_directory(str(directory))
    assert loaded_ids, f"No groups loaded from {directory}"

    listed_ids = manager.list_group_ids()
    assert len(listed_ids) == len(loaded_ids), (
        "Loaded IDs and manager group count disagree. "
        "This usually indicates duplicate (name, timestep) keys."
    )

    return manager


# ---------------------------------------------------------------------
# TDA workflow helpers
# ---------------------------------------------------------------------

def compute_diagrams_and_landscapes(manager: CloudDataManager) -> CloudDataManager:
    """
    Run the core TDA pipeline.

    Replace these calls if your module's public API changes.
    """
    manager.compute_all_persistence_diagrams(**TDA_PARAMS["diagram"])
    manager.compute_all_persistence_landscapes(**TDA_PARAMS["landscape"])
    return manager


# def canonicalize_diagram(diagram_result: Dict[str, Any], decimals: int = 12) -> Tuple[Any, ...]:
#     """
#     Convert a diagram dictionary into a stable comparable representation.

#     Rounding protects against tiny floating-point representation noise.
#     """
#     dgm = diagram_result.get("dgm", [])
#     canonical = []

#     for dim, pair in dgm:
#         birth, death = pair

#         birth_val = round(float(birth), decimals)
#         if np.isinf(death):
#             death_val: Any = "inf"
#         else:
#             death_val = round(float(death), decimals)

#         canonical.append((int(dim), birth_val, death_val))

#     return tuple(sorted(canonical))

def canonicalize_diagram(diagram_result: Dict[str, Any], decimals: int = 12) -> Tuple[Any, ...]:
    """
    Convert a diagram dictionary into a stable comparable representation.

    Infinite deaths are encoded numerically to avoid sorting mixed float/string values.
    """
    dgm = diagram_result.get("dgm", [])
    canonical = []

    for dim, pair in dgm:
        birth, death = pair

        birth_val = round(float(birth), decimals)

        if np.isposinf(death):
            death_val = float("inf")
        elif np.isneginf(death):
            death_val = float("-inf")
        else:
            death_val = round(float(death), decimals)

        canonical.append((int(dim), birth_val, death_val))

    return tuple(sorted(canonical, key=lambda x: (x[0], x[1], x[2])))

def assert_diagrams_present(manager: CloudDataManager) -> None:
    for gid in manager.list_group_ids():
        group = manager.get_group(gid)
        assert "dgm" in group.diagrams, f"Missing diagram for {gid}"
        assert isinstance(group.diagrams["dgm"], list), f"Diagram is not a list for {gid}"


def assert_landscapes_present(manager: CloudDataManager) -> None:
    properties = TDA_PARAMS["matrix_properties"]
    for gid in manager.list_group_ids():
        group = manager.get_group(gid)
        for prop in properties:
            assert prop in group.landscapes, f"Missing H{prop} landscape for {gid}"
            arr = np.asarray(group.landscapes[prop])
            assert arr.ndim == 1, f"H{prop} landscape is not flat for {gid}"
            assert arr.size > 0, f"H{prop} landscape is empty for {gid}"


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_consistency(subset_dir: Path, records: List[PerformanceRecord]) -> None:
    """
    Test that repeated full pipeline runs on identical input produce identical
    persistence diagrams and matrix row order.
    """
    with performance_logger("consistency_run_1", records):
        manager_1 = load_manager_from_directory(subset_dir)
        compute_diagrams_and_landscapes(manager_1)

    with performance_logger("consistency_run_2", records):
        manager_2 = load_manager_from_directory(subset_dir)
        compute_diagrams_and_landscapes(manager_2)

    ids_1 = manager_1.list_group_ids()
    ids_2 = manager_2.list_group_ids()

    assert ids_1 == ids_2, "Group ordering differs across repeated loads."

    for gid in ids_1:
        dgm_1 = canonicalize_diagram(manager_1.get_group(gid).diagrams)
        dgm_2 = canonicalize_diagram(manager_2.get_group(gid).diagrams)
        assert dgm_1 == dgm_2, f"Persistence diagram differs across runs for {gid}"

    matrix_1 = manager_1.build_landscape_matrix(properties=TDA_PARAMS["matrix_properties"])
    matrix_2 = manager_2.build_landscape_matrix(properties=TDA_PARAMS["matrix_properties"])

    assert matrix_1["row_order"] == matrix_2["row_order"], (
        "Landscape matrix row_order differs across repeated runs."
    )
    np.testing.assert_allclose(
        matrix_1["matrix"],
        matrix_2["matrix"],
        rtol=1e-12,
        atol=1e-12,
        err_msg="Landscape matrices differ across repeated runs.",
    )


def test_matrix_row_alignment(manager: CloudDataManager) -> None:
    """
    Test that matrix rows stay parallel with row_order.

    For every row_order entry, rebuild the expected landscape vector directly
    from manager.groups and compare it against the matrix row.
    """
    properties = TDA_PARAMS["matrix_properties"]
    matrix_result = manager.build_landscape_matrix(properties=properties)

    matrix = matrix_result["matrix"]
    row_order = matrix_result["row_order"]

    assert matrix.shape[0] == len(row_order), (
        "Matrix row count does not match row_order length."
    )

    expected_row_count = len(manager.list_group_ids()) * len(properties)
    assert len(row_order) == expected_row_count, (
        f"Expected {expected_row_count} matrix rows, got {len(row_order)}."
    )

    for row_idx, (gid, prop) in enumerate(row_order):
        expected = np.asarray(manager.get_group(gid).landscapes[prop]).flatten()
        observed = matrix[row_idx, : expected.size]

        np.testing.assert_allclose(
            observed,
            expected,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Matrix row {row_idx} is not aligned with row_order entry {(gid, prop)}.",
        )

        if expected.size < matrix.shape[1]:
            padding = matrix[row_idx, expected.size:]
            assert np.allclose(padding, 0.0), (
                f"Nonzero padding found in matrix row {row_idx} for {(gid, prop)}."
            )

    # The module sorts group IDs by (name, timestep), then appends properties in order.
    expected_order = [
        (gid, prop)
        for gid in manager.list_group_ids()
        for prop in properties
    ]

    assert row_order == expected_order, (
        "row_order does not match expected manager.list_group_ids() x properties ordering."
    )


def test_limited_plotting(
    manager: CloudDataManager,
    records: List[PerformanceRecord],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """
    Exercise plotting functions exactly twice total.

    Plot 1:
        manager.plot_tda_summary_for_group

    Plot 2:
        manager.PCA.plot_trajectories

    This intentionally does not call every plotting function to avoid excessive
    plotting cost during stress tests.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    gids = manager.list_group_ids()
    assert gids, "No group IDs available for plotting."

    with performance_logger("plot_1_tda_summary_for_group", records):
        fig1 = manager.plot_tda_summary_for_group(gids[0], add_label=True)
        fig1.savefig(output_dir / "plot_1_tda_summary_for_group.png", dpi=150)
        plt.close(fig1)

    with performance_logger("plot_2_pca_trajectories", records):
        pca_result = manager.PCA.run(
            n_components=2,
            name=None,
            label=None,
            properties=TDA_PARAMS["matrix_properties"],
        )
        assert pca_result["transformed"].shape[1] >= 2, (
            "PCA produced fewer than 2 components; trajectory plot may fail."
        )

        fig2 = manager.PCA.plot_trajectories(
            name=None,
            label=None,
            properties=TDA_PARAMS["matrix_properties"],
            show_trajectories=True,
        )
        fig2.savefig(output_dir / "plot_2_pca_trajectories.png", dpi=150)
        plt.close(fig2)


def run_initial_subset_tests(
    subset_dir: Path,
    records: List[PerformanceRecord],
) -> None:
    """
    Run the 20% test workflow before full-sample stress testing.
    """
    test_consistency(subset_dir=subset_dir, records=records)

    with performance_logger("subset_pipeline_for_alignment_and_plotting", records):
        manager = load_manager_from_directory(subset_dir)
        compute_diagrams_and_landscapes(manager)

    assert_diagrams_present(manager)
    assert_landscapes_present(manager)

    test_matrix_row_alignment(manager)
    test_limited_plotting(manager, records=records)


def run_full_stress_test(
    full_data_dir: Path,
    records: List[PerformanceRecord],
) -> None:
    """
    Run the full sample load after initial subset tests pass.
    """
    with performance_logger(
        "full_stress_load_compute_matrix",
        records,
        notes=(
            "Full load includes all generated files. "
            "Monitor elapsed time and RSS delta for memory/processor pressure."
        ),
    ):
        manager = load_manager_from_directory(full_data_dir)

        expected_loaded = EXPECTED_TOTAL_FILES
        actual_loaded = len(manager.list_group_ids())
        assert actual_loaded == expected_loaded, (
            f"Expected {expected_loaded} loaded groups, got {actual_loaded}. "
            "Likely duplicate (name, timestep) keys or filename parsing issue."
        )

        compute_diagrams_and_landscapes(manager)

        assert_diagrams_present(manager)
        assert_landscapes_present(manager)

        matrix_result = manager.build_landscape_matrix(
            properties=TDA_PARAMS["matrix_properties"]
        )

        expected_rows = EXPECTED_TOTAL_FILES * len(TDA_PARAMS["matrix_properties"])
        assert matrix_result["matrix"].shape[0] == expected_rows, (
            f"Expected full matrix to have {expected_rows} rows, "
            f"got {matrix_result['matrix'].shape[0]}."
        )

        test_matrix_row_alignment(manager)

    del manager
    gc.collect()


def assess_resource_pressure(records: Sequence[PerformanceRecord]) -> Dict[str, Any]:
    """
    Lightweight heuristic summary.

    This does not prove hardware sufficiency. It flags obvious warning signs.
    """
    max_rss_end = None
    max_rss_delta = None

    rss_ends = [r.rss_end_mb for r in records if r.rss_end_mb is not None]
    rss_deltas = [r.rss_delta_mb for r in records if r.rss_delta_mb is not None]

    if rss_ends:
        max_rss_end = max(rss_ends)
    if rss_deltas:
        max_rss_delta = max(rss_deltas)

    warnings = []

    if max_rss_end is None:
        warnings.append("Memory usage was not measured because psutil is unavailable.")
    elif max_rss_end > 8000:
        warnings.append("Peak RSS exceeded 8 GB; memory may be an issue on smaller machines.")

    if max_rss_delta is not None and max_rss_delta > 4000:
        warnings.append("A single test phase increased RSS by more than 4 GB.")

    slow_steps = [r for r in records if r.elapsed_seconds > 600]
    if slow_steps:
        warnings.append("At least one phase exceeded 10 minutes.")

    if not warnings:
        warnings.append("No obvious memory or processor issue was detected by this driver.")

    return {
        "max_rss_end_mb": max_rss_end,
        "max_rss_delta_mb": max_rss_delta,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fake centroid data and test tda_modulev3.py."
    )

    parser.add_argument(
        "--fake-data-dir",
        type=Path,
        default=FAKE_DATA_DIR,
        help="Directory where fake CSV data will be generated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where plots and performance logs will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Deterministic random seed.",
    )
    parser.add_argument(
        "--subset-fraction",
        type=float,
        default=SUBSET_FRACTION,
        help="Fraction of generated files used for initial testing.",
    )
    parser.add_argument(
        "--skip-full-stress",
        action="store_true",
        help="Run only fake data generation and 20%% subset tests.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    records: List[PerformanceRecord] = []

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subset_dir = args.output_dir / "subset_20_percent"

    print("=== TDA Module Test Driver ===")
    print(f"fake_data_dir:     {args.fake_data_dir}")
    print(f"output_dir:        {args.output_dir}")
    print(f"random_seed:       {args.seed}")
    print(f"point_dimension:   {POINT_DIMENSION}")
    print(f"expected_files:    {EXPECTED_TOTAL_FILES}")

    with performance_logger("fake_data_generation", records):
        generated_paths = generate_fake_data(
            fake_data_dir=args.fake_data_dir,
            seed=args.seed,
        )

    print(f"Generated {len(generated_paths)} fake centroid CSV files.")

    all_paths = load_sample_paths(args.fake_data_dir)
    subset_paths = select_subset(
        all_paths,
        fraction=args.subset_fraction,
        seed=args.seed,
    )
    copy_subset_to_temp_dir(subset_paths, subset_dir)

    print(f"Selected {len(subset_paths)} files for initial subset testing.")

    run_initial_subset_tests(subset_dir=subset_dir, records=records)
    print("Initial 20% subset tests passed.")

    if not args.skip_full_stress:
        run_full_stress_test(full_data_dir=args.fake_data_dir, records=records)
        print("Full stress test passed.")
    else:
        print("Full stress test skipped by --skip-full-stress.")

    assessment = assess_resource_pressure(records)

    performance_path = args.output_dir / "performance_log.json"
    write_performance_json(records, performance_path)

    assessment_path = args.output_dir / "resource_assessment.json"
    with assessment_path.open("w", encoding="utf-8") as f:
        json.dump(assessment, f, indent=2)

    print_performance_summary(records)

    print("\n=== Resource Assessment ===")
    print(json.dumps(assessment, indent=2))

    print("\n=== Output Files ===")
    print(f"Performance log:      {performance_path}")
    print(f"Resource assessment:  {assessment_path}")
    print(f"Plots directory:      {args.output_dir}")

    print("\nAll requested tests completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print("\nTEST FAILURE:", exc, file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        print("\nUNEXPECTED ERROR:", repr(exc), file=sys.stderr)
        raise