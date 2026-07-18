# run_tda_pipeline.py

import os
import sys
import pickle
from pathlib import Path

import numpy as np

from tda_module.v3_1 import CloudDataManager


def run_tda_pipeline(
    data_dir,
    outfile=None,
    max_dimension=2,
    max_edge_length=np.inf,
    homology_coeff_field=2,
    min_persistence=0.0,
    resolution=10,
    n_layers=5,
    run_pca=True,
    pca_components=2,
    pca_properties=None,
    verbose=True,
):
    """
    Load point-cloud CSV files, compute persistence diagrams and landscapes,
    optionally run PCA, and save the completed CloudDataManager instance.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the CSV files.

    outfile : str or Path, optional
        Output pickle path. If None, saves to data_dir / "tda_manager.pkl".

    Returns
    -------
    manager : CloudDataManager
        The fully computed manager instance.
    """

    data_dir = Path(data_dir).expanduser().resolve()

    if outfile is None:
        outfile = data_dir / "tda_manager.pkl"
    else:
        outfile = Path(outfile).expanduser().resolve()

    if pca_properties is None:
        pca_properties = [0, 1]

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    manager = CloudDataManager()

    loaded_ids = manager.load_groups_from_directory(str(data_dir))

    if verbose:
        print(f"Loaded {len(loaded_ids)} files")
        print(f"Stored {len(manager.groups)} unique groups")
    print("Computing persistence diagrams...")
    manager.compute_all_persistence_diagrams(
        max_dimension=max_dimension,
        max_edge_length=max_edge_length,
        homology_coeff_field=homology_coeff_field,
        min_persistence=min_persistence,
    )
    print("Computing persistence landscapes...")
    manager.compute_all_persistence_landscapes(
        min_persistence=min_persistence,
        resolution=resolution,
        n_layers=n_layers,
    )
    print("Done computing persistence landscapes.")
    print("PCA will be run on the computed persistence landscapes if run_pca=True and there are enough groups to perform PCA.")
    pca_result = None
    if run_pca:
        print("Running PCA on persistence landscapes...")
        pca_result = manager.PCA.run(
            n_components=pca_components,
            embryo_name=None,
            condition=None,
            properties=pca_properties,
        )

        if verbose:
            print("PCA shape:", pca_result["transformed"].shape)
            print("Explained variance:", pca_result["explained_variance_ratio"])

    outfile.parent.mkdir(parents=True, exist_ok=True)

    with open(outfile, "wb") as f:
        pickle.dump(manager, f)

    if verbose:
        print("cwd:", os.getcwd())
        print("python:", sys.executable)
        print("sys.path[0]:", sys.path[0])
        print(f"Saved computed manager to {outfile}")

    return manager