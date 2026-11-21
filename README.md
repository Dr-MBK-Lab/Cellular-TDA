# TDA Point Cloud Pipeline

A comprehensive Python pipeline designed to load 3D point cloud time-series data (e.g., biological embryos), compute topological descriptors via **GUDHI**, and perform dimensionality reduction and trajectory analysis using **Principal Component Analysis (PCA)**.

## Features

* **Data Management:** Parses and aggregates spatial time-series datasets from directory structures using a unified `CloudDataManager`.
* **Topological Data Analysis:** Computes persistence diagrams (Rips complexes) and extracts multi-layer persistence landscapes using `gudhi`.
* **Dimensionality Reduction:** Formulates landscape-ready feature matrices for scaling and PCA analysis across multiple dimensions and experimental groups.
* **Rich Visualization Engine:** Built-in tools for tracking and rendering experimental trajectories over time-steps, vector fields in PCA space, and an interactive `ipywidgets` slider tracking step-by-step structural changes.
* **Distance Profiling:** Tools for assessing Euclidean deviations from control-group mean shapes over developmental windows.

---

## Architecture Overview

The system is organized into a modular structure where specific tasks are decoupled yet accessible through a central manager API:

```
                  ┌─────────────────────────────────┐
                  │        CloudDataManager         │◄───────── Main public API & entrypoint
                  └────────────────┬────────────────┘
                                   │
      ┌────────────────────────────┼────────────────────────────┐
      ▼                            ▼                            ▼
┌──────────────┐             ┌────────────┐               ┌────────────┐
│ CloudGroup   │             │CloudGroup─ │               │Landscape─  │
│ (Data Model) │             │TDA (Gudhi) │               │Matrix─     │
└──────────────┘             └────────────┘               │Builder     │
                                                          └────────────┘
      ┌────────────────────────────┼────────────────────────────┘
      ▼                            ▼
┌──────────────┐             ┌────────────┐
│ CloudGroup─  │             │TDAPlotter  │
│ PCA          │             │(Matplotlib)│
└──────────────┘             └────────────┘
```

* **`CloudGroup`**: A lightweight data class holding single point-cloud instances, metadata fields, calculated diagrams, and landscapes.
* **`CloudDataManager`**: The orchestrator pipeline handling data splitting, I/O operations (CSV & Pickle formats), and delegating execution tasks to submodules.
* **`CloudGroupTDA`**: The pipeline compute manager engine utilizing GUDHI wrappers for diagrams and landscape feature mapping.
* **`LandscapeMatrixBuilder`**: Filters and formats matrices for downstream statistical learning applications.
* **`CloudGroupPCA`**: Handles vector calculations, variance scoring, and structural spatial mappings.
* **`TDAPlotter`**: Manages underlying 3D renderings, persistence diagrams, landscape charts, and trajectory-vector visualizations.

---

## Prerequisites & Installation

The script requires a standard PyData stack alongside the GUDHI library for topological structures. 

```bash
pip install numpy pandas matplotlib scikit-learn gudhi ipywidgets
```

---

## Core Usage Guide

### 1. Data Initialization & Setup
The input pipeline expects CSV format data profiles containing positional coordinates labeled as `"Center of the object_0"`, `"Center of the object_1"`, `"Center of the object_2"`, and a `"timestep"` field. Filenames should explicitly reference experimental labels (`"control"` or `"epha2a"`).

```python
from your_script_name import CloudDataManager

# Instantiate the primary manager object
manager = CloudDataManager()

# Ingest an entire folder of embryo configurations
manager.load_groups_from_directory("./data/embryos/")
```

### 2. Executing the TDA Pipeline
Run persistence computations across every group structure loaded into the object manager:

```python
# 1. Compute Rips complex diagrams up to dimension H1
manager.compute_all_persistence_diagrams(max_dimension=2, min_persistence=0.0)

# 2. Extract multi-layered persistence landscapes
manager.compute_all_persistence_landscapes(n_layers=5, resolution=100)
```

### 3. PCA Trajectory Analysis
Extract properties and compute principal projection trajectories:

```python
# Fit standard scaler and run PCA coordinates
pca_results = manager.compute_pca(
    condition=None,             # Set explicitly to "control" or "epha2a" to isolate 
    properties=[0, 1],          # Include H0 and H1 components
    n_components=2
)

# Render standard developmental paths
fig_traj = manager.plot_trajectories(properties=[0, 1])
fig_traj.show()
```

### 4. Vector Profiles & Interactive Visualization
To display vectors showing movement through shape space across consecutive developmental time intervals:

```python
# Plot step-by-step continuous vector arrows
fig_vectors = manager.plot_trajectory_vectors(properties=[0, 1], scale=1.0)
fig_vectors.show()

# Open an interactive notebook slider evaluating path trajectories
manager.slider_plot_trajectory_vectors(properties=[0, 1])
```

---

## API Summary Reference Table

| Target Method | Main Delegate Class | Description |
| :--- | :--- | :--- |
| `load_groups_from_directory(path)` | `CloudDataManager` | Ingests structural CSV directories and extracts meta-properties. |
| `compute_all_persistence_diagrams()` | `CloudGroupTDA` | Computes persistence structures via filtration. |
| `compute_all_persistence_landscapes()`| `CloudGroupTDA` | Transforms standard diagrams into vector functional spaces. |
| `build_landscape_matrix()` | `LandscapeMatrixBuilder` | Constructs structured, flatted arrays filtered by conditions. |
| `compute_pca()` | `CloudGroupPCA` | Transforms high-dimensional landscape matrices into low-dimensional PCA space. |
| `plot_trajectory_vectors()` | `CloudGroupPCA` | Visualizes consecutive time shifts as directed spatial vector mappings. |
| `distance_table_from_control_mean()` | `CloudDataManager` | Calculates metric deviations from baseline averages over uniform developmental timesteps. |
