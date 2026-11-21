import os
import random
import re
import pickle
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import gudhi
import pandas as pd
import numpy as np
import gudhi.representations
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import ipywidgets as widgets
from ipywidgets import interact

GroupID = Tuple[str, int]


@dataclass
# Data organizer class for CloudManager. Each CloudGroup represents a single point cloud dataset with associated metadata, diagrams, and landscapes.
class CloudGroup:
    group_id: GroupID
    embryo_name: str
    condition: str
    timestep: int
    original_data: pd.DataFrame
    diagrams: Dict[str, Any] = field(default_factory=dict)
    landscapes: Dict[Any, Any] = field(default_factory=dict)
# Computing class for persistence diagrams and landscapes from point cloud data.
class CloudGroupTDA:
    def __init__(self, manager: "CloudDataManager") -> None:
        self.manager = manager

    def compute_persistence_diagram(
        self,
        points: np.ndarray,
        max_dimension: int = 2,
        max_edge_length: Optional[float] = np.inf,
        homology_coeff_field: int = 2,
        min_persistence: float = 0.0,
    ) -> Dict[str, Any]:
        """Compute a persistence diagram from a point cloud using a Rips complex."""

        pts = np.asarray(points)
        if max_dimension <= 0:
            raise ValueError("max_dimension must be a positive integer")
        
        if pts.ndim != 2 or pts.shape[0] == 0:
            raise ValueError("points must be a non-empty 2D array")

        result: Dict[str, Any] = {
            "metadata": {
                "max_dimension": max_dimension,
                "max_edge_length": max_edge_length,
                "homology_coeff_field": homology_coeff_field,
                "min_persistence": min_persistence,
            }
        }

        rips = gudhi.RipsComplex(points=pts, max_edge_length=max_edge_length)
        simplex_tree = rips.create_simplex_tree(max_dimension=max_dimension)
        simplex_tree.compute_persistence(homology_coeff_field=homology_coeff_field)

        persistence = simplex_tree.persistence(homology_coeff_field=homology_coeff_field)
        result["dgm"] = [
            (dim, (birth, death))
            for dim, (birth, death) in persistence
            if np.isinf(death) or (death - birth) >= min_persistence
        ]
        return result

    def compute_persistence_landscapes(
        self,
        diagram: List[Tuple[int, Tuple[float, float]]],
        min_persistence: float = 0.0,
        resolution: int = 100,
        n_layers: int = 5,
    ) -> Dict[Any, Any]:
        """Compute persistence landscapes and store each dimension's coordinate range."""
        max_dimension = self.manager._tda_params["max_dimension"] if self.manager._tda_params else None
        if max_dimension is None:
            raise ValueError("TDA parameters not found. Run compute_all_persistence_diagrams first.")
        
        result: Dict[Any, Any] = {
            "metadata": {
                "n_layers": n_layers,
                "resolution": resolution,
                "min_persistence": min_persistence,
                "max_dimension": max_dimension,
                "ranges": {},
            }
        }

        for dim in range(self.manager._tda_params["max_dimension"]):
            pts = [
                [birth, death]
                for d, (birth, death) in diagram
                if d == dim
                and np.isfinite(death)
                and (death - birth) >= min_persistence
            ]

            if not pts:
                result[dim] = np.zeros(n_layers * resolution)
                result["metadata"]["ranges"][dim] = (0.0, 1.0)
                continue

            diagram_array = np.array(pts)
            landscape = gudhi.representations.Landscape(
                num_landscapes=n_layers,
                resolution=resolution,
            )
            vector = landscape.fit_transform([diagram_array]).flatten()

            x_min = min(birth for birth, _ in pts)
            x_max = max(death for _, death in pts)
            if x_min == x_max:
                x_max = x_min + 1.0

            result[dim] = vector
            result["metadata"]["ranges"][dim] = (float(x_min), float(x_max))

        return result

    def compute_all_persistence_diagrams(
        self,
        max_dimension: int = 2,
        max_edge_length: Optional[float] = np.inf,
        homology_coeff_field: int = 2,
        min_persistence: float = 0.0,
    ) -> "CloudDataManager":
        """Compute persistence diagrams for every loaded CloudGroup."""
       
        if max_dimension <= 0:
            raise ValueError("max_dimension must be a positive integer")
            
        self.manager._tda_params = {
            "max_dimension": max_dimension,
            "max_edge_length": max_edge_length,
            "homology_coeff_field": homology_coeff_field,
            "min_persistence": min_persistence,
            "dimensions": list(range(max_dimension)),
        }

        for gid, group in self.manager.groups.items():
            group.diagrams = self.compute_persistence_diagram(
                group.original_data.values.copy(),
                max_dimension=max_dimension,
                max_edge_length=max_edge_length,
                homology_coeff_field=homology_coeff_field,
                min_persistence=min_persistence,
            )
            self.manager.groups[gid] = group

        return self.manager

    def compute_all_persistence_landscapes(
        self,
        min_persistence: float = 0.0,
        resolution: int = 100,
        n_layers: int = 5,
    ) -> "CloudDataManager":
        """Compute landscapes for every group using already-computed diagrams."""
        
        if self.manager._tda_params is None:
            raise ValueError("Run compute_all_persistence_diagrams first.")

        max_dimension = self.manager._tda_params["max_dimension"] if self.manager._tda_params else None
        if max_dimension is None:
            raise ValueError("TDA parameters not found. Set max_dimension in compute_all_persistence_diagrams.")

        self.manager._tda_params.update(
            {
                "min_persistence": min_persistence,
                "resolution": resolution,
                "n_layers": n_layers,
                "dimensions": list(range(max_dimension)),
            }
        )

        for gid, group in self.manager.groups.items():
            if "dgm" not in group.diagrams:
                raise ValueError(f"Group {gid} has no persistence diagram.")

            group.landscapes = self.compute_persistence_landscapes(
                diagram=group.diagrams["dgm"],
                min_persistence=min_persistence,
                resolution=resolution,
                n_layers=n_layers,
            )
            self.manager.groups[gid] = group

        return self.manager
# Build PCA-ready matrices from stored persistence landscapes.
class LandscapeMatrixBuilder:
    """Build PCA-ready matrices from stored persistence landscapes."""

    def __init__(self, manager: "CloudDataManager") -> None:
        self.manager = manager

    @staticmethod
    def _normalize_embryo_names(embryo_name: Optional[Iterable[str]]) -> Optional[set]:
        if embryo_name is None:
            return None
        if isinstance(embryo_name, str):
            raise TypeError("embryo_name must be a list/iterable of embryo names, not a string.")
        return set(embryo_name)
    
    def build_landscape_matrix(
        self,
        PCA_key: Optional[Any] = None,
        embryo_name: Optional[List[str]] = None,
        condition: Optional[Union[str, List[str]]] = None,
        properties: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Build a matrix of flattened landscapes filtered by embryo_name, condition, and dimensions.

        Args:
            PCA_key: Can be a dict {"condition": str or [...], "embryo_name": "...", "properties": [...]} 
                     or tuple (condition, embryo_name, properties). When provided, overrides other parameters.
            embryo_name: List of embryo names to include. None includes all embryos.
            condition: Condition or list of conditions to include. None includes all.
            properties: Homology dimensions to include. None uses stored TDA dimensions.
        """
        if PCA_key is not None:
            embryo_name, condition, properties = CloudGroupPCA._normalize_pca_key(PCA_key)
       
        if properties is None:
            if self.manager._tda_params is None:
                raise ValueError("No TDA parameters found. Compute landscapes first.")
            properties = list(self.manager._tda_params["dimensions"])

        embryo_names = self._normalize_embryo_names(embryo_name)
        ids = self.manager.list_group_ids()
        if embryo_names is not None:
            ids = [gid for gid in ids if gid[0] in embryo_names]
      
        if condition is not None:
            conditions = {condition} if isinstance(condition, str) else set(condition)
            ids = [
                gid
                for gid in ids
                if self.manager.groups[gid].condition in conditions
            ]

        row_order = []
        max_width = 0
        
        for gid in ids:
            group = self.manager.groups[gid]
            for prop in properties:
                vector = group.landscapes.get(prop)
                if vector is None:
                    continue
                vector = np.asarray(vector)
                if vector.ndim != 1:
                    vector = vector.flatten()
                row_order.append((gid, prop))
                max_width = max(max_width, vector.size)

        if not row_order:
            raise ValueError("No valid landscape rows match the given criteria.")

        matrix = np.zeros((len(row_order), max_width))
        for row_idx, (gid, prop) in enumerate(row_order):
            vector = np.asarray(self.manager.groups[gid].landscapes[prop]).flatten()
            matrix[row_idx, : vector.size] = vector

        return {
            "matrix": matrix,
            "row_order": row_order,
            "tda_params": self.manager._tda_params,
            "landscape_ranges": self.manager.collect_landscape_ranges(row_order),
        }
# PCA-based analysis for TDA landscape matrices.
class CloudGroupPCA:
    """PCA-based analysis for TDA landscape matrices."""

    def __init__(self, manager: "CloudDataManager") -> None:
        self.manager = manager
        self.results: Dict[Any, Dict[str, Any]] = {}
    
    def run(
    self,
    PCA_key: Optional[Any] = None,
    n_components: int = 2,
    condition: Optional[Union[str, List[str]]] = None,
    embryo_name: Optional[List[str]] = None,
    properties: Optional[List[int]] = None,
) -> Dict[str, Any]:
        """Standardize the landscape matrix and run PCA.

        Args:)
            PCA_key: Can be a dict {"condition": str or [...], "embryo_name": "...", "properties": [...]} 
                     or tuple (condition, embryo_name, properties). None values become "All".
                     When provided, overrides other parameters for key generation.
            n_components: Number of PCA components to compute.
            condition: Condition or list of conditions to include in PCA.
            embryo_name: List of embryo names to include in landscape matrix.
            properties: List of properties (dimensions) to include in landscape matrix.
        """
        
        if properties is None and PCA_key["properties"] is None:
            raise ValueError("Properties (homology dimensions) must be specified for PCA.")
        
        if PCA_key is not None:
            embryo_name, condition, properties = CloudGroupPCA._normalize_pca_key(PCA_key)
        # matrix is filtered and flattened landscapes, row_order is list of (group_id, property) for each row
        pca_ = self.manager.build_landscape_matrix(
            PCA_key=PCA_key,
            condition=condition,
            embryo_name=embryo_name,
            properties=properties,
        )

        scaled_matrix = StandardScaler().fit_transform(pca_["matrix"])
        
        n_components_used = min(
            n_components,
            scaled_matrix.shape[0],
            scaled_matrix.shape[1],
        )
        pca = PCA(n_components=n_components_used, random_state=42)
        transformed = pca.fit_transform(scaled_matrix)

        key = self._key(embryo_name, condition, properties); 

        self.results[key] = {
            "transformed": transformed,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "row_order": pca_["row_order"],
            "components": pca.components_,
            "pca": pca,
            "scaler_mean": scaled_matrix.mean(axis=0),
            "matrix_info": pca_,
        }
        return self.results[key]

    def plot_trajectories(
        self,
        PCA_key: Optional[Any] = None,
        show_trajectories: bool = True,
        do_not_include: Optional[Dict[str, Any]] = None,
        figsize: Tuple[int, int] = (10, 8),
        embryo_name: Optional[List[str]] = None,
        condition: Optional[str] = None,
        properties: Optional[List[int]] = None,
    ) -> plt.Figure:
        """Plot PCA trajectories by embryo name and homology dimension.

        Args:
            PCA_key: Can be a dict {"embryo_name": [...], "condition": "...", "properties": [...]} 
                    or tuple (embryo_name, condition, properties). Overrides other parameters.
            plot_criteria: Legacy dict parameter with embryo_name, condition, properties.
            do_not_include: Exclusion criteria dict with properties, embryo_names, condition to exclude.
            show_trajectories: Whether to plot connecting lines and arrows.
            figsize: Figure size tuple.
            embryo_name: Fallback list of embryo names to include.
            condition: Fallback condition string to include.
            properties: Fallback list of properties to include.
        """
        if PCA_key is not None:
            embryo_name, condition, properties = CloudGroupPCA._normalize_pca_key(PCA_key)
        if properties is None:
            properties = [0, 1]

        key = self._key(embryo_name, condition, properties)

        if key not in self.results:
            raise ValueError(f"Run PCA for {key} first.")

        res = self.results[key]
        coords = res["transformed"]
        row_order = res["row_order"]
        coords, metadata = self._filter_plot_rows(coords, row_order, do_not_include)

        unique_names = sorted({grp.embryo_name for grp, _ in metadata})
        unique_conditions = sorted({grp.condition for grp, _ in metadata})
        unique_props = sorted({prop for _, prop in metadata})

        condition_color_map = {
            "control-l": "blue",
            "control-m": "blue",
            "epha2a-wm": "red",
            "epha2a-sm": "red",
        }

        fig = plt.figure(figsize=figsize, dpi=120)

        if show_trajectories:
            for embryo_name_val in unique_names:
                for prop in unique_props:
                    sample_indices = [
                        i
                        for i, (grp, p) in enumerate(metadata)
                        if grp.embryo_name == embryo_name_val and p == prop
                    ]
                    sample_indices = sorted(
                        sample_indices,
                        key=lambda i: metadata[i][0].timestep,
                    )

                    if len(sample_indices) > 1:
                        traj_coords = coords[sample_indices]
                        cond = metadata[sample_indices[0]][0].condition
                        color = condition_color_map.get(cond, "gray")
                        plt.plot(
                            traj_coords[:, 0],
                            traj_coords[:, 1],
                            color=color,
                            alpha=0.3,
                            linestyle="-",
                            linewidth=2,
                        )
                        plt.arrow(
                            traj_coords[-2, 0],
                            traj_coords[-2, 1],
                            (traj_coords[-1, 0] - traj_coords[-2, 0]) * 0.9,
                            (traj_coords[-1, 1] - traj_coords[-2, 1]) * 0.9,
                            shape="full",
                            lw=0,
                            length_includes_head=True,
                            head_width=0.02,
                            color=color,
                            alpha=0.5,
                        )

        for cond in unique_conditions:
            idx = [i for i, (grp, _) in enumerate(metadata) if grp.condition == cond]
            if idx:
                plt.scatter(
                    coords[idx, 0],
                    coords[idx, 1],
                    color=condition_color_map.get(cond, "gray"),
                    label=cond,
                    s=300,
                    edgecolors="white",
                    zorder=5,
                )

        for prop in unique_props:
            idx = [i for i, (_, p) in enumerate(metadata) if p == prop]
            if idx:
                color = "skyblue" if prop == 1 else "green"
                plt.scatter(
                    coords[idx, 0],
                    coords[idx, 1],
                    color=color,
                    marker="o",
                    s=150,
                    zorder=10,
                    label=f"H{prop}",
                )

        for i, (grp, _) in enumerate(metadata):
            plt.text(
                coords[i, 0],
                coords[i, 1],
                str(grp.timestep),
                color="white",
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=15,
            )
        evr = res["explained_variance_ratio"]
        x_explain = evr[0]*100
        y_explain = evr[1]*100
        plt.xlabel(f' PC1 ({x_explain:.2f})\%', fontsize=14)
        plt.ylabel(f" PC2 ({(y_explain):.2f})\%", fontsize=14)
        print({x_explain,y_explain})
        plt.title(
            f"TDA PCA Trajectories: {embryo_name or ''} {condition or ''} H{properties}\n"
            "Lines connect timesteps for each sample and homology"
        )
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.5)
        fig.tight_layout()
        plt.close(fig)
        return fig
    
    def plot_trajectory_vectors(
        self,
        PCA_key: Optional[Any] = None,
        plot_criteria: Optional[Dict[str, Any]] = None,
        show_trajectories: bool = True,
        do_not_include: Optional[Dict[str, Any]] = None,
        figsize: Tuple[int, int] = (12, 9),
        scale: float = 1.0,
        annotate: bool = False,
        embryo_name: Optional[List[str]] = None,
        condition: Optional[str] = None,
        properties: Optional[List[int]] = None,
    ) -> plt.Figure:
        """Plot embryo paths through PCA space as a sequence of vectors.

        Each embryo/property trajectory is rendered as a series of consecutive arrows
        from one timestep to the next. The entire path for one embryo/property is
        drawn in a single color.

        Args:
            PCA_key: Can be a dict {"embryo_name": [...], "condition": "...", "properties": [...]} 
                    or tuple (embryo_name, condition, properties). Overrides other parameters.
            plot_criteria: Legacy dict parameter with embryo_name, condition, properties.
            display: Exclusion criteria dict with properties, embryo_names, condition to exclude.
            show_trajectories: Whether to plot connecting vector arrows.
            figsize: Figure size tuple.
            scale: Scale factor for arrow vectors.
            annotate: Whether to annotate trajectory endpoints.
            embryo_name: Fallback list of embryo names to include.
            condition: Fallback condition string to include.
            properties: Fallback list of properties to include.
        """
        if PCA_key is not None:
            embryo_name, condition, properties = CloudGroupPCA._normalize_pca_key(PCA_key)
        elif plot_criteria is not None:
            embryo_name = plot_criteria.get("embryo_name", embryo_name)
            condition = plot_criteria.get("condition", condition)
            properties = plot_criteria.get("properties", properties)

        if properties is None:
            properties = [0, 1]

        key = self._key(embryo_name, condition, properties)
        if key not in self.results:
            raise ValueError(f"Run PCA for {key} first.")

        res = self.results[key]
        coords = res["transformed"]

        if coords.shape[1] < 2:
            raise ValueError("Need at least two PCA components to plot trajectory vectors.")

        row_order = res["row_order"]
        coords, metadata = self._filter_plot_rows(coords, row_order, do_not_include)

        unique_names = sorted({grp.embryo_name for grp, _ in metadata})
        unique_props = sorted({prop for _, prop in metadata})

        # Create a unique color for each embryo
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab20')
        embryo_color_map = {
            name: cmap(i % 20) for i, name in enumerate(unique_names)
        }

        fig, ax = plt.subplots(figsize=figsize, ) #dpi=120)

        for embryo_name_val in unique_names:
            for prop in unique_props:
                sample_indices = [
                    i
                    for i, (grp, p) in enumerate(metadata)
                    if grp.embryo_name == embryo_name_val and p == prop
                ]

                sample_indices = sorted(
                    sample_indices,
                    key=lambda i: metadata[i][0].timestep,
                )

                if len(sample_indices) < 2:
                    continue

                traj_coords = coords[sample_indices]
                color = embryo_color_map[embryo_name_val]

                if show_trajectories:
                    for j in range(len(traj_coords) - 1):
                        start = traj_coords[j]
                        end = traj_coords[j + 1]
                        vec = (end - start) * scale
                        ax.quiver(
                            start[0],
                            start[1],
                            vec[0],
                            vec[1],
                            angles="xy",
                            scale_units="xy",
                            scale=1,
                            color=color,
                            alpha=0.8,
                            width=0.006,
                            headwidth=5,
                            headlength=7,
                            headaxislength=6,
                            zorder=4,
                        )

                ax.scatter(
                    traj_coords[:, 0],
                    traj_coords[:, 1],
                    color=color,
                    marker="o",
                    s=60,
                    edgecolors="white",
                    linewidth=0.8,
                    zorder=5,
                )

                ax.scatter(
                    traj_coords[0, 0],
                    traj_coords[0, 1],
                    color=color,
                    marker="x",
                    s=100,
                    linewidth=2,
                    zorder=6,
                )

                if annotate:
                    ax.text(
                        traj_coords[-1, 0] + 0.01,
                        traj_coords[-1, 1] + 0.01,
                        f"{embryo_name_val} H{prop}",
                        fontsize=8,
                        ha="left",
                        va="bottom",
                        color="black",
                        zorder=6,
                    )

        evr = res["explained_variance_ratio"]
        x_exp = evr[0]*100
        y_exp = evr[1]*100
        ax.set_xlabel(f"PC1 ({x_exp:.2f})\%", fontsize=14)
        ax.set_ylabel(f"PC2 ({y_exp:.2f})\%", fontsize=14)

        ax.set_title(
            f"Trajectory Vectors in PCA Space: {embryo_name or ''} "
            f"{condition or ''} H{properties}\n"
            "Each arrow shows one embryo's timestep-to-timestep motion"
        )

        ax.axhline(0, color="black", linewidth=0.8, alpha=0.3)
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.3)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        plt.close(fig)
        return fig

    def slider_plot_trajectory_vectors(
        self,
        PCA_key: Optional[Any] = None,
        plot_criteria: Optional[Dict[str, Any]] = None,
        show_trajectories: bool = True,
        do_not_include: Optional[Dict[str, Any]] = None,
        figsize: Tuple[int, int] = (8, 6),
        scale: float = 1.0,
        annotate: bool = False,
        embryo_name: Optional[List[str]] = None,
        condition: Optional[str] = None,
        properties: Optional[List[int]] = None,
    ):
        """Displays an interactive plot of embryo paths through PCA space with a slider.

        The slider controls the number of vectors (timesteps) to display for each trajectory.
        This method uses ipywidgets and is intended for use in a Jupyter environment.
        It does not return a figure object but displays an interactive widget.

        Args:
            PCA_key: Can be a dict {"embryo_name": [...], "condition": "...", "properties": [...]}
                     or tuple (embryo_name, condition, properties). Overrides other parameters.
            plot_criteria: Legacy dict parameter with embryo_name, condition, properties.
            display: Exclusion criteria dict with properties, embryo_names, condition to exclude.
            show_trajectories: Whether to plot connecting vector arrows.
            figsize: Figure size tuple.
            scale: Scale factor for arrow vectors.
            annotate: Whether to annotate trajectory endpoints.
            embryo_name: Fallback list of embryo names to include.
            condition: Fallback condition string to include.
            properties: Fallback list of properties to include.
        """
        if PCA_key is not None:
            embryo_name, condition, properties = CloudGroupPCA._normalize_pca_key(PCA_key)
        elif plot_criteria is not None:
            embryo_name = plot_criteria.get("embryo_name", embryo_name)
            condition = plot_criteria.get("condition", condition)
            properties = plot_criteria.get("properties", properties)

        if properties is None:
            properties = [0, 1]

        key = self._key(embryo_name, condition, properties)
        if key not in self.results:
            raise ValueError(f"Run PCA for {key} first.")

        res = self.results[key]
        coords = res["transformed"]

        if coords.shape[1] < 2:
            raise ValueError("Need at least two PCA components to plot trajectory vectors.")

        row_order = res["row_order"]
        filtered_coords, metadata = self._filter_plot_rows(coords, row_order, do_not_include)

        unique_names = sorted({grp.embryo_name for grp, _ in metadata})
        unique_props = sorted({prop for _, prop in metadata})

        # Determine axis limits once to keep them fixed across slider updates.
        if filtered_coords.size > 0:
            x_min, x_max = filtered_coords[:, 0].min(), filtered_coords[:, 0].max()
            y_min, y_max = filtered_coords[:, 1].min(), filtered_coords[:, 1].max()
            x_pad = (x_max - x_min) * 0.1 if (x_max - x_min) > 1e-6 else 0.1
            y_pad = (y_max - y_min) * 0.1 if (y_max - y_min) > 1e-6 else 0.1
            xlim = (x_min - x_pad, x_max + x_pad)
            ylim = (y_min - y_pad, y_max + y_pad)
        else:
            xlim = (-1, 1)
            ylim = (-1, 1)

        max_vectors = 0
        for embryo_name_val in unique_names:
            for prop in unique_props:
                sample_indices = [
                    i
                    for i, (grp, p) in enumerate(metadata)
                    if grp.embryo_name == embryo_name_val and p == prop
                ]
                if len(sample_indices) > 1:
                    max_vectors = max(max_vectors, len(sample_indices) - 1)

        def _plot_for_slider(num_vectors: int):
            import matplotlib.cm as cm
            from matplotlib.colors import LinearSegmentedColormap
            cmap = cm.get_cmap('tab20')
            embryo_color_map = {name: cmap(i % 20) for i, name in enumerate(unique_names)}

            fig, ax = plt.subplots(figsize=figsize)
            
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            for embryo_name_val in unique_names:
                base_color = embryo_color_map[embryo_name_val]
                # Create a lighter starting color by mixing with white
                light_color = (
                    base_color[0] + (1 - base_color[0]) * 0.7,
                    base_color[1] + (1 - base_color[1]) * 0.7,
                    base_color[2] + (1 - base_color[2]) * 0.7,
                    base_color[3]
                )
                embryo_cmap = LinearSegmentedColormap.from_list('embryo_grad', [light_color, base_color])

                for prop in unique_props:
                    sample_indices = sorted(
                        [
                        i for i, (grp, p) in enumerate(metadata)
                        if grp.embryo_name == embryo_name_val and p == prop
                        ],
                        key=lambda i: metadata[i][0].timestep,
                    )

                    if len(sample_indices) < 2:
                        continue

                    traj_coords = filtered_coords[sample_indices]
                    num_points_to_plot = min(len(traj_coords), num_vectors + 1)
                    num_total_points = len(traj_coords)
                    denominator = num_total_points - 1 if num_total_points > 1 else 1

                    if show_trajectories:
                        if num_total_points > 1:
                            num_current_vectors = num_points_to_plot - 1
                            last_vec_idx = num_current_vectors - 1
                            for j in range(num_current_vectors):
                                start, end = traj_coords[j], traj_coords[j + 1]
                                vec = (end - start) * scale

                                # Color darkens with absolute position in the full trajectory
                                color_fraction = j / denominator
                                vector_color = embryo_cmap(color_fraction)

                                # Alpha logic: 5 most recent vectors fade from 1.0 down to 0.6.
                                # All older vectors have a fixed alpha of 0.1.
                                distance_from_last = last_vec_idx - j
                                if distance_from_last < 3:
                                    vector_alpha = 1.0 - distance_from_last * 0.1
                                else:
                                    vector_alpha = 0.4

                                ax.quiver(start[0], start[1], vec[0], vec[1], angles="xy", scale_units="xy", scale=1,
                                          color=vector_color, alpha=vector_alpha, width=0.006, headwidth=5, headlength=7,
                                          headaxislength=6, zorder=4)

                    point_colors = [embryo_cmap(i / denominator) for i in range(num_points_to_plot)]
                    # ax.scatter(
                    #     traj_coords[:num_points_to_plot, 0],
                    #     traj_coords[:num_points_to_plot, 1],
                    #     color=point_colors,
                    #     marker="o", s=60, edgecolors="white", linewidth=0.8, zorder=5,
                    # )

                    if num_points_to_plot > 0:
                        ax.scatter(traj_coords[0, 0], traj_coords[0, 1], color=embryo_cmap(0.0), marker="x", s=100, linewidth=2, zorder=6)

                    if annotate and num_points_to_plot > 0:
                        ax.text(traj_coords[num_points_to_plot - 1, 0] + 0.01, traj_coords[num_points_to_plot - 1, 1] + 0.01,
                                f"{embryo_name_val} H{prop}", fontsize=8, ha="left", va="bottom", color="black", zorder=6)

            evr = res["explained_variance_ratio"]
            ax.set_xlabel(f"PC1 ({evr[0]*100:.2f}\%)", fontsize=14)
            ax.set_ylabel(f"PC2 ({evr[1]*100:.2f}\%)", fontsize=14)
            ax.set_title(f"Trajectory Vectors in PCA Space (showing {num_vectors} vectors)\nData: {embryo_name or 'All'} | {condition or 'All'} | H{properties}")
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.3); ax.axvline(0, color="black", linewidth=0.8, alpha=0.3)
            ax.grid(True, linestyle=":", alpha=0.4); ax.set_aspect("equal", adjustable="datalim"); plt.show()

        interact(_plot_for_slider, num_vectors=widgets.IntSlider(min=0, max=max_vectors, step=1, value=1, description='Num Vectors'))

    @staticmethod
    def _key(
        embryo_name: Optional[List[str]],
        condition: Optional[Union[str, List[str]]],
        properties: Optional[List[int]],
    ) -> Tuple[Any, Any, Tuple[int, ...]]:
        embryo_name_key = "All" if embryo_name is None else tuple(embryo_name)
        if condition is None:
            condition_key = "All"
        elif isinstance(condition, str):
            condition_key = condition
        else:
            condition_key = tuple(condition)
        prop_key = tuple([0,1] if properties is None else properties)
        return embryo_name_key, condition_key, prop_key

    @staticmethod
    def _normalize_pca_key(
        pca_key: Optional[Any],
    ) -> Tuple[
        Optional[List[str]],
        Optional[Union[str, List[str]]],
        Optional[List[int]],
    ]:
        """Normalize PCA_key from dict or tuple format to (embryo_name, condition, properties).

        Accepts:
        - dict: {"embryo_name": [...], "condition": str or [...], "properties": [...]}
        - tuple: (embryo_name, condition or [conditions], properties)
        - None: returns (None, None, None)

        Converts "All" into None and wraps single values in lists.
        """
        if pca_key is None:
            return None, None, None

        if isinstance(pca_key, dict):
            embryo_name = pca_key.get("embryo_name")
            condition = pca_key.get("condition")
            properties = pca_key.get("properties")
        elif isinstance(pca_key, tuple):
            if len(pca_key) == 3:
                embryo_name, condition, properties = pca_key
            else:
                raise ValueError(
                    f"PCA_key tuple must have 3 elements, got {len(pca_key)}"
                )
        else:
            raise TypeError(
                f"PCA_key must be dict or tuple, got {type(pca_key).__name__}"
            )

        if condition == "All" or condition is None:
            condition = None
        elif isinstance(condition, str):
            pass
        elif isinstance(condition, (list, tuple)):
            condition = list(condition)
            if not all(isinstance(item, str) for item in condition):
                raise TypeError("PCA_key conditions must be strings.")
        else:
            raise TypeError(
                "PCA_key condition must be a string or list/tuple of strings."
            )
        if embryo_name == "All" or embryo_name is None:
            embryo_name = None
        elif isinstance(embryo_name, str):
            embryo_name = [embryo_name]
        elif isinstance(embryo_name, (list, tuple)):
            embryo_name = list(embryo_name)

        if properties == "All" or properties == () or properties is None:
            properties = None
        elif isinstance(properties, int):
            properties = [properties]
        elif isinstance(properties, (list, tuple)):
            properties = list(properties)

        return embryo_name, condition, properties
    @staticmethod
    def _normalize_display(display: Optional[Dict[str, Any]]) -> Dict[str, List[Any]]:
        if display is None:
            return {"properties": [], "embryo_names": [], "condition": []}

        properties = list(display.get("properties") or [])
        embryo_names = list(display.get("embryo_names") or [])
        condition = display.get("condition") or []
        if isinstance(condition, str):
            condition = [condition]

        return {
            "properties": properties,
            "embryo_names": embryo_names,
            "condition": list(condition),
        }  

    def _filter_plot_rows(
        self, 
        coords: np.ndarray,
        row_order: List[Tuple[GroupID, int]],
        do_not_include: Optional[Dict[str, Any]],
    ) -> Tuple[np.ndarray, List[Tuple[CloudGroup, int]]]:
        filters = CloudGroupPCA._normalize_display(do_not_include)
        exclude_props = set(filters["properties"])
        exclude_names = set(filters["embryo_names"])
        exclude_conditions = set(filters["condition"])

        metadata = [(self.manager.groups[gid], prop) for gid, prop in row_order]
        filtered_indices = [
            i
            for i, (grp, prop) in enumerate(metadata)
            if prop not in exclude_props
            and grp.embryo_name not in exclude_names
            and grp.condition not in exclude_conditions
        ]

        if not filtered_indices:
            raise ValueError("Display filters excluded all plotted rows.")

        return coords[filtered_indices], [metadata[i] for i in filtered_indices]

    def plot_small_multiples(self, PCA_key=None, figsize=(15, 10)):
        """Strategy 1: Small Multiples (Subplotting)"""
        import math
        import matplotlib.cm as cm
        
        embryo_name, condition, properties = self._normalize_pca_key(PCA_key)
        key = self._key(embryo_name, condition, properties)
        res = self.results[key]
        coords, metadata = self._filter_plot_rows(res["transformed"], res["row_order"], None)
        
        unique_names = sorted({grp.embryo_name for grp, _ in metadata})
        n_embryos = len(unique_names)
        cols = 3
        rows = math.ceil(n_embryos / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True, sharey=True)
        axes = axes.flatten()
        cmap = cm.get_cmap('tab20')

        for idx, embryo_val in enumerate(unique_names):
            ax = axes[idx]
            color = cmap(idx % 20)
            
            indices = [i for i, (grp, p) in enumerate(metadata) if grp.embryo_name == embryo_val]
            indices = sorted(indices, key=lambda i: metadata[i][0].timestep)
            
            if len(indices) > 1:
                traj = coords[indices]
                ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.6, lw=2)
                ax.scatter(traj[:, 0], traj[:, 1], color=color, s=20)
                ax.scatter(traj[0, 0], traj[0, 1], color='black', marker='x', s=60, label='Start')
                
            ax.set_title(embryo_val)
            ax.grid(True, linestyle=":", alpha=0.4)

        for i in range(n_embryos, len(axes)):
            fig.delaxes(axes[i])
            
        fig.suptitle(f"PCA Trajectories (Small Multiples): H{properties}")
        plt.tight_layout()
        return fig

    def plot_condition_averages(self, PCA_key=None, figsize=(10, 8)):
        """Strategy 2: Condition Averaging (Macro View)"""
        embryo_name, condition, properties = self._normalize_pca_key(PCA_key)
        key = self._key(embryo_name, condition, properties)
        res = self.results[key]
        coords, metadata = self._filter_plot_rows(res["transformed"], res["row_order"], None)
        
        df = pd.DataFrame({
            'PC1': coords[:, 0], 'PC2': coords[:, 1],
            'Condition': [grp.condition for grp, _ in metadata],
            'Timestep': [grp.timestep for grp, _ in metadata]
        })
        
        agg_df = df.groupby(['Condition', 'Timestep']).agg(['mean', 'std']).reset_index()
        
        fig, ax = plt.subplots(figsize=figsize)
        colors = {'control': 'blue', 'epha2a': 'red'}
        
        for cond in agg_df['Condition'].unique():
            cond_data = agg_df[agg_df['Condition'] == cond].sort_values('Timestep')
            t = cond_data['Timestep']
            pc1_mean, pc1_std = cond_data['PC1']['mean'], cond_data['PC1']['std']
            pc2_mean, pc2_std = cond_data['PC2']['mean'], cond_data['PC2']['std']
            
            color = colors.get(cond, 'green')
            ax.plot(pc1_mean, pc2_mean, label=f"{cond} (Mean)", color=color, linewidth=3)
            ax.scatter(pc1_mean.iloc[0], pc2_mean.iloc[0], color=color, marker='s', s=100) # Start point
            
            # Shaded variance approximations
            ax.fill_between(pc1_mean, pc2_mean - pc2_std, pc2_mean + pc2_std, color=color, alpha=0.15)
            
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Condition Averaged Trajectories with Std Dev")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.4)
        return fig

    def plot_spline_smoothing(self, PCA_key=None, smooth_factor=1, step=5, figsize=(12, 9)):
        """Strategy 3: Downsampling and Spline Smoothing"""
        from scipy.interpolate import make_interp_spline
        import matplotlib.cm as cm

        embryo_name, condition, properties = self._normalize_pca_key(PCA_key)
        key = self._key(embryo_name, condition, properties)
        res = self.results[key]
        coords, metadata = self._filter_plot_rows(res["transformed"], res["row_order"], None)
        
        unique_names = sorted({grp.embryo_name for grp, _ in metadata})
        fig, ax = plt.subplots(figsize=figsize)
        cmap = cm.get_cmap('tab20')
        
        for idx, embryo_val in enumerate(unique_names):
            indices = [i for i, (grp, p) in enumerate(metadata) if grp.embryo_name == embryo_val]
            indices = sorted(indices, key=lambda i: metadata[i][0].timestep)
            
            if len(indices) > smooth_factor:
                traj = coords[indices]
                color = cmap(idx % 20)
                
                # B-Spline smoothing
                # t = np.arange(len(traj))
                # t_smooth = np.linspace(0, len(traj) - 1, 300)
                # spline_x = make_interp_spline(t, traj[:, 0], k=smooth_factor)(t_smooth)
                # spline_y = make_interp_spline(t, traj[:, 1], k=smooth_factor)(t_smooth)
                
                ax.plot(spline_x, spline_y, color=color, alpha=0.7, lw=2)
                
                # Downsampled arrows
                downsampled = traj[::step]
                for j in range(len(downsampled) - 1):
                    start, end = downsampled[j], downsampled[j+1]
                    ax.annotate('', xy=end, xytext=start,
                                arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.8))
                                
                ax.scatter(traj[0, 0], traj[0, 1], color=color, marker='x', s=80)

        ax.set_title("Spline Smoothed Trajectories with Downsampled Vectors")
        ax.grid(True, linestyle=":", alpha=0.4)
        return fig

    def plot_averaged_vectors(
        self, 
        PCA_key=None, 
        do_not_include=None, 
        step=5, 
        figsize=(12, 9)
    ):
        """Strategy 3: Averaged Trajectory Vectors
        
        Averages the PCA coordinates over every `step` timesteps to reduce 
        visual clutter, plotting the resulting macro-trajectory as connected arrows.
        Filters out specific labels, conditions, or properties using do_not_include.
        """
  

        embryo_name, condition, properties = self._normalize_pca_key(PCA_key)
        key = self._key(embryo_name, condition, properties)
        
        if key not in self.results:
            raise ValueError(f"Run PCA for {key} first.")
            
        res = self.results[key]
        
        # We now pass the do_not_include dictionary into your existing filter method
        coords, metadata = self._filter_plot_rows(res["transformed"], res["row_order"], do_not_include)
        
        unique_names = sorted({grp.embryo_name for grp, _ in metadata})
        fig, ax = plt.subplots(figsize=figsize)
        cmap = cm.get_cmap('tab20')
        
        for idx, embryo_val in enumerate(unique_names):
            # Only process if this embryo hasn't been completely filtered out
            indices = [i for i, (grp, p) in enumerate(metadata) if grp.embryo_name == embryo_val]
            if not indices:
                continue
                
            indices = sorted(indices, key=lambda i: metadata[i][0].timestep)
            
            if len(indices) >= step:
                traj = coords[indices]
                color = cmap(idx % 20)
                
                # 1. Chunking and Averaging the trajectory
                averaged_points = []
                for i in range(0, len(traj), step):
                    chunk = traj[i:i+step]
                    averaged_points.append(chunk.mean(axis=0))
                averaged_points = np.array(averaged_points)
                
                # 2. Draw lines connecting the averaged points
                ax.plot(averaged_points[:, 0], averaged_points[:, 1], color=color, alpha=0.5, lw=2)
                
                # 3. Draw arrows between the averaged window points
                for j in range(len(averaged_points) - 1):
                    start, end = averaged_points[j], averaged_points[j+1]
                    ax.annotate('', xy=end, xytext=start,
                                arrowprops=dict(arrowstyle="->", color=color, lw=2, alpha=0.9))
                                
                # Mark the starting point of the averaged trajectory
                ax.scatter(averaged_points[0, 0], averaged_points[0, 1], color='black', marker='x', s=80, zorder=5)
                
                # Optional: Add a label at the end of the trajectory so you know which is which
                ax.text(averaged_points[-1, 0], averaged_points[-1, 1], embryo_val, fontsize=9, zorder=10)

        evr = res["explained_variance_ratio"]
        ax.set_xlabel(f"PC1 ({evr[0]*100:.2f}%)")
        ax.set_ylabel(f"PC2 ({evr[1]*100:.2f}%)")
        ax.set_title(f"Averaged Trajectory Vectors (Window = {step})")
        ax.grid(True, linestyle=":", alpha=0.4)
        
        plt.tight_layout()
        plt.close(fig)
        return fig

    def plot_sequential_vectors(
        self,
        PCA_key=None,
        do_not_include=None,
        figsize=(12, 9),
    ):
        """Plot trajectories with sequentially reoriented displacement vectors.

        For each embryo and homology property, the first displacement is kept
        unchanged. Each later displacement is oriented along the cumulative sum
        of the original displacements so far, while retaining its original
        length. Segments with a zero cumulative direction are skipped.
        """
        embryo_name, condition, properties = self._normalize_pca_key(PCA_key)
        key = self._key(embryo_name, condition, properties)
        if key not in self.results:
            raise ValueError(f"Run PCA for {key} first.")

        res = self.results[key]
        coords, metadata = self._filter_plot_rows(
            res["transformed"], res["row_order"], do_not_include
        )
        trajectories = sorted(
            {(group.embryo_name, prop) for group, prop in metadata},
            key=lambda item: (item[0], item[1]),
        )

        fig, ax = plt.subplots(figsize=figsize)
        cmap = cm.get_cmap("tab20")
        plotted_points = []

        for trajectory_idx, (embryo_val, prop) in enumerate(trajectories):
            indices = [
                i
                for i, (group, row_prop) in enumerate(metadata)
                if group.embryo_name == embryo_val and row_prop == prop
            ]
            indices.sort(key=lambda i: metadata[i][0].timestep)
            trajectory = coords[indices]
            start = trajectory[0].copy()
            plotted_points.append(start)
            color = cmap(trajectory_idx % 20)

            original_vectors = np.diff(trajectory, axis=0)
            current_point = start.copy()
            path_points = [current_point.copy()]

            for vector_idx, vector in enumerate(original_vectors):
                if vector_idx == 0:
                    adjusted_vector = vector.copy()
                else:
                    cumulative_vector = original_vectors[: vector_idx + 1].sum(axis=0)
                    cumulative_norm = np.linalg.norm(cumulative_vector)
                    if cumulative_norm <= np.finfo(float).eps:
                        continue
                    adjusted_vector = (
                        cumulative_vector / cumulative_norm
                    ) * np.linalg.norm(vector)

                if np.linalg.norm(adjusted_vector) <= np.finfo(float).eps:
                    continue

                next_point = current_point + adjusted_vector
                ax.annotate(
                    "",
                    xy=next_point,
                    xytext=current_point,
                    arrowprops=dict(
                        arrowstyle="->",
                        color=color,
                        linewidth=1.8,
                        alpha=0.85,
                    ),
                )
                current_point = next_point
                path_points.append(current_point.copy())

            path = np.asarray(path_points)
            plotted_points.extend(path[1:])
            ax.plot(path[:, 0], path[:, 1], color=color, alpha=0.5, linewidth=1.5)
            ax.scatter(start[0], start[1], color="black", marker="x", s=70, zorder=5)
            label = embryo_val if len({p for _, p in trajectories}) == 1 else f"{embryo_val} H{prop}"
            ax.text(
                current_point[0],
                current_point[1],
                label,
                fontsize=8,
                ha="left",
                va="bottom",
            )

        points_array = np.asarray(plotted_points)
        x_min, y_min = points_array.min(axis=0)
        x_max, y_max = points_array.max(axis=0)
        x_padding = (x_max - x_min) * 0.05 if x_max > x_min else 0.1
        y_padding = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

        explained_variance = res["explained_variance_ratio"]
        ax.set_xlabel(f"PC1 ({explained_variance[0] * 100:.2f}%)")
        ax.set_ylabel(f"PC2 ({explained_variance[1] * 100:.2f}%)")
        ax.set_title("Sequentially Reoriented Trajectory Vectors")
        ax.grid(True, linestyle=":", alpha=0.4)
        fig.tight_layout()
        plt.close(fig)
        return fig
    
    def plot_interactive_plotly(
        self,
        PCA_key=None,
        showTrajectories=True,
        do_not_include=None,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        features: Optional[Dict[str, Any]] = None,
        *,
        PCA_Space=None,
    ):
        """Create an interactive Plotly view with configurable display features.
        ``features`` accepts:
        - ``show_trajectories``: draw connecting lines (default: True).
        - ``x_range`` / ``y_range``: fixed axis bounds; omitted ranges are padded
          around the displayed points.
        - ``do_not_include``: rows to exclude, using the standard display filters.
        - ``markers``: show markers (bool), map embryo names to Plotly symbols
          (for example ``{"CL1": "square"}``), or configure global ``size``,
          ``symbol``, ``opacity``, ``line_width``, and ``line_color``.
        - ``marker_symbols``: explicit per-embryo Plotly marker symbol mapping.
        - ``group_colors``: mapping from embryo name to Plotly color.
        - ``condition_colors``: mapping from condition name to Plotly color,
          matched case-insensitively (for example ``{"control-l": "blue"}``).
          ``group_colors`` takes precedence when both mappings match.
        - ``line_width`` / ``line_dash``: trajectory line styling.
        - ``title``, ``template``, ``hovermode``, and ``show_legend``: layout controls.

        Values supplied in ``features`` take precedence over the corresponding
        legacy keyword arguments.
        """
        import plotly.graph_objects as go
        
        allowed_features = {
            "show_trajectories",
            "x_range",
            "y_range",
            "do_not_include",
            "markers",
            "marker_symbols",
            "group_colors",
            "condition_colors",
            "line_width",
            "line_dash",
            "title",
            "template",
            "hovermode",
            "show_legend",
        }
        features = {} if features is None else features
        if not isinstance(features, dict):
            raise TypeError("features must be a dictionary.")
        unknown_features = set(features) - allowed_features
        if unknown_features:
            raise ValueError(f"Unknown plot feature(s): {sorted(unknown_features)}")

        show_trajectories = features.get("show_trajectories", showTrajectories)
        if not isinstance(show_trajectories, bool):
            raise TypeError("features['show_trajectories'] must be a bool.")
        x_range = features.get("x_range", x_range)
        y_range = features.get("y_range", y_range)
        do_not_include = features.get("do_not_include", do_not_include)
        group_colors = features.get("group_colors", {})
        if not isinstance(group_colors, dict):
            raise TypeError("features['group_colors'] must be a dictionary.")
        normalized_group_colors = {}
        for name, color in group_colors.items():
            if not isinstance(name, str):
                raise TypeError("features['group_colors'] keys must be embryo names.")
            normalized_name = name.strip()
            if normalized_name in normalized_group_colors:
                raise ValueError(
                    "features['group_colors'] contains duplicate embryo names "
                    f"after trimming whitespace: {normalized_name!r}"
                )
            normalized_group_colors[normalized_name] = color

        condition_colors = features.get("condition_colors", {})
        if not isinstance(condition_colors, dict):
            raise TypeError("features['condition_colors'] must be a dictionary.")
        normalized_condition_colors = {}
        for name, color in condition_colors.items():
            if not isinstance(name, str):
                raise TypeError(
                    "features['condition_colors'] keys must be condition names."
                )
            normalized_name = name.strip().casefold()
            if normalized_name in normalized_condition_colors:
                raise ValueError(
                    "features['condition_colors'] contains duplicate condition "
                    f"names after normalization: {normalized_name!r}"
                )
            normalized_condition_colors[normalized_name] = color

        markers = features.get("markers", True)
        allowed_marker_options = {
            "size",
            "symbol",
            "opacity",
            "line_width",
            "line_color",
        }
        if isinstance(markers, bool):
            show_markers = markers
            marker_options = {}
            marker_symbols = {}
        elif isinstance(markers, dict):
            show_markers = True
            marker_option_keys = set(markers) & allowed_marker_options
            if marker_option_keys and not set(markers) <= allowed_marker_options:
                raise ValueError(
                    "Do not mix global marker options with embryo names in "
                    "features['markers']; use features['marker_symbols'] for "
                    "per-embryo symbols alongside global marker styling."
                )
            if set(markers) <= allowed_marker_options:
                marker_options = markers
                marker_symbols = {}
            else:
                marker_options = {}
                marker_symbols = markers
        else:
            raise TypeError(
                "features['markers'] must be a bool or dictionary of marker "
                "options or per-embryo symbols."
            )

        explicit_marker_symbols = features.get("marker_symbols", {})
        if not isinstance(explicit_marker_symbols, dict):
            raise TypeError("features['marker_symbols'] must be a dictionary.")
        if set(marker_symbols) & set(explicit_marker_symbols):
            raise ValueError(
                "Provide each embryo's marker symbol in either 'markers' or "
                "'marker_symbols', not both."
            )
        marker_symbols = {**marker_symbols, **explicit_marker_symbols}
        normalized_marker_symbols = {}
        for name, symbol in marker_symbols.items():
            if not isinstance(name, str):
                raise TypeError(
                    "Per-embryo marker symbol keys must be embryo names."
                )
            normalized_name = name.strip()
            if normalized_name in normalized_marker_symbols:
                raise ValueError(
                    "Per-embryo marker symbols contain duplicate embryo names "
                    f"after trimming whitespace: {normalized_name!r}"
                )
            if not isinstance(symbol, str):
                raise TypeError(
                    f"Marker symbol for {name!r} must be a Plotly symbol string."
                )
            normalized_marker_symbols[normalized_name] = symbol

        line_width = features.get("line_width", 2)
        if (
            not isinstance(line_width, (int, float, np.number))
            or not np.isfinite(line_width)
            or line_width < 0
        ):
            raise ValueError("features['line_width'] must be a non-negative number.")
        line_dash = features.get("line_dash", "solid")

        if PCA_key is not None and PCA_Space is not None:
            raise ValueError("Provide either PCA_key or PCA_Space, not both.")
        pca_key = PCA_key if PCA_key is not None else PCA_Space
        embryo_name, condition, properties = self._normalize_pca_key(pca_key)
        key = self._key(embryo_name, condition, properties)
        res = self.results[key]
        coords, metadata = self._filter_plot_rows(
            res["transformed"], res["row_order"], do_not_include
        )

        def resolve_axis_range(
            values: np.ndarray,
            requested_range: Optional[Tuple[float, float]],
            axis_name: str,
        ) -> Tuple[float, float]:
            if requested_range is not None:
                if not isinstance(requested_range, (tuple, list, np.ndarray)):
                    raise ValueError(
                        f"{axis_name}_range must contain two finite values "
                        "in increasing order."
                    )
                bounds = np.asarray(requested_range)
                if (
                    bounds.shape != (2,)
                    or not np.issubdtype(bounds.dtype, np.number)
                    or not np.all(np.isfinite(bounds))
                    or bounds[0] >= bounds[1]
                ):
                    raise ValueError(
                        f"{axis_name}_range must contain two finite values "
                        "in increasing order."
                    )
                return float(bounds[0]), float(bounds[1])

            lower = float(np.min(values))
            upper = float(np.max(values))
            padding = (upper - lower) * 0.05 if upper > lower else 0.1
            return lower - padding, upper + padding

        x_axis_range = resolve_axis_range(coords[:, 0], x_range, "x")
        y_axis_range = resolve_axis_range(coords[:, 1], y_range, "y")
        
        unique_names = sorted({grp.embryo_name for grp, _ in metadata})
        fig = go.Figure()
        
        for embryo_val in unique_names:
            indices = [i for i, (grp, p) in enumerate(metadata) if grp.embryo_name == embryo_val]
            indices = sorted(indices, key=lambda i: metadata[i][0].timestep)
            
            if not indices:
                continue
                
            traj = coords[indices]
            timesteps = [metadata[i][0].timestep for i in indices]
            hover_text = [f"Embryo: {embryo_val}<br>Time: {t}<br>Cond: {metadata[indices[0]][0].condition}" for t in timesteps]
            mode = "+".join(
                mode_part
                for enabled, mode_part in (
                    (show_trajectories, "lines"),
                    (show_markers, "markers"),
                )
                if enabled
            ) or "none"
            trace = {
                "x": traj[:, 0],
                "y": traj[:, 1],
                "mode": mode,
                "name": embryo_val,
                "text": hover_text,
                "hoverinfo": "text",
                "showlegend": features.get("show_legend", True),
            }
            if show_trajectories:
                trace["line"] = {
                    "width": line_width,
                    "dash": line_dash,
                }
            if show_markers:
                marker = {
                    "size": 6,
                    "symbol": "circle",
                    "opacity": 1,
                    "line": {"width": 1, "color": "White"},
                }
                marker.update(
                    {
                        "size": marker_options.get("size", marker["size"]),
                        "symbol": normalized_marker_symbols.get(
                            embryo_val.strip(),
                            marker_options.get("symbol", marker["symbol"]),
                        ),
                        "opacity": marker_options.get("opacity", marker["opacity"]),
                        "line": {
                            "width": marker_options.get(
                                "line_width", marker["line"]["width"]
                            ),
                            "color": marker_options.get(
                                "line_color", marker["line"]["color"]
                            ),
                        },
                    }
                )
                trace["marker"] = marker
            normalized_embryo_name = embryo_val.strip()
            condition_name = metadata[indices[0]][0].condition.strip().casefold()
            color = normalized_condition_colors.get(condition_name)
            color = normalized_group_colors.get(normalized_embryo_name, color)
            if color is not None:
                if show_trajectories:
                    trace["line"]["color"] = color
                if show_markers:
                    trace["marker"]["color"] = color
            
            fig.add_trace(go.Scatter(**trace))
            
        fig.update_layout(
            title=features.get(
                "title",
                (
                    "Interactive PCA Trajectories (Click legend to toggle)"
                    if show_trajectories
                    else "Interactive PCA Points"
                ),
            ),
            xaxis_title=f"PC1 ({res['explained_variance_ratio'][0]*100:.2f}%)",
            yaxis_title=f"PC2 ({res['explained_variance_ratio'][1]*100:.2f}%)",
            hovermode=features.get("hovermode", "closest"),
            template=features.get("template", "plotly_white"),
            xaxis=dict(range=x_axis_range, autorange=False),
            yaxis=dict(range=y_axis_range, autorange=False),
        )
        return fig
   
    def plot_trajectory_gradient_vectors(
        self,
        PCA_key: Optional[Any] = None,
        plot_criteria: Optional[Dict[str, Any]] = None,
        show_trajectories: bool = True,
        do_not_include: Optional[Dict[str, Any]] = None,
        figsize: Tuple[int, int] = (12, 9),
        scale: float = 1.0,
        annotate: bool = False,
        embryo_name: Optional[List[str]] = None,
        condition: Optional[str] = None,
        properties: Optional[List[int]] = None,
    ) -> plt.Figure:
        """Plot embryo paths through PCA space as a sequence of gradient vectors.

        Each embryo/property trajectory is rendered as a series of consecutive arrows.
        The color of the arrows and points transitions smoothly from a light tint at 
        t=0 to the embryo's full base color at the final timestep.
        """
        import matplotlib.cm as cm
        from matplotlib.colors import LinearSegmentedColormap

        if PCA_key is not None:
            embryo_name, condition, properties = CloudGroupPCA._normalize_pca_key(PCA_key)
        elif plot_criteria is not None:
            embryo_name = plot_criteria.get("embryo_name", embryo_name)
            condition = plot_criteria.get("condition", condition)
            properties = plot_criteria.get("properties", properties)

        if properties is None:
            properties = [0, 1]

        key = self._key(embryo_name, condition, properties)
        if key not in self.results:
            raise ValueError(f"Run PCA for {key} first.")

        res = self.results[key]
        coords = res["transformed"]

        if coords.shape[1] < 2:
            raise ValueError("Need at least two PCA components to plot trajectory vectors.")

        row_order = res["row_order"]
        coords, metadata = self._filter_plot_rows(coords, row_order, do_not_include)

        unique_names = sorted({grp.embryo_name for grp, _ in metadata})
        unique_props = sorted({prop for _, prop in metadata})

        cmap = cm.get_cmap('tab20')
        embryo_color_map = {
            name: cmap(i % 20) for i, name in enumerate(unique_names)
        }

        fig, ax = plt.subplots(figsize=figsize)

        for embryo_name_val in unique_names:
            base_color = embryo_color_map[embryo_name_val]
            light_color = (
                base_color[0] + (1 - base_color[0]) * 0.7,
                base_color[1] + (1 - base_color[1]) * 0.7,
                base_color[2] + (1 - base_color[2]) * 0.7,
                base_color[3]
            )
            embryo_cmap = LinearSegmentedColormap.from_list('embryo_grad', [light_color, base_color])

            for prop in unique_props:
                sample_indices = [
                    i for i, (grp, p) in enumerate(metadata)
                    if grp.embryo_name == embryo_name_val and p == prop
                ]

                sample_indices = sorted(
                    sample_indices,
                    key=lambda i: metadata[i][0].timestep,
                )

                if len(sample_indices) < 2:
                    continue

                traj_coords = coords[sample_indices]
                num_points = len(traj_coords)
                denominator = num_points - 1 if num_points > 1 else 1

                if show_trajectories:
                    for j in range(num_points - 1):
                        start = traj_coords[j]
                        end = traj_coords[j + 1]
                        vec = (end - start) * scale
                        
                        color_fraction = j / denominator
                        vector_color = embryo_cmap(color_fraction)

                        ax.quiver(
                            start[0], start[1], vec[0], vec[1],
                            angles="xy", scale_units="xy", scale=1,
                            color=vector_color, alpha=0.8,
                            width=0.006, headwidth=5, headlength=7,
                            headaxislength=6, zorder=4,
                        )

                point_colors = [embryo_cmap(i / denominator) for i in range(num_points)]
                
                ax.scatter(
                    traj_coords[:, 0], traj_coords[:, 1],
                    color=point_colors, marker="o", s=60,
                    edgecolors="white", linewidth=0.8, zorder=5,
                )

                ax.scatter(
                    traj_coords[0, 0], traj_coords[0, 1],
                    color=embryo_cmap(0.0), marker="x", s=100,
                    linewidth=2, zorder=6,
                )

                if annotate:
                    ax.text(
                        traj_coords[-1, 0] + 0.01, traj_coords[-1, 1] + 0.01,
                        f"{embryo_name_val} H{prop}", fontsize=8,
                        ha="left", va="bottom", color="black", zorder=6,
                    )

        evr = res["explained_variance_ratio"]
        ax.set_xlabel(f"PC1 ({evr[0]*100:.2f}%)", fontsize=14)
        ax.set_ylabel(f"PC2 ({evr[1]*100:.2f}%)", fontsize=14)

        ax.set_title(
            f"Gradient Trajectory Vectors in PCA Space: {embryo_name or ''} "
            f"{condition or ''} H{properties}\n"
            "Color intensifies over time for each embryo"
        )

        ax.axhline(0, color="black", linewidth=0.8, alpha=0.3)
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.3)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        plt.close(fig)
        return fig
    
    def plot_sequential_pca_trajectories(
        self,
        PCA_key: Optional[Any] = None,
        n_components: int = 2,
    ):
        """Create one Plotly figure of sequentially reoriented PCA trajectories.

        PCA is computed for ``PCA_key`` when no cached result exists or when the
        cached result has a different number of components.
        """
        import plotly.graph_objects as go
        from plotly.colors import qualitative

        if not isinstance(n_components, int) or n_components < 2:
            raise ValueError("n_components must be an integer greater than or equal to 2.")

        embryo_name, condition, properties = self._normalize_pca_key(PCA_key)
        key = self._key(embryo_name, condition, properties)

        cached_result = self.results.get(key)
        cached_components = (
            cached_result["transformed"].shape[1]
            if cached_result is not None
            else None
        )

        if cached_result is None or cached_components != n_components:
            result = self.run(
                PCA_key=PCA_key,
                n_components=n_components,
            )
        else:
            result = cached_result

        coords = np.asarray(result["transformed"])
        row_order = result["row_order"]

        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(
                "At least two PCA components are required for trajectory plotting."
            )

        trajectories = sorted(
            {
                (
                    self.manager.groups[group_id].embryo_name,
                    prop,
                )
                for group_id, prop in row_order
            },
            key=lambda item: (item[0], item[1]),
        )

        trajectory_data = []
        all_points = []

        for embryo_value, prop in trajectories:
            indices = [
                index
                for index, (group_id, row_prop) in enumerate(row_order)
                if (
                    self.manager.groups[group_id].embryo_name == embryo_value
                    and row_prop == prop
                )
            ]
            indices.sort(
                key=lambda index: self.manager.groups[
                    row_order[index][0]
                ].timestep
            )

            if len(indices) < 2:
                continue

            trajectory = coords[indices, :2]
            original_vectors = np.diff(trajectory, axis=0)

            current_point = trajectory[0].copy()
            path_points = [current_point.copy()]
            timesteps = [
                self.manager.groups[row_order[indices[0]][0]].timestep
            ]

            for vector_index, vector in enumerate(original_vectors):
                if vector_index == 0:
                    adjusted_vector = vector.copy()
                else:
                    cumulative_vector = original_vectors[
                        : vector_index + 1
                    ].sum(axis=0)
                    cumulative_norm = np.linalg.norm(cumulative_vector)

                    if cumulative_norm <= np.finfo(float).eps:
                        continue

                    adjusted_vector = (
                        cumulative_vector / cumulative_norm
                    ) * np.linalg.norm(vector)

                if np.linalg.norm(adjusted_vector) <= np.finfo(float).eps:
                    continue

                current_point = current_point + adjusted_vector
                path_points.append(current_point.copy())

                endpoint_index = indices[vector_index + 1]
                timesteps.append(
                    self.manager.groups[
                        row_order[endpoint_index][0]
                    ].timestep
                )

            path_points = np.asarray(path_points)

            if len(path_points) < 2:
                continue

            trajectory_data.append(
                {
                    "name": f"{embryo_value} | H{prop}",
                    "legendgroup": f"{embryo_value}-H{prop}",
                    "points": path_points,
                    "timesteps": timesteps,
                }
            )
            all_points.extend(path_points)

        if not trajectory_data:
            raise ValueError(
                "No trajectories with at least two timesteps were found."
            )

        fig = go.Figure()
        trajectory_colors = qualitative.Dark24

        for trajectory_index, trajectory in enumerate(trajectory_data):
            points = trajectory["points"]
            color = trajectory_colors[
                trajectory_index % len(trajectory_colors)
            ]

            endpoint_labels = [""] * len(points)
            endpoint_labels[-1] = trajectory["name"]

            fig.add_trace(
                go.Scatter(
                    x=points[:, 0],
                    y=points[:, 1],
                    customdata=np.asarray(
                        trajectory["timesteps"]
                    ).reshape(-1, 1),
                    mode="lines+markers+text",
                    text=endpoint_labels,
                    textposition="top right",
                    name=trajectory["name"],
                    legendgroup=trajectory["legendgroup"],
                    marker={"size": 6, "color": color},
                    line={"width": 1.8, "color": color},
                    hovertemplate=(
                        f"{trajectory['name']}<br>"
                        "Timestep: %{customdata[0]}<br>"
                        "PC1: %{x:.3f}<br>"
                        "PC2: %{y:.3f}<extra></extra>"
                    ),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[points[0, 0]],
                    y=[points[0, 1]],
                    mode="markers",
                    name=f"Start: {trajectory['name']}",
                    legendgroup=trajectory["legendgroup"],
                    showlegend=False,
                    marker={
                        "symbol": "x",
                        "size": 10,
                        "color": "black",
                    },
                    hoverinfo="skip",
                )
            )

        all_points_array = np.asarray(all_points)
        x_min, y_min = all_points_array.min(axis=0)
        x_max, y_max = all_points_array.max(axis=0)

        x_padding = (x_max - x_min) * 0.2 if x_max > x_min else 0.1
        y_padding = (y_max - y_min) * 0.2 if y_max > y_min else 0.1

        variance_ratio = np.asarray(
            result["explained_variance_ratio"]
        )

        conditions = condition
        if isinstance(conditions, str):
            conditions = [conditions]
        if conditions is None:
            conditions = ["All"]

        property_label = (
            "All"
            if properties is None
            else ", ".join(f"H{prop}" for prop in properties)
        )

        title = (
            f"{' + '.join(conditions)} | "
            f"{property_label} | "
            f"{n_components} PCA components"
        )

        fig.update_layout(
            title=title,
            height=800,
            width=1100,
            template="plotly_white",
            xaxis={
                "title": f"PC1 ({variance_ratio[0]:.1%})",
                "range": [
                    x_min - x_padding,
                    x_max + x_padding,
                ],
                "autorange": False,
                "showgrid": True,
                "zeroline": True,
            },
            yaxis={
                "title": f"PC2 ({variance_ratio[1]:.1%})",
                "range": [
                    y_min - y_padding,
                    y_max + y_padding,
                ],
                "autorange": False,
                "showgrid": True,
                "zeroline": True,
                "scaleanchor": "x",
                "scaleratio": 1,
            },
            legend={
                "orientation": "h",
                "y": -0.15,
                "x": 0.5,
                "xanchor": "center",
            },
            margin={"l": 70, "r": 30, "t": 70, "b": 120},
        )

        return fig
# Plotting utilities for point clouds, diagrams, and landscapes.
class TDAPlotter:
    """Plotting utilities for point clouds, diagrams, and landscapes."""

    def __init__(self, manager: "CloudDataManager") -> None:
        self.manager = manager

    def plot_persistence_landscapes(self, results: Dict[Any, Any]) -> plt.Figure:
        """Plot landscapes using coordinate ranges stored in metadata."""
        metadata = results.get("metadata", {})
        n_layers = metadata.get("n_layers", 5)
        resolution = metadata.get("resolution", 100)
        ranges = metadata.get("ranges", {})
        dimensions = sorted(k for k in results if isinstance(k, int))

        if not dimensions:
            raise ValueError("No dimension data found in landscape results.")

        fig, axes = plt.subplots(
            len(dimensions),
            1,
            figsize=(10, 5 * len(dimensions)),
        )
        if len(dimensions) == 1:
            axes = [axes]

        for ax, dim in zip(axes, dimensions):
            vector = np.asarray(results[dim])
            layers = vector.reshape(n_layers, resolution)
            x_min, x_max = ranges.get(dim, (0.0, 1.0))
            x_vals = np.linspace(x_min, x_max, resolution)

            for layer_idx in range(n_layers):
                if np.any(layers[layer_idx] != 0):
                    ax.plot(
                        x_vals,
                        layers[layer_idx],
                        label=f"Layer {layer_idx + 1}",
                        linewidth=1.5,
                    )

            ax.set_title(f"Persistence Landscape - Dimension {dim}")
            ax.set_xlabel("Birth/Death Coordinate")
            ax.set_ylabel("Amplitude")
            ax.legend(loc="upper right", fontsize="small")
            ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.close(fig)
        return fig

    def compare_tda_summaries(
        self,
        group_ids: List[GroupID],
        add_label: bool = False,
        shared_axes: bool = False,
    ) -> plt.Figure:
        """Nx4 summary comparing a list of groups."""
        if not group_ids:
            raise ValueError("group_ids list cannot be empty.")

        groups = [self.manager.get_group(gid) for gid in group_ids]
        num_groups = len(groups)

        all_scaled = [
            StandardScaler().fit_transform(group.original_data.values)
            for group in groups
        ]
        all_pts = np.vstack(all_scaled)
        xlim = (all_pts[:, 0].min() - 0.1, all_pts[:, 0].max() + 0.1)
        ylim = (all_pts[:, 1].min() - 0.1, all_pts[:, 1].max() + 0.1)
        zlim = (all_pts[:, 2].min() - 0.1, all_pts[:, 2].max() + 0.1)

        max_persistence_val, global_max_land = self._summary_ranges(groups)
        fig = plt.figure(figsize=(20, 5 * num_groups))

        for row, group in enumerate(groups):
            scaled_pts = StandardScaler().fit_transform(group.original_data.values)
            self._plot_summary_row(
                fig=fig,
                group=group,
                scaled_pts=scaled_pts,
                row=row,
                nrows=num_groups,
                xlim=xlim,
                ylim=ylim,
                zlim=zlim,
                max_persistence_val=max_persistence_val,
                global_max_land=global_max_land,
                add_label=add_label,
                shared_axes=shared_axes,
            )

        plt.tight_layout()
        plt.close(fig)
        return fig

    def _summary_ranges(
        self,
        groups: List[CloudGroup],
    ) -> Tuple[float, Dict[int, float]]:
        dimensions = self.manager._tda_params["dimensions"]
        max_persistence_val = 0.0
        global_max_land = {dim: 0.0 for dim in dimensions}

        for group in groups:
            for _dim, (birth, death) in group.diagrams["dgm"]:
                if np.isfinite(death):
                    max_persistence_val = max(max_persistence_val, birth, death)

            for dim in dimensions:
                if dim in group.landscapes:
                    global_max_land[dim] = max(
                        global_max_land[dim],
                        float(np.max(group.landscapes[dim])),
                    )

        if max_persistence_val == 0.0:
            max_persistence_val = 1.0
        return max_persistence_val, global_max_land

    def _plot_summary_row(
        self,
        fig: plt.Figure,
        group: CloudGroup,
        scaled_pts: np.ndarray,
        row: int,
        nrows: int,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        zlim: Tuple[float, float],
        max_persistence_val: float,
        global_max_land: Dict[int, float],
        add_label: bool,
        shared_axes: bool = False,
    ) -> None:
        base = row * 4

        ax3d = fig.add_subplot(nrows, 4, base + 1, projection="3d")
        ax3d.scatter(scaled_pts[:, 0], scaled_pts[:, 1], scaled_pts[:, 2], s=10, alpha=0.6)
        ax3d.view_init(elev=20, azim=45)
        ax3d.set_xlim(xlim)
        ax3d.set_ylim(ylim)
        ax3d.set_zlim(zlim)
        label_segment = f" [{group.condition}]" if add_label else ""
        ax3d.set_title(
            f"3D View: {group.embryo_name}{label_segment} (t={group.timestep})",
            fontweight="bold",
            fontsize=12,
        )

        axdg = fig.add_subplot(nrows, 4, base + 2)
        all_diag = []
        for dim in self.manager._tda_params["dimensions"]:
            diag_dim = [p for d, p in group.diagrams["dgm"] if d == dim]
            all_diag.extend((dim, p) for p in diag_dim)
        if all_diag:
            gudhi.plot_persistence_diagram(all_diag, axes=axdg)
            for poly in axdg.findobj(plt.Polygon):
                poly.set_visible(False)
            if shared_axes:
                axdg.set_xlim(-0.05 * max_persistence_val, max_persistence_val * 1.1)
                axdg.set_ylim(-0.05 * max_persistence_val, max_persistence_val * 1.1)
            axdg.spines["left"].set_position(("outward", 10))
            axdg.spines["bottom"].set_position(("outward", 10))
            axdg.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
            has_infinity = any(np.isinf(p[1]) for _, p in all_diag)
            if has_infinity:
                yticks = axdg.get_yticks()
                yticklabels = [f'{y:.0f}' for y in yticks]
                if yticklabels:
                    yticklabels[-1] = r'+$\infty$'
                axdg.set_yticklabels(yticklabels)
            else:
                axdg.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        axdg.set_title("Persistence Diagrams", fontweight="bold", fontsize=12)

        self._plot_landscape_panel(
            fig=fig,
            group=group,
            dim=0,
            subplot_index=base + 3,
            nrows=nrows,
            max_persistence_val=max_persistence_val,
            y_max=global_max_land.get(0, 0.0),
        )
        self._plot_landscape_panel(
            fig=fig,
            group=group,
            dim=1,
            subplot_index=base + 4,
            nrows=nrows,
            max_persistence_val=max_persistence_val,
            y_max=global_max_land.get(1, 0.0),
        )

    def _plot_landscape_panel(
        self,
        fig: plt.Figure,
        group: CloudGroup,
        dim: int,
        subplot_index: int,
        nrows: int,
        max_persistence_val: float,
        y_max: float,
    ) -> None:
        ax = fig.add_subplot(nrows, 4, subplot_index)
        if dim in group.landscapes:
            n_layers = self.manager._tda_params["n_layers"]
            resolution = self.manager._tda_params["resolution"]
            land = np.asarray(group.landscapes[dim]).reshape(n_layers, -1)
            x_vals = np.linspace(0, max_persistence_val, resolution)

            for layer_idx in range(len(land)):
                ax.plot(x_vals, land[layer_idx], label=f"L{layer_idx + 1}")

            ax.set_ylim(-0.02, max(y_max * 1.1, 0.02))
            ax.set_xlim(-0.05, max_persistence_val * 1.1)
            ax.spines["left"].set_position(("outward", 10))
            ax.spines["bottom"].set_position(("outward", 10))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

        ax.legend(fontsize=12)
        ax.set_title(f"H{dim} Landscapes", fontweight="bold", fontsize=12)
# Main manager for TDA data loading, storage, and public API delegation.
class CloudDataManager:
    """Main manager for TDA data loading, storage, and public API delegation."""

    VALID_CONDITIONS = {"control-L","control-M", "epha2a-SM", "epha2a-WM", "epha2a-WM"}
    REQUIRED_COLUMNS = [
        "centroid_x",
        "centroid_y",
        "centroid_z",
        "timepoint",
    ]

    def __init__(self) -> None:
        # All cloud point data is stored in CloudGroup objects, indexed by (embryo_name, timestep)
        self.groups: Dict[GroupID, CloudGroup] = {}
        # 80% of embryos are used for training, 
        self.embryo_groups: Dict[str, List[GroupID]] = {}
        # 20% of embryos are used for testing, stored separately
        self.testing_groups: Dict[GroupID, CloudGroup] = {}
        # TDA parameters used for computing diagrams, landscapes, and PCA. This is set when TDA is run.
        self._tda_params: Optional[Dict[str, Any]] = None

        self.TDA = CloudGroupTDA(self)
        self.Matrix = LandscapeMatrixBuilder(self)
        self.PCA = CloudGroupPCA(self)
        self.Plotter = TDAPlotter(self)

    @classmethod
    def load(cls, filepath: str) -> "CloudDataManager":
        with open(filepath, "rb") as f:
            return pickle.load(f)
    
    def save(self, filepath: str) -> None:
        """Pickle this manager and its stored data to ``filepath``."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    def list_group_ids(self) -> List[GroupID]:
        return sorted(self.groups.keys(), key=lambda x: (x[0], x[1]))

    def get_group(self, group_id: GroupID) -> CloudGroup:
        return self.groups[group_id]

    def get_trajectory(self, embryo_name: str) -> List[CloudGroup]:
        ids = sorted(self.embryo_groups.get(embryo_name, []), key=lambda x: x[1])
        return [self.groups[gid] for gid in ids if gid in self.groups]

    def split_embryos(
        self,
        test_fraction: float = 0.2,
        strategy: Optional[Callable[[str, List[GroupID]], bool]] = None,
        random_state: Optional[int] = None,
    ) -> Tuple[List[str], List[str]]:
        embryo_names = sorted(self.embryo_groups.keys())
        if strategy is None:
            rng = random.Random(random_state)
            n_test = max(1, int(round(len(embryo_names) * test_fraction)))
            test_set = set(rng.sample(embryo_names, min(n_test, len(embryo_names))))

            def strategy_fn(name: str, gids: List[GroupID]) -> bool:
                return name in test_set

            strategy = strategy_fn

        train_embryos: List[str] = []
        test_embryos: List[str] = []

        for embryo_name, gids in self.embryo_groups.items():
            if strategy(embryo_name, list(gids)):
                test_embryos.append(embryo_name)
                for gid in gids:
                    group = self.groups.pop(gid, None)
                    if group is not None:
                        self.testing_groups[gid] = group
            else:
                train_embryos.append(embryo_name)

        return train_embryos, test_embryos

    def load_groups_from_directory(self, directory_path: str) -> List[GroupID]:
        """Load valid CSV files from a directory."""
        loaded_ids = []
        files = sorted(os.listdir(directory_path))
        for filename in files:
            lower = filename.lower()
            if not lower.endswith(".csv"):
                continue
            if not re.search(r"(^|[^a-z0-9])(epha2a|control)([^a-z0-9]|$)", lower):
                continue
            # loaded_ids.append(self.load_group_from_file(os.path.join(directory_path, filename)))
            self.load_group_from_file(os.path.join(directory_path, filename))
        return loaded_ids

    def load_group_from_file(self, filepath: str) -> GroupID:
        """Parse filename metadata and load one point cloud."""
        filename = os.path.basename(filepath)
        lower = filename.lower()

        condition_match = re.search(
            r"(?:^|[^a-z0-9])(epha2a-(?:sx|wl|wm|sm)|control-[lm])(?:[^a-z0-9]|$)",
            lower,
        )
        if not condition_match:
            raise ValueError(f"Could not infer condition from filename: {filename}")
        condition = condition_match.group(1)

        embryo_name = "unknown"
        embryo_name_match = re.search(r"name=([^\.]+)", filename, re.IGNORECASE)
        if embryo_name_match:
            raw_name = embryo_name_match.group(1)
            embryo_name = re.split(r"_t=|\.|$", raw_name)[0]

        # time step is in side csv file
        # t_match = re.search(r"(?:^|[ _])t\s*=\s*(\d+)(?=[_\.\s]|$)", filename)
        # timestep = int(t_match.group(1)) if t_match else 0

        df = pd.read_csv(filepath)
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {filename}: {missing}")

        df = df.dropna(subset=self.REQUIRED_COLUMNS).reset_index(drop=True)
        df = df.rename(
            columns={
                "centroid_x": "x",
                "centroid_y": "y",
                "centroid_z": "z",
                "timepoint": "t"
            }
        )
        group_ids = []
        subgroups = {name: group for name, group in df.groupby('t')}
        # we are only taking every 8th timestep to reduce the number of groups, but you can change this as needed
        # for timestep in range(0,60,6):
        for timestep in subgroups.keys():  #load all timesteps
            gid = (embryo_name, timestep)
            if gid in self.groups:
                raise ValueError(
                    f"Duplicate group ID {gid} from file {filename}. "
                    "Current GroupID is only (embryo_name, timestep); include condition/replicate "
                    "in the sample name or refactor GroupID."
                )
            self.groups[gid] = CloudGroup(
                group_id=gid,
                embryo_name=embryo_name,
                condition=condition,
                timestep=timestep,
                original_data=subgroups[timestep].copy(deep=True),
            )
            self.embryo_groups.setdefault(embryo_name, []).append(gid)
            group_ids.append(gid)
        return group_ids

    def get_landscape_range(self, group_id: GroupID, dim: int) -> Tuple[float, float]:
        """Return the stored coordinate range for one group's landscape dimension."""
        group = self.get_group(group_id)
        ranges = group.landscapes.get("metadata", {}).get("ranges", {})
        return tuple(ranges.get(dim, (0.0, 1.0)))

    def collect_landscape_ranges(
        self,
        row_order: Optional[List[Tuple[GroupID, int]]] = None,
    ) -> Dict[Tuple[GroupID, int], Tuple[float, float]]:
        """Collect stored landscape coordinate ranges for matrix rows."""
        if row_order is None:
            row_order = [
                (gid, dim)
                for gid in self.list_group_ids()
                for dim in self.groups[gid].landscapes.get("metadata", {}).get("ranges", {})
            ]
        return {(gid, dim): self.get_landscape_range(gid, dim) for gid, dim in row_order}
    
    def distance_table_from_control_mean(self, PCA_key) -> pd.DataFrame:
        """
        Parameter: PCA_key: Can be a dict {"embryo_name": [...], "condition": "...", "properties": [...]}
        or tuple (embryo_name, condition, properties). Overrides other parameters.
        This funciton will compute the distance from the mean control vector for each timestep for the specified group and return a table with embryo_name, timestep, and distance.
        """
        embryo_name, condition, properties = CloudGroupPCA._normalize_pca_key(PCA_key)
        # build a matrix for the control group and compute the mean vector for each timestep
        controlLandscapeVectors = self.Matrix.build_landscape_matrix( PCA_key=(["control2", "control3","control4"], "control", properties) )
        meanControlVector = pd.DataFrame(controlLandscapeVectors["matrix"], index=[ timestep for id in controlLandscapeVectors["row_order"] for timestep in [id[0][1]] ]).groupby(level=0).mean()
        
        # build a matrix for the specified group and compute the distance from the mean control vector for each timestep
        results = self.Matrix.build_landscape_matrix( PCA_key=(embryo_name, condition, properties) )
        conditionEntriesPerTimeStep = pd.DataFrame(results["matrix"], index=[ timestep for id in results["row_order"] for timestep in [id[0][1]] ])
        
        # place holder for mean distance from mean vector for each timestep
        distance_from_mean_vector = np.zeros((len(conditionEntriesPerTimeStep),1))
        # get the timestep for each row in the results
        row_timestep = [ timestep for id in results["row_order"] for timestep in [id[0][1]] ]

        for index, timestep in enumerate(row_timestep):
            distance_from_mean_vector[index] = np.linalg.norm(conditionEntriesPerTimeStep.iloc[index] - meanControlVector.loc[timestep])

        embryo_names = [ id[0][0] for id in results["row_order"] ]
        time_steps = [ id[0][1] for id in results["row_order"] ]
        table = pd.DataFrame( {"embryo_name": embryo_names, "timestep": time_steps, "dist": distance_from_mean_vector.reshape(-1) } )

        return table
    # User interface methods that delegate to PCA, TDA, Matrix, and Plotter classes.
    def compute_pca(self, *args, **kwargs): return self.PCA.run(*args, **kwargs)
    
    def plot_trajectories(self, *args, **kwargs): return self.PCA.plot_trajectories(*args, **kwargs)
    
    def plot_trajectory_vectors(self, *args, **kwargs): return self.PCA.plot_trajectory_vectors(*args, **kwargs)
    
    def plot_trajectory_gradient_vectors(self, *args, **kwargs): return self.PCA.plot_trajectory_gradient_vectors(*args, **kwargs)

    def slider_plot_trajectory_vectors(self, *args, **kwargs): return self.PCA.slider_plot_trajectory_vectors(*args, **kwargs)
    def plot_sequential_pca_trajectories(self,*args, **kwargs,): return self.PCA.plot_sequential_pca_trajectories(*args,**kwargs,)
    
    def compute_persistence_diagram(self, *args, **kwargs): return self.TDA.compute_persistence_diagram(*args, **kwargs)

    def compute_persistence_landscapes(self, *args, **kwargs): return self.TDA.compute_persistence_landscapes(*args, **kwargs)

    def compute_all_persistence_diagrams(self, *args, **kwargs): return self.TDA.compute_all_persistence_diagrams(*args, **kwargs)

    def compute_all_persistence_landscapes(self, *args, **kwargs): return self.TDA.compute_all_persistence_landscapes(*args, **kwargs)

    def build_landscape_matrix(self, *args, **kwargs): return self.Matrix.build_landscape_matrix(*args, **kwargs)

    def plot_persistence_landscapes(self, *args, **kwargs): return self.Plotter.plot_persistence_landscapes(*args, **kwargs)

    def compare_tda_summaries(self, *args, **kwargs): return self.Plotter.compare_tda_summaries(*args, **kwargs)

    def plot_tda_summary_for_group(self, *args, **kwargs): return self.Plotter.plot_tda_summary_for_group(*args, **kwargs)

    def plot_small_multiples(self, *args, **kwargs): return self.PCA.plot_small_multiples(*args, **kwargs)

    def plot_condition_averages(self, *args, **kwargs): return self.PCA.plot_condition_averages(*args, **kwargs)

    def plot_spline_smoothing(self, *args, **kwargs): return self.PCA.plot_spline_smoothing(*args, **kwargs)

    def plot_interactive_plotly(self, *args, **kwargs): return self.PCA.plot_interactive_plotly(*args, **kwargs)

    def plot_averaged_vectors(self, *args, **kwargs): return self.PCA.plot_averaged_vectors(*args, **kwargs)

    def plot_sequential_vectors(self, *args, **kwargs): return self.PCA.plot_sequential_vectors(*args, **kwargs)