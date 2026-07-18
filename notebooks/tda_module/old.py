import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gudhi
import gudhi.representations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

GroupID = Tuple[str, int]

@dataclass
class CloudGroup:
    """Container for one point cloud corresponding to a single (name, timestep)."""
    group_id: GroupID
    name: str
    label: str
    timestep: int
    original_data: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)
    diagrams: Dict[int, Optional[np.ndarray]] = field(default_factory=dict)
    landscapes: Dict[int, Optional[np.ndarray]] = field(default_factory=dict)

class CloudGroupPCA:
    """PCA-based analysis for TDA landscapes."""
    VALID_KINDS = {"h0", "h1", "combined"}

    def __init__(self, manager: "CloudDataManager") -> None:
        self.manager = manager
        self.results = {}

    def run(self,
        n_components: int = 2,
        label: str = None,
        name: str = None,
        properties: List[int] = None 
    ):
        if properties is None: 
            properties=[0, 1]
        """Standardizes and runs PCA on the feature matrices."""
        hom_props = self.manager.build_landscape_matrix( label= label, names = name, properties=properties)
      
        scaled_matrix = StandardScaler().fit_transform(hom_props["matrix"])
        # Inside run_pca in tda_module.py
        pca = PCA(n_components=min(n_components, scaled_matrix.shape[0], scaled_matrix.shape[1]), random_state=42)
        transformed = pca.fit_transform(scaled_matrix)
        self.results[(name, label, tuple(properties))] = {
            "transformed": transformed,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "row_order": hom_props["row_order"],
            "components":pca.components_, 
        }


    def plot_trajectories(
        self,
        name: str = None,
        label: str = None,
        properties: List[int] = [0, 1],
        show_trajectories: bool = True,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plots PCA trajectories considering name, label, time, and homology property.
        Connects points for each sample name and homology property across timesteps.
        """

        key = (name, label, tuple(properties))
        if key not in self.results:
            raise ValueError(f"Run PCA for {key} first.")
        
        res = self.results[key]
        coords = res["transformed"]
        row_order = res["row_order"]
        
        if coords.shape[1] < 2:
            raise ValueError("plot_trajectories requires at least 2 PCA components.")

        # Metadata: list of (group, prop) for each row
        metadata = [(self.manager.groups[gid], prop) for gid, prop in row_order]
        
        # Unique names, labels, props
        unique_names = sorted({m[0].name for m in metadata})
        unique_labels = sorted({m[0].label for m in metadata})
        unique_props = sorted({m[1] for m in metadata})
        
        # Colors for labels
        label_color_map = {
            'control': 'blue',
            'epha2a': 'red',
        }
        
        fig = plt.figure(figsize=figsize, dpi=120)
        
        if show_trajectories:
            # For each name and prop, connect timesteps
            for name_val in unique_names:
                for prop in unique_props:
                    # Find indices for this name and prop, sorted by time
                    sample_indices = [
                        i for i, (grp, p) in enumerate(metadata)
                        if grp.name == name_val and p == prop
                    ]
                    sample_indices = sorted(sample_indices, key=lambda i: metadata[i][0].timestep)
                    
                    if len(sample_indices) > 1:
                        traj_coords = coords[sample_indices]
                        lbl = metadata[sample_indices[0]][0].label
                        color = label_color_map[lbl]
                        plt.plot(traj_coords[:, 0], traj_coords[:, 1], 
                                 color=color, alpha=0.3, linestyle='-', linewidth=2)
                        # Arrow for direction
                        plt.arrow(traj_coords[-2, 0], traj_coords[-2, 1], 
                                  (traj_coords[-1, 0] - traj_coords[-2, 0])*0.5, 
                                  (traj_coords[-1, 1] - traj_coords[-2, 1])*0.5,
                                  shape='full', lw=0, length_includes_head=True, 
                                  head_width=0.02, color=color, alpha=0.5)
        
        # Plot points by label
        for lbl in unique_labels:
            idx = [i for i, (grp, p) in enumerate(metadata) if grp.label == lbl]
            if idx:
                color = label_color_map[lbl]
                plt.scatter(coords[idx, 0], coords[idx, 1], 
                            color=color, label=f"{lbl}", 
                            s=200, edgecolors='white', zorder=5)
        
        # Second scatter: circles, bigger markers, by homology
        for prop in unique_props:
            idx = [i for i, (grp, p) in enumerate(metadata) if p == prop]
            if idx:
                color = 'skyblue' if prop == 1 else 'green'
                plt.scatter(coords[idx, 0], coords[idx, 1], 
                            color=color, marker='o', s=100, zorder=10, label=f"H{prop}")
        
        # Text labels for timesteps
        for i, (grp, p) in enumerate(metadata):
            plt.text(coords[i, 0], coords[i, 1], str(grp.timestep), 
                     color='black', fontsize=8,fontweight="bold", ha='center', va='center', zorder=15)
        
        # # Annotate if sparse, with offset to avoid overlap
        # if len(coords) < 50:
        #     for i, (grp, p) in enumerate(metadata):
        #         offset_x = 5 + (i % 4) * 8  # Vary x offset
        #         offset_y = 5 + (i % 3) * 6  # Vary y offset
        #         plt.annotate(f"t={grp.timestep}, H{p}", (coords[i, 0], coords[i, 1]), 
        #                      fontsize=8, alpha=0.7, xytext=(offset_x, offset_y), textcoords='offset points')
        
        evr = res["explained_variance_ratio"]
        plt.xlabel(f"PC1 ({evr[0]*100:.1f}%)")
        plt.ylabel(f"PC2 ({evr[1]*100:.1f}%)")
        plt.title(f"TDA PCA Trajectories: {name or ''} {label or ''} H{properties}\n(Lines connect timesteps for each sample and homology)")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.close(fig)
        return fig

class CloudDataManager:
    """Main manager for TDA data processing and visualization."""
    VALID_LABELS = {"control", "epha2a"}

    def __init__(self):
        self.groups: Dict[GroupID, CloudGroup] = {}
        self._tda_params = None
        self.PCA = CloudGroupPCA(self)

    def list_group_ids(self):
        return sorted(self.groups.keys(), key=lambda x: (x[0], x[1]))

    def get_group(self, group_id: GroupID):
        return self.groups[group_id]

    def get_trajectory(self, name: str) -> List[CloudGroup]:
        """IMPROVEMENT 4: Returns sorted timesteps for a specific sample name."""
        ids = [gid for gid in self.list_group_ids() if gid[0] == name]
        return [self.groups[gid] for gid in ids]
    
    def load_groups_from_directory(self, directory_path: str) -> List[GroupID]:
        """Load all valid CSV files from a directory, ignoring non-matching files."""
        loaded_ids = []
        for filename in os.listdir(directory_path):
            if filename.endswith('.csv') and (filename.startswith('control') or filename.startswith('epha2a')):
                filepath = os.path.join(directory_path, filename)
                gid = self.load_group_from_file(filepath)
                loaded_ids.append(gid)
        return loaded_ids
    
    def load_group_from_file(self, filepath: str):
        """Parses filename and loads point cloud."""
        filename = os.path.basename(filepath)
        label = "control" if "control" in filename.lower() else "epha2a"
        
        # Regex to extract name=... and t=...
        name_match = re.search(r"name=([^.\s]+)", filename)
        name = name_match.group(1) if name_match else "unknown"
        t_match = re.search(r"t=(\d+)", filename)
        t = int(t_match.group(1)) if t_match else 0
        
        # Inside load_group_from_file:
        df = pd.read_csv(filepath).dropna(subset=["Center of the object_0", "Center of the object_1", "Center of the object_2"]).reset_index(drop=True)
        rename_map = {
            "Center of the object_0": "x", 
            "Center of the object_1": "y", 
            "Center of the object_2": "z"
        }
        df = df.rename(columns=rename_map)
        
        gid = (name, t)
        self.groups[gid] = CloudGroup(gid, name, label, t, df[["x", "y", "z"]])
        return gid
    
    def compute_persistence_diagram(self,
        points: np.ndarray,
        max_dimension: int = 1,
        max_edge_length: Optional[float] = np.inf,
        homology_coeff_field: int = 2, # Default to Z/2Z for simplicity and speed 
        min_persistence: float = 0.0
    ) -> np.ndarray:
        """Compute a persistence diagram from a point cloud using a Rips complex."""
        pts = np.asarray(points)
       
        if pts.ndim != 2 or pts.shape[0] == 0:
            raise ValueError("points must be a non-empty 2D array")
        
        results = {
            "metadata": {
                "max_dimension": max_dimension,
                "max_edge_length": max_edge_length,
                "homology_coeff_field": homology_coeff_field,
                "min_persistence": min_persistence
            }
        }
        
        rips = gudhi.RipsComplex(points=pts, max_edge_length=max_edge_length) 
        simplex_tree = rips.create_simplex_tree(max_dimension=max_dimension)
        simplex_tree.compute_persistence(homology_coeff_field=homology_coeff_field)

        persistence = simplex_tree.persistence(homology_coeff_field=homology_coeff_field)
        filtered = [
            (dim, (birth, death))
            for dim, (birth, death) in persistence
            if np.isinf(death) or (death - birth) >= min_persistence
        ]
        results["dgm"] = filtered

        return results

    def compute_persistence_landscapes( self,
        diagram: List[Tuple[int, Tuple[float, float]]],
        max_dimension: int = 1,
        min_persistence: float = 0.0,
        resolution: int = 100,
        n_layers: int = 5
    ) -> Dict[Any, Any]:
        results = {
            "metadata": {
                "n_layers": n_layers,
                "resolution": resolution,
                "min_persistence": min_persistence,
                "max_dimension": max_dimension,
                "ranges": {} 
            }
        }
        for dim in range(max_dimension + 1):
            pts = [
                [birth, death] for d, (birth, death) in diagram 
                if d == dim and (death - birth) >= min_persistence and np.isfinite(death)
            ]
            if not pts:
                results[dim] = np.zeros(n_layers * resolution)
                results["metadata"]["ranges"][dim] = (0.0, 1.0)
                continue

            diagram_array = np.array(pts)
            lsc = gudhi.representations.Landscape(num_landscapes=n_layers, resolution=resolution)
            vector = lsc.fit_transform([diagram_array])
            
            # Store the actual birth/death range used for this dimension
            finite_deaths = [death for _, death in pts if np.isfinite(death)]
            x_min = min(birth for birth, _ in pts)
            x_max = max(finite_deaths) if finite_deaths else max(birth for birth, _ in pts) + 1.0
            if x_min == x_max:
                x_max = x_min + 1.0
            results["metadata"]["ranges"][dim] = (float(x_min), float(x_max))

            results[dim] = vector.flatten()

        return results
        
    def compute_all_persistence_diagrams(
        self,
        max_dimension: int = 2,
        max_edge_length: Optional[float] = np.inf,
        homology_coeff_field: int = 2,
        min_persistence: float = 0.0,
        resolution: int = 500,
        n_layers: int = 5
    ) -> Dict[GroupID, Dict[int, List[Tuple[float, float]]]]:
        """Compute persistence diagrams and landscapes for every loaded CloudGroup."""
        self._tda_params = {
            "max_dimension": max_dimension,
            "max_edge_length": max_edge_length,
            "homology_coeff_field": homology_coeff_field,
            "min_persistence": min_persistence,
            "resolution": resolution,
            "n_layers": n_layers,
            "dimensions": list(range(max_dimension))
        }
        for gid, group in self.groups.items():
            points = group.original_data.values.copy()
            persistence_result = self.compute_persistence_diagram(
                points,
                max_dimension=max_dimension,
                max_edge_length=max_edge_length,
                homology_coeff_field=homology_coeff_field,
                min_persistence=min_persistence
            )
            self.groups[gid].diagrams = persistence_result
        return self
    
    def compute_all_persistence_landscapes(self, 
        max_dimension: int = 2,
        min_persistence: float = 0.0,
        resolution: int = 100,
        n_layers: int = 5
    ) -> Dict[GroupID, Dict[int, np.ndarray]]:
        """Compute landscapes for all groups based on their persistence diagrams."""
        if self._tda_params is None:
            raise ValueError("Run compute_all_persistence_diagrams first to set TDA parameters.")
        
        self._tda_params.update({
            "min_persistence": min_persistence,
            "resolution": resolution,
            "n_layers": n_layers,
            "dimensions": list(range(max_dimension))
        })
        
        for gid, group in self.groups.items():
            diagram = group.diagrams["dgm"]
            # print(f"Computing landscapes for group {gid} with {len(diagram)} features in diagram...")
            landscape_result = self.compute_persistence_landscapes(
                diagram=diagram,
                max_dimension=max_dimension - 1,
                min_persistence=min_persistence,
                resolution=resolution,
                n_layers=n_layers
            )
            self.groups[gid].landscapes = landscape_result
        return self
    
    def plot_persistence_landscapes(self, results: Dict[Any, Any]):
        """
        Plots the landscapes using the coordinate ranges stored in metadata.
        """
        metadata = results.get("metadata", {})
        n_layers = metadata.get("n_layers", 5)
        res = metadata.get("resolution", 100)
        ranges = metadata.get("ranges", {})
        
        # Get sorted dimension keys
        dimensions = sorted([k for k in results.keys() if isinstance(k, int)])
        
        if not dimensions:
            print("No dimension data found in results.")
            return

        fig, axes = plt.subplots(len(dimensions), 1, figsize=(10, 5 * len(dimensions)))
        if len(dimensions) == 1:
            axes = [axes]

        for ax, dim in zip(axes, dimensions):
            vector = results[dim]
            layers = vector.reshape(n_layers, res)
            
            # Retrieve the specific range for this dimension from metadata
            # ranges[dim] is expected to be [min_val, max_val]
            dim_range = ranges.get(dim, [0.0, 1.0])
            x_vals = np.linspace(dim_range[0], dim_range[1], res)
            
            for i in range(n_layers):
                # Only plot layers that aren't entirely zero to keep it clean
                if np.any(layers[i] != 0):
                    ax.plot(x_vals, layers[i], label=f"Layer {i+1}", linewidth=1.5)
            
            ax.set_title(f"Persistence Landscape - Dimension {dim}")
            ax.set_xlabel("Birth/Death Coordinate")
            ax.set_ylabel("Amplitude")
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()
    
    def build_landscape_matrix(
        self,
        names: Optional[List[str]] = None,
        label: Optional[str] = None,
        properties: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Build a matrix of landscapes filtered by name, label, and properties.
        
        Args:
            names: List of sample names to include (e.g., ["stewie", "meg"]). None = all names.
            label: Label to filter by (e.g., "control"). None = all labels.
            properties: List of homology dimensions to use (e.g., [0, 1]). None = all dimensions.
        
        Returns:
            Matrix of shape (sample_size, max_width) containing flattened landscapes.
        """
        # Get all group IDs
        ids = self.list_group_ids()
        
        # Filter by name
        if names is not None:
            ids = [gid for gid in ids if gid[0] in names]
        
        # Filter by label
        if label is not None:
            ids = [gid for gid in ids if self.groups[gid].label == label]
        
        # Determine which properties to use
        if properties is None:
            properties = self._tda_params["dimensions"]
        

        sample_size = len(ids) * len(properties)
        
        if sample_size == 0:
            raise ValueError("No CloudGroups match the given criteria.")
        
        
        # Find max_width across all matching groups and selected properties
        max_width = 0
        for gid in ids:
            group = self.groups[gid]
            for prop in properties:
                if prop in group.landscapes:
                    landscape_vec = group.landscapes[prop]
                    if isinstance(landscape_vec, np.ndarray):
                        max_width = max(max_width, landscape_vec.size)
                    else:
                        max_width = max(max_width, len(landscape_vec))
    
        if max_width == 0:
            raise ValueError("No valid landscape data found for the given properties.")
        
        # Initialize matrix with zeros
        matrix = np.zeros((sample_size, max_width))
        
        # Fill in the landscapes
        row_order = [ (gid, prop) for gid in ids for prop in properties if self.groups[gid].landscapes[prop] is not None]
        for index, (gid, prop) in enumerate(row_order): 
            matrix[index, 0: self.groups[gid].landscapes[prop].shape[0] ] =  self.groups[gid].landscapes[prop]

        results = {"matrix": matrix, "row_order": row_order, "tda_params": self._tda_params}
        return results
    
    def compare_tda_summaries_among_two_groups(
        self,
        group_id1: GroupID,
        group_id2: GroupID,
        add_label: bool = False,
    ):
        """2x4 Summary Plot comparing two groups: 3D view, persistence diagrams, H0 landscapes, H1 landscapes."""
        group1 = self.get_group(group_id1)
        group2 = self.get_group(group_id2)
        groups = [group1, group2]
        
        # Compute shared axis limits
        all_scaled = []
        for group in groups:
            scaled_pts = StandardScaler().fit_transform(group.original_data.values)
            all_scaled.append(scaled_pts)
        all_pts = np.vstack(all_scaled)
        xlim = (all_pts[:, 0].min() - 0.1, all_pts[:, 0].max() + 0.1)
        ylim = (all_pts[:, 1].min() - 0.1, all_pts[:, 1].max() + 0.1)
        zlim = (all_pts[:, 2].min() - 0.1, all_pts[:, 2].max() + 0.1)
        
        # Compute max_finite_death and global max landscapes
        max_finite_death = 0.0
        global_max_land = {}
        for dim in self._tda_params["dimensions"]:
            global_max_land[dim] = 0.0
        for group in groups:
            for dim in self._tda_params["dimensions"]:
                diag_dim = [p for d, p in group.diagrams['dgm'] if d == dim]
                for birth, death in diag_dim:
                    if np.isfinite(death):
                        max_finite_death = max(max_finite_death, death)
                if dim in group.landscapes:
                    global_max_land[dim] = max(global_max_land[dim], np.max(group.landscapes[dim]))
        
        fig = plt.figure(figsize=(20, 10))
        
        for row, group in enumerate(groups):
            scaled_pts = StandardScaler().fit_transform(group.original_data.values)
            
            # 1. 3D Cloud
            ax3d = fig.add_subplot(2, 4, row*4+1, projection='3d')
            ax3d.scatter(scaled_pts[:,0], scaled_pts[:,1], scaled_pts[:,2], s=10, alpha=0.6)
            ax3d.view_init(elev=20, azim=45)
            ax3d.set_xlim(xlim)
            ax3d.set_ylim(ylim)
            ax3d.set_zlim(zlim)
            label_segment = f" [{group.label}]" if add_label else ""
            ax3d.set_title(f"3D View: {group.name}{label_segment} (t={group.timestep})", fontweight='bold', fontsize=12)
            
            # 2. Persistence Diagrams
            axdg = fig.add_subplot(2, 4, row*4+2)
            all_diag = []
            for dim in self._tda_params["dimensions"]:
                diag_dim = [p for d, p in group.diagrams['dgm'] if d == dim]
                all_diag.extend([(dim, p) for p in diag_dim])
            if all_diag:
                gudhi.plot_persistence_diagram(all_diag, axes=axdg)
                for poly in axdg.findobj(plt.Polygon): poly.set_visible(False)
                # axdg.set_xlim(-0.5 , max_finite_death * 1.1)
                # axdg.set_ylim(-0.5 , max_finite_death * 1.1)
                # Shift spines diagonally outward
                axdg.spines['left'].set_position(('outward', 10))
                axdg.spines['bottom'].set_position(('outward', 10))
            axdg.set_title("Persistence Diagrams", fontweight='bold', fontsize=12)
            
            # 3. H0 Landscapes
            axls0 = fig.add_subplot(2, 4, row*4+3)
            if 0 in group.landscapes:
                land = group.landscapes[0].reshape(self._tda_params["n_layers"], -1)
                x_vals = np.linspace(0, max_finite_death, self._tda_params["resolution"])
                for i in range(len(land)):
                    axls0.plot(x_vals, land[i], label=f"L{i+1}")
                axls0.set_ylim(-0.02 * max_finite_death, global_max_land[0] * 1.1)
                axls0.set_xlim(-0.05 * max_finite_death, max_finite_death * 1.1)
                # Shift spines diagonally outward
                # axls0.spines['left'].set_position(('outward', 10))
                # axls0.spines['bottom'].set_position(('outward', 10))
            axls0.legend(fontsize=12)
            axls0.set_title("H0 Landscapes", fontweight='bold', fontsize=12)
            
            # 4. H1 Landscapes
            axls1 = fig.add_subplot(2, 4, row*4+4)
            if 1 in group.landscapes:
                land = group.landscapes[1].reshape(self._tda_params["n_layers"], -1)
                x_vals = np.linspace(0, max_finite_death, self._tda_params["resolution"])
                for i in range(len(land)):
                    axls1.plot(x_vals, land[i], label=f"L{i+1}")
                axls1.set_ylim(-0.02 , global_max_land[1] * 1.1)
                axls1.set_xlim(-0.05 , max_finite_death * 1.1)
                # Shift spines diagonally outward
                axls1.spines['left'].set_position(('outward', 10))
                axls1.spines['bottom'].set_position(('outward', 10))
            axls1.legend(fontsize=12)
            axls1.set_title("H1 Landscapes", fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.close(fig)
        return fig

    def plot_tda_summary_for_group(
        self,
        group_id: GroupID,
        add_label: bool = False,
    ) -> plt.Figure:
        """Single-row 1x4 summary plot for one group: 3D, persistence diagram, H0 and H1 landscapes."""
        group = self.get_group(group_id)

        scaled_pts = StandardScaler().fit_transform(group.original_data.values)
        xlim = (scaled_pts[:, 0].min() - 0.1, scaled_pts[:, 0].max() + 0.1)
        ylim = (scaled_pts[:, 1].min() - 0.1, scaled_pts[:, 1].max() + 0.1)
        zlim = (scaled_pts[:, 2].min() - 0.1, scaled_pts[:, 2].max() + 0.1)

        max_finite_death = 0.0
        global_max_land = {dim: 0.0 for dim in self._tda_params["dimensions"]}
        for dim in self._tda_params["dimensions"]:
            diag_dim = [p for d, p in group.diagrams['dgm'] if d == dim]
            for birth, death in diag_dim:
                if np.isfinite(death):
                    max_finite_death = max(max_finite_death, death)
            if dim in group.landscapes:
                global_max_land[dim] = np.max(group.landscapes[dim])

        fig = plt.figure(figsize=(20, 5))

        ax3d = fig.add_subplot(1, 4, 1, projection='3d')
        ax3d.scatter(scaled_pts[:, 0], scaled_pts[:, 1], scaled_pts[:, 2], s=10, alpha=0.6)
        ax3d.view_init(elev=20, azim=45)
        ax3d.set_xlim(xlim)
        ax3d.set_ylim(ylim)
        ax3d.set_zlim(zlim)
        label_segment = f" [{group.label}]" if add_label else ""
        ax3d.set_title(f"3D View: {group.name}{label_segment} (t={group.timestep})", fontweight='bold', fontsize=12)

        axdg = fig.add_subplot(1, 4, 2)
        all_diag = []
        for dim in self._tda_params["dimensions"]:
            diag_dim = [p for d, p in group.diagrams['dgm'] if d == dim]
            all_diag.extend([(dim, p) for p in diag_dim])
        if all_diag:
            gudhi.plot_persistence_diagram(all_diag, axes=axdg)
            for poly in axdg.findobj(plt.Polygon):
                poly.set_visible(False)
            axdg.set_xlim(-.04 * max_finite_death, max_finite_death * 1.1)
            axdg.set_ylim(-.04 * max_finite_death, max_finite_death * 1.2)
            axdg.spines['left'].set_position(('outward', 10))
            axdg.spines['bottom'].set_position(('outward', 10))
        axdg.set_title("Persistence Diagrams", fontweight='bold', fontsize=12)

        axls0 = fig.add_subplot(1, 4, 3)
        if 0 in group.landscapes:
            land = group.landscapes[0].reshape(self._tda_params["n_layers"], -1)
            x_vals = np.linspace(0, max_finite_death, self._tda_params["resolution"])
            for i in range(len(land)):
                axls0.plot(x_vals, land[i], label=f"L{i+1}")
            axls0.set_ylim(-0.02, global_max_land[0] * 1.1)
            axls0.set_xlim(-0.05, max_finite_death * 1.1)
            axls0.spines['left'].set_position(('outward', 10))
            axls0.spines['bottom'].set_position(('outward', 10))
        axls0.legend(fontsize=12)
        axls0.set_title("H0 Landscapes", fontweight='bold', fontsize=12)

        axls1 = fig.add_subplot(1, 4, 4)
        if 1 in group.landscapes:
            land = group.landscapes[1].reshape(self._tda_params["n_layers"], -1)
            x_vals = np.linspace(0, max_finite_death, self._tda_params["resolution"])
            for i in range(len(land)):
                axls1.plot(x_vals, land[i], label=f"L{i+1}")
            axls1.set_ylim(-0.02, global_max_land[1] * 1.1)
            axls1.set_xlim(-0.05, max_finite_death * 1.1)
            axls1.spines['left'].set_position(('outward', 10))
            axls1.spines['bottom'].set_position(('outward', 10))
        axls1.legend(fontsize=12)
        axls1.set_title("H1 Landscapes", fontweight='bold', fontsize=12)

        plt.tight_layout()
        plt.close(fig)
        return fig
    
    
    