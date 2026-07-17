import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Assuming tda_module.py is in the same directory or accessible
from tda_module import CloudDataManager, GroupID, CloudGroup


def plot_kmeans_elbow(points: np.ndarray, max_k: int = 10) -> plt.Figure:
    """
    Calculates and plots the inertia for a range of k values to find the 'elbow'.

    This helps in choosing an optimal number of clusters for K-Means.

    Args:
        points: The point cloud data as a NumPy array (n_samples, n_features).
        max_k: The maximum number of clusters to test.

    Returns:
        A matplotlib Figure object showing the elbow plot.
    """
    # It's good practice to scale data before clustering
    scaler = StandardScaler()
    scaled_points = scaler.fit_transform(points)

    inertias = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_points)
        inertias.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_range, inertias, marker='o', linestyle='--')
    ax.set_title('Elbow Method for Optimal k')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia (Within-cluster sum of squares)')
    ax.grid(True, linestyle=':')
    plt.close(fig)
    return fig


def analyze_point_cloud_pca_and_kmeans(
    manager: CloudDataManager,
    group_id: GroupID,
    n_clusters: int
) -> plt.Figure:
    """
    Performs PCA and KMeans clustering on a single point cloud and plots the results.

    Args:
        manager: The CloudDataManager instance containing the data.
        group_id: The GroupID of the point cloud to analyze.
        n_clusters: The number of clusters (k) to use for KMeans.

    Returns:
        A matplotlib Figure object with two side-by-side plots:
        1. A 2D scatter plot of the point cloud projected onto its first two principal components.
        2. A 3D scatter plot of the original point cloud, colored by K-Means cluster.
    """
    try:
        group = manager.get_group(group_id)
    except KeyError:
        raise ValueError(f"GroupID {group_id} not found in the manager.")

    points = group.original_data.values

    # --- 1. Scale the data for both PCA and KMeans ---
    scaler = StandardScaler()
    scaled_points = scaler.fit_transform(points)

    # --- 2. PCA Analysis (2 components) ---
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(scaled_points)
    evr = pca.explained_variance_ratio_

    # --- 3. KMeans Clustering ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_points)

    # --- 4. Plotting ---
    fig = plt.figure(figsize=(20, 9))
    fig.suptitle(f"Analysis for Embryo: {group.embryo_name} | Timestep: {group.timestep}", fontsize=16)

    # --- Plot 1: PCA Result ---
    ax1 = fig.add_subplot(1, 2, 1)
    scatter1 = ax1.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis', alpha=0.8)
    ax1.set_title(f'PCA of Point Cloud (Colored by k={n_clusters} Clusters)')
    ax1.set_xlabel(f'Principal Component 1 ({evr[0]*100:.2f}%)')
    ax1.set_ylabel(f'Principal Component 2 ({evr[1]*100:.2f}%)')
    ax1.grid(True, linestyle=':')
    legend1 = ax1.legend(*scatter1.legend_elements(), title="Clusters")
    ax1.add_artist(legend1)

    # --- Plot 2: 3D K-Means Result ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=cluster_labels, cmap='viridis', s=15, alpha=0.7)
    ax2.set_title(f'3D Point Cloud with K-Means Clusters (k={n_clusters})')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_zlabel('Z Coordinate')
    ax2.view_init(elev=20, azim=45)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.close(fig)
    return fig


if __name__ == '__main__':
    # This is an example of how you would use these functions.
    print("Creating a dummy CloudDataManager for demonstration.")
    manager = CloudDataManager()
    dummy_gid = ('dummy_embryo', 1)
    # Create two distinct blobs for a clear clustering example
    blob1 = np.random.randn(75, 3) + np.array([2, 2, 2])
    blob2 = np.random.randn(75, 3) + np.array([-2, -2, -2])
    dummy_points = np.vstack([blob1, blob2])
    dummy_df = pd.DataFrame(dummy_points, columns=['x', 'y', 'z'])
    manager.groups[dummy_gid] = CloudGroup(group_id=dummy_gid, embryo_name='dummy_embryo', condition='control', timestep=1, original_data=dummy_df)
    
    print(f"\nStep 1: Generating Elbow Plot for {dummy_gid} to help choose k.")
    elbow_fig = plot_kmeans_elbow(manager.get_group(dummy_gid).original_data.values)
    elbow_fig.savefig('elbow_plot_example.png')
    print("   -> Elbow plot saved to 'elbow_plot_example.png'. Inspect it to choose k.")

    chosen_k = 2
    print(f"\nStep 2: Running analysis with chosen k={chosen_k}.")
    analysis_fig = analyze_point_cloud_pca_and_kmeans(manager=manager, group_id=dummy_gid, n_clusters=chosen_k)
    analysis_fig.savefig('pca_kmeans_analysis_example.png')
    print(f"   -> Analysis plot saved to 'pca_kmeans_analysis_example.png'.")