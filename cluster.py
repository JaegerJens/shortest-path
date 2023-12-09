import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt


def plotClusters(locations: np.ndarray, clusters: np.ndarray) -> None:
    """
    plot clustered locations with different colors
    """
    colors = ["#FF0000", "#FFA500", "#FFFF00", "#008000",
              "#00FFFF", "#0000FF", "#800080", "#FFC0CB",
              "#FF69B4", "#FF1493", "#00FF00", "#7CFC00",
              "#00CED1", "#000080", "#8B008B", "#FF00FF"]
    clusterColors = [colors[index] for index in clusters]

    plt.scatter(locations[:, 1], locations[:, 0], c=clusterColors)
    plt.show()


def plotVoroni(clusterCenters: np.ndarray) -> None:
    """
    plot a Voronoi figure of the cluster centers
    """
    vor = Voronoi(clusterCenters)

    fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',
                          line_width=2, line_alpha=0.6, point_size=8)
    plt.plot(fig)
    plt.show()


def separateClusters(locations: np.ndarray, labels: np.ndarray) \
        -> dict[int, np.ndarray]:
    """
    separate locations in own array for every cluster label
    """
    locWithLabels = np.hstack((locations, labels[:, np.newaxis]))
    return {label: [city[[0, 1]] for city in locWithLabels if city[2] == label]
            for label in labels}


def clusterCities(locations: np.ndarray, numClusters: int) \
        -> (np.ndarray, np.ndarray):
    """
    cluster all locations

    return
    - center point for every cluster
    - location arrays separated by cluster label
    """
    kmeans = KMeans(n_clusters=numClusters, n_init="auto")
    kmeans.fit(locations)

    clusteredLocations = separateClusters(locations, kmeans.labels_)
    return (kmeans.cluster_centers_, clusteredLocations)
