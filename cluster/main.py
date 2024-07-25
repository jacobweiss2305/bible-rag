from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

def normalize_X(embeddings: list) -> np.ndarray:
    """Normalize the embeddings to have a mean of 0 and a standard deviation of 1."""
    X = np.array(embeddings)
    return StandardScaler().fit_transform(X)

def find_optimal_clusters(X, max_k):
    inertias = []
    silhouette_scores = []
    K = range(5, max_k+1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    
    # Plot elbow curve
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    
    # Plot silhouette scores
    plt.subplot(1, 2, 2)
    plt.plot(K, silhouette_scores, 'rx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method')
    
    plt.tight_layout()
    plt.show()
    
    # Find optimal k
    optimal_k = K[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_k}")
    
    return optimal_k

def cluster_kmeans_embeddings(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels

def cluster_dbscan_embeddings(X_normalized: np.ndarray, min_samples: int = 5) -> np.ndarray:
    """Cluster the embeddings using DBSCAN."""
    dbscan = DBSCAN(min_samples=min_samples).fit(X_normalized)
    return dbscan.fit_predict(X_normalized)