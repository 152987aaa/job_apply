import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_iris
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class KMeansClustering:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None
    
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def fit(self, X):
        n_samples, n_features = X.shape
        
        np.random.seed(42)
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[indices].astype(np.float64)
        
        for _ in range(self.max_iter):
            self.labels = self._assign_clusters(X)
            new_centroids = self._update_centroids(X)
            
            centroid_shift = np.sum(np.abs(new_centroids - self.centroids))
            self.centroids = new_centroids
            
            if centroid_shift < self.tol:
                break
        
        return self.labels
    
    def _assign_clusters(self, X):
        labels = []
        for x in X:
            distances = [self._euclidean_distance(x, centroid) for centroid in self.centroids]
            labels.append(np.argmin(distances))
        return np.array(labels)
    
    def _update_centroids(self, X):
        new_centroids = []
        for i in range(self.n_clusters):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                new_centroids.append(self.centroids[i])
        return np.array(new_centroids)
    
    def predict(self, X):
        return self._assign_clusters(X)

class DBSCANClustering:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
    
    def _region_query(self, X, point_idx):
        neighbors = []
        for i, x in enumerate(X):
            distance = np.sqrt(np.sum((X[point_idx] - x) ** 2))
            if distance <= self.eps:
                neighbors.append(i)
        return neighbors
    
    def fit(self, X):
        n_samples = X.shape[0]
        self.labels = np.full(n_samples, -1)
        cluster_id = 0
        
        for i in range(n_samples):
            if self.labels[i] != -1:
                continue
            
            neighbors = self._region_query(X, i)
            
            if len(neighbors) < self.min_samples:
                self.labels[i] = -1
                continue
            
            self.labels[i] = cluster_id
            
            j = 0
            while j < len(neighbors):
                neighbor_idx = neighbors[j]
                
                if self.labels[neighbor_idx] == -1:
                    self.labels[neighbor_idx] = cluster_id
                    
                    new_neighbors = self._region_query(X, neighbor_idx)
                    if len(new_neighbors) >= self.min_samples:
                        neighbors.extend(new_neighbors)
                
                j += 1
            
            cluster_id += 1
        
        return self.labels

class HierarchicalClustering:
    def __init__(self, n_clusters=3, linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels = None
        self.X = None
    
    def _single_linkage_distance(self, cluster1, cluster2):
        min_dist = float('inf')
        for i in cluster1:
            for j in cluster2:
                dist = np.sqrt(np.sum((self.X[i] - self.X[j]) ** 2))
                if dist < min_dist:
                    min_dist = dist
        return min_dist
    
    def _complete_linkage_distance(self, cluster1, cluster2):
        max_dist = -float('inf')
        for i in cluster1:
            for j in cluster2:
                dist = np.sqrt(np.sum((self.X[i] - self.X[j]) ** 2))
                if dist > max_dist:
                    max_dist = dist
        return max_dist
    
    def _average_linkage_distance(self, cluster1, cluster2):
        total_dist = 0.0
        count = 0
        for i in cluster1:
            for j in cluster2:
                total_dist += np.sqrt(np.sum((self.X[i] - self.X[j]) ** 2))
                count += 1
        return total_dist / count if count > 0 else 0
    
    def _ward_linkage_distance(self, cluster1, cluster2):
        mean1 = self.X[cluster1].mean(axis=0)
        mean2 = self.X[cluster2].mean(axis=0)
        dist = np.sqrt(np.sum((mean1 - mean2) ** 2))
        return dist
    
    def fit(self, X):
        self.X = X.astype(np.float64)
        n_samples = X.shape[0]
        
        clusters = [[i] for i in range(n_samples)]
        
        while len(clusters) > self.n_clusters:
            min_dist = float('inf')
            merge_i, merge_j = -1, -1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    if self.linkage == 'single':
                        dist = self._single_linkage_distance(clusters[i], clusters[j])
                    elif self.linkage == 'complete':
                        dist = self._complete_linkage_distance(clusters[i], clusters[j])
                    elif self.linkage == 'average':
                        dist = self._average_linkage_distance(clusters[i], clusters[j])
                    else:
                        dist = self._ward_linkage_distance(clusters[i], clusters[j])
                    
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j
            
            if merge_i != -1 and merge_j != -1:
                clusters[merge_i].extend(clusters[merge_j])
                del clusters[merge_j]
        
        self.labels = np.zeros(n_samples, dtype=int)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                self.labels[sample_idx] = cluster_idx
        
        return self.labels

def calculate_sse(X, labels, centroids=None):
    if centroids is None:
        centroids = []
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label != -1:
                centroids.append(X[labels == label].mean(axis=0))
        centroids = np.array(centroids)
    
    sse = 0.0
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        if label != -1:
            cluster_points = X[labels == label]
            centroid = centroids[i] if i < len(centroids) else cluster_points.mean(axis=0)
            sse += np.sum((cluster_points - centroid) ** 2)
    return sse

def calculate_dunn_index(X, labels):
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]
    
    if len(unique_labels) < 2:
        return np.nan
    
    min_inter_cluster = float('inf')
    max_intra_cluster = -float('inf')
    
    for i, label1 in enumerate(unique_labels):
        cluster1 = X[labels == label1]
        max_dist = 0
        for j in range(len(cluster1)):
            for k in range(j + 1, len(cluster1)):
                dist = np.sqrt(np.sum((cluster1[j] - cluster1[k]) ** 2))
                if dist > max_dist:
                    max_dist = dist
        if max_dist > max_intra_cluster:
            max_intra_cluster = max_dist
        
        for label2 in unique_labels[i + 1:]:
            cluster2 = X[labels == label2]
            min_dist = float('inf')
            for point1 in cluster1:
                for point2 in cluster2:
                    dist = np.sqrt(np.sum((point1 - point2) ** 2))
                    if dist < min_dist:
                        min_dist = dist
            if min_dist < min_inter_cluster:
                min_inter_cluster = min_dist
    
    if max_intra_cluster == 0:
        return np.nan
    
    return min_inter_cluster / max_intra_cluster

def calculate_sbd(X, labels):
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]
    
    if len(unique_labels) < 2:
        return np.nan
    
    total_sbd = 0.0
    n_clusters = len(unique_labels)
    
    for i, label1 in enumerate(unique_labels):
        cluster1 = X[labels == label1]
        centroid1 = cluster1.mean(axis=0)
        
        for j in range(i + 1, n_clusters):
            label2 = unique_labels[j]
            cluster2 = X[labels == label2]
            centroid2 = cluster2.mean(axis=0)
            
            inter_dist = np.sqrt(np.sum((centroid1 - centroid2) ** 2))
            
            intra_dist1 = np.mean([np.sqrt(np.sum((x - centroid1) ** 2)) for x in cluster1])
            intra_dist2 = np.mean([np.sqrt(np.sum((x - centroid2) ** 2)) for x in cluster2])
            
            sbd = inter_dist / (intra_dist1 + intra_dist2)
            total_sbd += sbd
    
    return total_sbd / (n_clusters * (n_clusters - 1) / 2)

def evaluate_clustering(X, labels, algorithm_name, centroids=None):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])
    
    if n_clusters < 2:
        print(f"{algorithm_name}: 无法评估（聚类数少于2）")
        return None
    
    silhouette = silhouette_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    dbi = davies_bouldin_score(X, labels)
    sse = calculate_sse(X, labels, centroids)
    dunn = calculate_dunn_index(X, labels)
    sbd = calculate_sbd(X, labels)
    
    print(f"\n{algorithm_name} 评估结果:")
    print(f"  聚类数量: {n_clusters}")
    print(f"  SSE (误差平方和): {sse:.4f}")
    print(f"  SC (轮廓系数): {silhouette:.4f}")
    print(f"  CH (Calinski-Harabasz指数): {calinski:.4f}")
    print(f"  DBI (Davies-Bouldin指数): {dbi:.4f}")
    print(f"  DI (Dunn指数): {dunn:.4f}")
    print(f"  SBD (简化有界距离): {sbd:.4f}")
    
    return {
        '算法': algorithm_name,
        '聚类数': n_clusters,
        'SSE': sse,
        'SC': silhouette,
        'CH': calinski,
        'DBI': dbi,
        'DI': dunn,
        'SBD': sbd
    }

if __name__ == '__main__':
    print("===== 机器学习实验六：聚类算法 =====")
    
    print("\n1. 加载数据集...")
    iris = load_iris()
    X = iris.data
    y_true = iris.target
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"数据集大小: {X.shape}")
    print(f"特征数量: {X.shape[1]}")
    print(f"真实类别数: {len(np.unique(y_true))}")
    
    results = []
    
    print("\n===== K-Means 聚类 =====")
    kmeans = KMeansClustering(n_clusters=3, max_iter=100)
    kmeans_labels = kmeans.fit(X_scaled)
    result = evaluate_clustering(X_scaled, kmeans_labels, "K-Means", kmeans.centroids)
    if result:
        results.append(result)
    
    print("\n===== DBSCAN 聚类 =====")
    dbscan = DBSCANClustering(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit(X_scaled)
    result = evaluate_clustering(X_scaled, dbscan_labels, "DBSCAN")
    if result:
        results.append(result)
    
    print("\n===== 层次聚类（Ward） =====")
    hierarchical = HierarchicalClustering(n_clusters=3, linkage='ward')
    hierarchical_labels = hierarchical.fit(X_scaled)
    result = evaluate_clustering(X_scaled, hierarchical_labels, "层次聚类")
    if result:
        results.append(result)
    
    print("\n===== sklearn 对比 =====")
    
    print("\nsklearn K-Means:")
    sk_kmeans = KMeans(n_clusters=3, random_state=42)
    sk_kmeans_labels = sk_kmeans.fit_predict(X_scaled)
    result = evaluate_clustering(X_scaled, sk_kmeans_labels, "sklearn K-Means", sk_kmeans.cluster_centers_)
    if result:
        results.append(result)
    
    print("\nsklearn DBSCAN:")
    sk_dbscan = DBSCAN(eps=0.5, min_samples=5)
    sk_dbscan_labels = sk_dbscan.fit_predict(X_scaled)
    result = evaluate_clustering(X_scaled, sk_dbscan_labels, "sklearn DBSCAN")
    if result:
        results.append(result)
    
    print("\nsklearn 层次聚类:")
    sk_hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
    sk_hierarchical_labels = sk_hierarchical.fit_predict(X_scaled)
    result = evaluate_clustering(X_scaled, sk_hierarchical_labels, "sklearn 层次聚类")
    if result:
        results.append(result)
    
    print("\n===== 聚类结果对比 =====")
    df_results = pd.DataFrame(results)
    print(df_results.round(4))
    
    print("\n===== 指标说明 =====")
    print("""
    SSE (误差平方和): 越小越好，表示簇内样本越紧密
    SC (轮廓系数): 取值[-1,1]，越接近1越好
    CH (Calinski-Harabasz指数): 越大越好
    DBI (Davies-Bouldin指数): 越小越好
    DI (Dunn指数): 越大越好
    SBD (简化有界距离): 越大越好
    """)