import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class PCADimensionalityReduction:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean = None
        self.eigenvectors = None
        self.eigenvalues = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        n_samples, n_features = X.shape

        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        covariance_matrix = np.cov(X_centered, rowvar=False)

        # 修复：避免数值误差产生负特征值/复数
        eigenvalues, eigenvectors = np.linalg.eig(np.real(covariance_matrix))
        eigenvalues = np.abs(eigenvalues)

        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]

        total_variance = np.sum(self.eigenvalues)
        self.explained_variance_ratio_ = self.eigenvalues / total_variance

        if self.n_components is None:
            self.n_components = n_features

        return self

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.eigenvectors[:, :self.n_components]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class SVDimensionalityReduction:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.U = None
        self.S = None
        self.Vt = None
        self.singular_values_ = None

    def fit(self, X):
        U, S, Vt = np.linalg.svd(X, full_matrices=False)

        self.U = U
        self.S = S
        self.Vt = Vt
        self.singular_values_ = S

        if self.n_components is None:
            self.n_components = len(S)

        return self

    def transform(self, X):
        V = self.Vt[:self.n_components, :].T
        return X @ V

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def generate_music_data():
    np.random.seed(42)
    n_samples = 1000
    n_features = 30

    X = np.random.randn(n_samples, n_features)

    X[:, 0:5] *= 3
    X[:, 5:10] *= 2
    X[:, 10:15] *= 1.5
    X[:, 15:20] *= 1.2
    X[:, 20:25] *= 1.0
    X[:, 25:30] *= 0.8

    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    y = np.array([genres[i % 10] for i in range(n_samples)])

    feature_names = [
        'zero_crossing_rate', 'spectral_centroid', 'spectral_bandwidth',
        'spectral_rolloff', 'spectral_entropy', 'mfcc1', 'mfcc2', 'mfcc3',
        'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10',
        'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'chroma_stft1',
        'chroma_stft2', 'chroma_stft3', 'chroma_stft4', 'chroma_stft5',
        'chroma_stft6', 'chroma_stft7', 'chroma_stft8', 'chroma_stft9',
        'loudness'
    ]

    os.makedirs('./GTZAN', exist_ok=True)
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y
    df.to_csv('./GTZAN/features_30_sec.csv', index=False)
    print("GTZAN音乐数据集模拟生成完成")

    return X, y, feature_names


def plot_variance_explained(explained_var, algorithm_name):
    cumulative_var = np.cumsum(explained_var)

    plt.figure(figsize=(12, 5))

    # 修复1：X轴和Y轴长度保持一致（均为10）
    top_k = 10
    x = range(1, top_k + 1)
    y_val = explained_var[:top_k]
    plt.subplot(1, 2, 1)
    plt.bar(x, y_val, color='skyblue')
    plt.title(f'{algorithm_name}前10个成分方差解释率')
    plt.xlabel('成分编号')
    plt.ylabel('方差解释率')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, marker='o', color='orange')
    plt.axhline(y=0.95, color='red', linestyle='--', label='95%阈值')
    plt.title(f'{algorithm_name}累计方差解释率')
    plt.xlabel('成分数量')
    plt.ylabel('累计方差解释率')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'./{algorithm_name}_variance.png')
    plt.close()


# 修复2：重构函数，支持PCA和SVD正确计算重构误差
def evaluate_reduction(X, X_reduced, model=None):
    if model is None:
        reconstruction = X_reduced
    elif isinstance(model, PCADimensionalityReduction):
        # PCA重构公式
        reconstruction = X_reduced @ model.eigenvectors[:, :X_reduced.shape[1]].T + model.mean
    elif isinstance(model, SVDimensionalityReduction):
        # SVD重构公式
        reconstruction = X_reduced @ model.Vt[:X_reduced.shape[1], :]
    else:
        reconstruction = X_reduced

    reconstruction_error = np.mean((X - reconstruction) ** 2)

    explained_var = np.var(X_reduced, axis=0)
    total_var = np.sum(np.var(X, axis=0))
    cumulative_var = np.sum(explained_var) / total_var if total_var > 0 else 0

    return {
        '原始维度': X.shape[1],
        '降维后维度': X_reduced.shape[1],
        '重构误差': reconstruction_error,
        '累计方差解释率': cumulative_var
    }


if __name__ == '__main__':
    print("===== 机器学习实验七：PCA与SVD降维 =====")

    print("\n1. 生成模拟GTZAN音乐数据集...")
    X, y, feature_names = generate_music_data()

    print(f"数据集大小: {X.shape}")
    print(f"特征数量: {len(feature_names)}")
    print(f"音乐流派类别: {np.unique(y)}")

    print("\n2. 数据标准化...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n===== PCA降维 =====")
    pca = PCADimensionalityReduction()
    pca.fit(X_scaled)

    explained_var_pca = pca.explained_variance_ratio_
    cumulative_var_pca = np.cumsum(explained_var_pca)

    k_pca = np.argmax(cumulative_var_pca >= 0.95) + 1
    print(f"PCA最优降维维度（累计方差≥95%）: {k_pca}")
    print(f"累计方差解释率: {cumulative_var_pca[k_pca - 1]:.4f}")
    print(f"前10个主成分方差解释率: {explained_var_pca[:10].round(4)}")

    pca_opt = PCADimensionalityReduction(n_components=k_pca)
    X_pca = pca_opt.fit_transform(X_scaled)
    pca_result = evaluate_reduction(X_scaled, X_pca, pca_opt)
    print(f"PCA重构误差: {pca_result['重构误差']:.6f}")

    plot_variance_explained(explained_var_pca, "PCA")

    print("\n===== SVD降维 =====")
    svd = SVDimensionalityReduction()
    svd.fit(X_scaled)

    singular_values = svd.singular_values_
    singular_contribution = singular_values / np.sum(singular_values)
    cumulative_singular = np.cumsum(singular_contribution)

    k_svd = np.argmax(cumulative_singular >= 0.95) + 1
    print(f"SVD最优降维维度（累计奇异值贡献≥95%）: {k_svd}")
    print(f"累计奇异值贡献率: {cumulative_singular[k_svd - 1]:.4f}")
    print(f"前10个奇异值: {singular_values[:10].round(2)}")

    svd_opt = SVDimensionalityReduction(n_components=k_svd)
    X_svd = svd_opt.fit_transform(X_scaled)
    # 修复：传入SVD模型计算正确重构误差
    svd_result = evaluate_reduction(X_scaled, X_svd, svd_opt)
    print(f"SVD重构误差: {svd_result['重构误差']:.6f}")

    plot_variance_explained(singular_contribution, "SVD")

    print("\n===== sklearn对比 =====")

    print("\nsklearn PCA:")
    sk_pca = PCA(n_components=0.95, random_state=42)
    X_sk_pca = sk_pca.fit_transform(X_scaled)
    print(f"sklearn PCA降维后维度: {X_sk_pca.shape[1]}")
    print(f"累计方差解释率: {np.sum(sk_pca.explained_variance_ratio_):.4f}")

    print("\nsklearn TruncatedSVD:")
    # 修复3：TruncatedSVD不支持自动选维度，必须传整数
    sk_svd = TruncatedSVD(n_components=k_svd, random_state=42)
    X_sk_svd = sk_svd.fit_transform(X_scaled)
    print(f"sklearn SVD降维后维度: {X_sk_svd.shape[1]}")
    print(f"累计奇异值贡献率: {np.sum(sk_svd.explained_variance_ratio_):.4f}")

    print("\n===== 降维结果对比 =====")
    results = pd.DataFrame({
        '算法': ['PCA', 'SVD', 'sklearn PCA', 'sklearn SVD'],
        '原始维度': [X.shape[1], X.shape[1], X.shape[1], X.shape[1]],
        '降维后维度': [k_pca, k_svd, X_sk_pca.shape[1], X_sk_svd.shape[1]],
        '累计方差解释率': [
            cumulative_var_pca[k_pca - 1],
            cumulative_singular[k_svd - 1],
            np.sum(sk_pca.explained_variance_ratio_),
            np.sum(sk_svd.explained_variance_ratio_)
        ]
    })
    print(results.round(4))

    print("\n===== 奇异值分布 =====")
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(singular_values[:15]) + 1), singular_values[:15], color='purple')
    plt.title('SVD前15个奇异值分布')
    plt.xlabel('奇异值编号')
    plt.ylabel('奇异值大小')
    plt.savefig('./singular_values.png')
    plt.close()
    print("奇异值分布图已保存")

    print("\n===== 算法对比分析 =====")
    comparison = pd.DataFrame({
        '对比维度': ['数学原理', '计算效率', '信息保留', '适用场景', '是否需要中心化', '核心指标'],
        'PCA': ['协方差矩阵特征值分解', '高', '高', '数据压缩、去噪、可视化', '是', '方差解释率'],
        'SVD': ['奇异值分解', '较高', '高', '推荐系统、NLP、图像', '否', '奇异值贡献率']
    })
    print(comparison)

    print("\n===== 实验完成 =====")
    print("生成的文件:")
    print("- PCA_variance.png: PCA方差解释率图")
    print("- SVD_variance.png: SVD奇异值贡献率图")
    print("- singular_values.png: 奇异值分布图")
    print("- GTZAN/features_30_sec.csv: 模拟音乐数据集")