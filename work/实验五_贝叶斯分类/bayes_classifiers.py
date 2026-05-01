import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_priors = None
        self.mean = None
        self.var = None
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        
        self.class_priors = np.zeros(n_classes)
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_priors[i] = len(X_c) / len(y)
            self.mean[i] = X_c.mean(axis=0)
            self.var[i] = X_c.var(axis=0) + 1e-9
    
    def _gaussian_pdf(self, x, mean, var):
        exponent = np.exp(-(np.power(x - mean, 2) / (2 * var)))
        return (1 / np.sqrt(2 * np.pi * var)) * exponent
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for i, c in enumerate(self.classes):
                prior = np.log(self.class_priors[i])
                likelihood = np.sum(np.log(self._gaussian_pdf(x, self.mean[i], self.var[i])))
                posterior = prior + likelihood
                posteriors.append(posterior)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)

class SemiNaiveBayesODE:
    def __init__(self):
        self.classes = None
        self.class_priors = None
        self.parents = None
        self.conditional_probs = None
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        
        self.class_priors = np.zeros(n_classes)
        self.parents = np.zeros(n_features, dtype=int)
        self.conditional_probs = {}
        
        for i in range(n_features):
            self.parents[i] = 0 if i > 0 else -1
        
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_priors[i] = len(X_c) / len(y)
            
            for j in range(n_features):
                mean_all = X_c[:, j].mean()
                var_all = X_c[:, j].var() + 1e-9
                self.conditional_probs[(i, j, 'none')] = (mean_all, var_all)
                
                if self.parents[j] != -1:
                    parent_vals = np.unique(X_c[:, self.parents[j]])
                    for pv in parent_vals:
                        mask = X_c[:, self.parents[j]] == pv
                        child_vals = X_c[mask, j]
                        mean = child_vals.mean()
                        var = child_vals.var() + 1e-9
                        self.conditional_probs[(i, j, pv)] = (mean, var)
    
    def _gaussian_pdf(self, x, mean, var):
        exponent = np.exp(-(np.power(x - mean, 2) / (2 * var)))
        return (1 / np.sqrt(2 * np.pi * var)) * exponent
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for i, c in enumerate(self.classes):
                posterior = np.log(self.class_priors[i])
                
                for j in range(len(x)):
                    if self.parents[j] == -1:
                        mean, var = self.conditional_probs[(i, j, 'none')]
                    else:
                        pv = x[self.parents[j]]
                        if (i, j, pv) not in self.conditional_probs:
                            pv = np.round(pv, 1)
                            if (i, j, pv) not in self.conditional_probs:
                                if (i, j, 'none') in self.conditional_probs:
                                    mean, var = self.conditional_probs[(i, j, 'none')]
                                else:
                                    X_c = np.array([x for _, x in enumerate(self.class_priors)])
                                    mean = X_c.mean() if len(X_c) > 0 else 0
                                    var = X_c.var() + 1e-9 if len(X_c) > 0 else 1e-9
                            else:
                                mean, var = self.conditional_probs[(i, j, pv)]
                        else:
                            mean, var = self.conditional_probs[(i, j, pv)]
                    
                    posterior += np.log(self._gaussian_pdf(x[j], mean, var))
                
                posteriors.append(posterior)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)

class BayesianNetworkClassifier:
    def __init__(self, structure=None):
        self.classes = None
        self.class_priors = None
        self.structure = structure
        self.conditional_probs = None
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        
        if self.structure is None:
            self.structure = {i: [] for i in range(n_features)}
            for i in range(1, n_features):
                self.structure[i] = [i-1]
        
        self.class_priors = np.zeros(n_classes)
        self.conditional_probs = {}
        
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_priors[i] = len(X_c) / len(y)
            
            for j in range(n_features):
                parents = self.structure[j]
                if not parents:
                    mean = X_c[:, j].mean()
                    var = X_c[:, j].var() + 1e-9
                    self.conditional_probs[(i, j, 'none')] = (mean, var)
                else:
                    parent_combinations = np.unique(X_c[:, parents], axis=0)
                    for pc in parent_combinations:
                        mask = np.all(X_c[:, parents] == pc, axis=1)
                        child_vals = X_c[mask, j]
                        mean = child_vals.mean()
                        var = child_vals.var() + 1e-9
                        self.conditional_probs[(i, j, tuple(pc))] = (mean, var)
    
    def _gaussian_pdf(self, x, mean, var):
        exponent = np.exp(-(np.power(x - mean, 2) / (2 * var)))
        return (1 / np.sqrt(2 * np.pi * var)) * exponent
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for i, c in enumerate(self.classes):
                posterior = np.log(self.class_priors[i])
                
                for j in range(len(x)):
                    parents = self.structure[j]
                    if not parents:
                        mean, var = self.conditional_probs[(i, j, 'none')]
                    else:
                        pc = tuple(x[parents])
                        if pc not in [k[2] for k in self.conditional_probs.keys() if k[0] == i and k[1] == j]:
                            pc_rounded = tuple(np.round(pc, 1))
                            if (i, j, pc_rounded) in self.conditional_probs:
                                mean, var = self.conditional_probs[(i, j, pc_rounded)]
                            else:
                                mean, var = self.conditional_probs[(i, j, 'none')]
                        else:
                            mean, var = self.conditional_probs[(i, j, pc)]
                    
                    posterior += np.log(self._gaussian_pdf(x[j], mean, var))
                
                posteriors.append(posterior)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)

def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\n{model_name} 评估结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精准率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1值: {f1:.4f}")
    print("\n分类报告:")
    print(classification_report(y_true, y_pred))
    
    return accuracy, precision, recall, f1

if __name__ == '__main__':
    df = pd.read_csv('seattle_weather.csv')
    
    print("===== 数据集概览 =====")
    print(df.head())
    print(f"\n数据集大小: {df.shape}")
    print(f"天气类型分布:\n{df['weather'].value_counts()}")
    
    df['month'] = pd.to_datetime(df['date']).dt.month
    
    X = df[['precipitation', 'temp_max', 'temp_min', 'wind', 'month']].values
    y = df['weather'].values
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n===== 朴素贝叶斯分类器 =====")
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train_scaled, y_train)
    y_pred_gnb = gnb.predict(X_test_scaled)
    gnb_metrics = evaluate_model(y_test, y_pred_gnb, "朴素贝叶斯")
    
    print("\n===== 半朴素贝叶斯分类器 (ODE) =====")
    snb = SemiNaiveBayesODE()
    snb.fit(X_train_scaled, y_train)
    y_pred_snb = snb.predict(X_test_scaled)
    snb_metrics = evaluate_model(y_test, y_pred_snb, "半朴素贝叶斯")
    
    print("\n===== 贝叶斯网络分类器 =====")
    structure = {
        0: [],
        1: [0],
        2: [1],
        3: [1],
        4: [1]
    }
    bn = BayesianNetworkClassifier(structure=structure)
    bn.fit(X_train_scaled, y_train)
    y_pred_bn = bn.predict(X_test_scaled)
    bn_metrics = evaluate_model(y_test, y_pred_bn, "贝叶斯网络")
    
    print("\n===== 模型对比 =====")
    results = pd.DataFrame({
        '模型': ['朴素贝叶斯', '半朴素贝叶斯', '贝叶斯网络'],
        '准确率': [gnb_metrics[0], snb_metrics[0], bn_metrics[0]],
        '精准率': [gnb_metrics[1], snb_metrics[1], bn_metrics[1]],
        '召回率': [gnb_metrics[2], snb_metrics[2], bn_metrics[2]],
        'F1值': [gnb_metrics[3], snb_metrics[3], bn_metrics[3]]
    })
    print(results)
    
    print("\n===== sklearn GaussianNB 对比 =====")
    sklearn_gnb = GaussianNB()
    sklearn_gnb.fit(X_train_scaled, y_train)
    y_pred_sklearn = sklearn_gnb.predict(X_test_scaled)
    print(f"sklearn GaussianNB 准确率: {accuracy_score(y_test, y_pred_sklearn):.4f}")