import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

def convert_english_to_number(word):
    num_map = {
        'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10,
        'eleven':11, 'twelve':12, 'thirteen':13, 'fourteen':14, 'fifteen':15,
        'twenty':20, 'thirty':30, 'forty':40, 'fifty':50, 'thirty-nine':39, 'forty-five':45
    }
    word = word.lower().replace('-', ' ')
    return num_map.get(word, 0)  # 未知值返回0

# ---------------------- 3. 加载并预处理数据集 ----------------------
# 加载数据（替换为你的文件路径）
df = pd.read_csv('ParisHousing.csv')
print(f"数据集加载完成，形状：{df.shape}")

# 预处理：解决所有非数值问题
df['numberOfRooms'] = df['numberOfRooms'].apply(convert_english_to_number)  # 房间数转换
bool_cols = [col for col in ['isNewBuilt', 'hasStormProtector', 'hasStorageRoom'] if col in df.columns]
for col in bool_cols:
    df[col] = df[col].astype(int)  # 布尔列转0/1
str_cols = [col for col in df.select_dtypes(include=['object']).columns if col != 'category']
for col in str_cols:
    df[col] = LabelEncoder().fit_transform(df[col])  # 字符串列编码

# 特征/目标分离 + 标准化
X = df.drop(['price', 'category'], axis=1)
y = df['category']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分层划分数据集（7:3）
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
print(f"训练集：{X_train.shape} | 测试集：{X_test.shape}")

# ---------------------- 4. 构建C4.5决策树（核心代码） ----------------------
# C4.5核心参数：entropy（熵）→ 计算信息增益率；splitter='best'（最优划分）
c45_model = DecisionTreeClassifier(
    criterion='entropy',        # 熵 → 基础是信息增益，C4.5会自动计算增益率
    max_depth=8,                # 限制树深度，防止过拟合
    min_samples_split=20,       # 节点分裂最小样本数（C4.5剪枝思想）
    splitter='best',            # 选择最优划分点（C4.5核心逻辑）
    random_state=42,            # 固定种子，结果可复现
    min_impurity_decrease=0.01  # 纯度提升小于0.01则不分裂（C4.5预剪枝）
)

# 训练模型
c45_model.fit(X_train, y_train)
# 预测
y_pred_c45 = c45_model.predict(X_test)

# ---------------------- 5. C4.5模型评估（详细指标） ----------------------
print("\n" + "="*60)
print("C4.5决策树模型详细评估结果")
print("="*60)
# 核心指标
c45_acc = accuracy_score(y_test, y_pred_c45)
c45_pre = precision_score(y_test, y_pred_c45, average='macro')
c45_rec = recall_score(y_test, y_pred_c45, average='macro')
c45_f1 = f1_score(y_test, y_pred_c45, average='macro')

print(f"准确率(Accuracy)：{c45_acc:.4f}")
print(f"精准率(Precision)：{c45_pre:.4f}")
print(f"召回率(Recall)：{c45_rec:.4f}")
print(f"F1值(F1-Score)：{c45_f1:.4f}")

# 分类报告（更详细：每个类别的精准率/召回率/F1）
print("\n分类报告（按类别）：")
print(classification_report(y_test, y_pred_c45, target_names=['基础型住房', '豪华型住房']))

# 特征重要性（C4.5优先选择的核心特征）
feature_importance = pd.DataFrame({
    '特征名称': X.columns,
    '重要性': c45_model.feature_importances_
}).sort_values('重要性', ascending=False)
print("\nC4.5核心特征重要性（前5）：")
print(feature_importance.head(5))

# ---------------------- 6. C4.5决策树可视化（中文） ----------------------
plt.figure(figsize=(20, 12))
plot_tree(
    c45_model,
    max_depth=3,                # 展示前3层，避免图形过复杂
    feature_names=X.columns,    # 特征列名
    class_names=['基础型住房', '豪华型住房'],  # 中文类别
    filled=True,                # 颜色填充（按类别区分）
    fontsize=10,                # 字体大小
    impurity=True,              # 显示熵值（C4.5的核心参考）
    rounded=True,               # 圆角节点
    proportion=True             # 显示样本占比
)
plt.title('C4.5决策树（巴黎住房分类）- 前3层', fontsize=20, pad=25)
plt.xlabel('特征划分（信息增益率最优）', fontsize=14)
plt.ylabel('决策分支', fontsize=14)
# 保存高清图片
plt.savefig('C4.5决策树_巴黎住房分类.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------------- 7. 结果保存 ----------------------
# 保存指标和特征重要性
result_df = pd.DataFrame({
    '指标': ['准确率', '精准率', '召回率', 'F1值'],
    '数值': [c45_acc, c45_pre, c45_rec, c45_f1]
})
result_df.to_csv('C4.5模型评估指标.csv', index=False, encoding='utf-8-sig')
feature_importance.to_csv('C4.5特征重要性.csv', index=False, encoding='utf-8-sig')

print("\n✅ C4.5模型结果已保存：")
print("   - C4.5决策树_巴黎住房分类.png（可视化图）")
print("   - C4.5模型评估指标.csv（核心指标）")
print("   - C4.5特征重要性.csv（特征优先级）")