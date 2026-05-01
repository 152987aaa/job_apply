import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score, classification_report
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/WA_Fn-UseC_-HR-Employee-Attrition.csv'
    
    try:
        df = pd.read_csv(url)
        print("成功从网络加载数据集")
    except:
        print("网络加载失败，生成模拟数据")
        df = generate_simulation_data()
    
    return df

def generate_simulation_data():
    np.random.seed(42)
    n_samples = 1470
    
    age = np.random.randint(18, 60, n_samples)
    monthly_income = np.random.randint(1000, 20000, n_samples)
    distance_from_home = np.random.randint(1, 30, n_samples)
    years_at_company = np.random.randint(1, 40, n_samples)
    years_in_current_role = np.random.randint(1, 20, n_samples)
    years_since_last_promotion = np.random.randint(0, 15, n_samples)
    years_with_curr_manager = np.random.randint(1, 15, n_samples)
    num_companies_worked = np.random.randint(1, 10, n_samples)
    total_working_years = np.random.randint(1, 45, n_samples)
    training_times_last_year = np.random.randint(0, 6, n_samples)
    daily_rate = np.random.randint(80, 1500, n_samples)
    hourly_rate = np.random.randint(30, 100, n_samples)
    percent_salary_hike = np.random.randint(10, 25, n_samples)
    performance_rating = np.random.randint(1, 5, n_samples)
    
    business_travel = np.random.choice(['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'], n_samples)
    department = np.random.choice(['Sales', 'Research & Development', 'Human Resources'], n_samples)
    education_field = np.random.choice(['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other'], n_samples)
    gender = np.random.choice(['Male', 'Female'], n_samples)
    job_role = np.random.choice(['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative'], n_samples)
    marital_status = np.random.choice(['Single', 'Married', 'Divorced'], n_samples)
    over_time = np.random.choice(['Yes', 'No'], n_samples)
    
    environment_satisfaction = np.random.randint(1, 5, n_samples)
    job_satisfaction = np.random.randint(1, 5, n_samples)
    relationship_satisfaction = np.random.randint(1, 5, n_samples)
    work_life_balance = np.random.randint(1, 5, n_samples)
    
    attrition_prob = 0.2 * (over_time == 'Yes') + \
                     0.15 * (distance_from_home > 15) + \
                     0.12 * (monthly_income < 3000) + \
                     0.1 * (years_at_company < 2) + \
                     0.1 * (job_satisfaction == 1) + \
                     np.random.rand(n_samples) * 0.4
    
    attrition = (attrition_prob > 0.6).astype(int)
    
    df = pd.DataFrame({
        'Age': age,
        'BusinessTravel': business_travel,
        'DailyRate': daily_rate,
        'Department': department,
        'DistanceFromHome': distance_from_home,
        'EducationField': education_field,
        'EnvironmentSatisfaction': environment_satisfaction,
        'Gender': gender,
        'HourlyRate': hourly_rate,
        'JobRole': job_role,
        'JobSatisfaction': job_satisfaction,
        'MaritalStatus': marital_status,
        'MonthlyIncome': monthly_income,
        'NumCompaniesWorked': num_companies_worked,
        'OverTime': over_time,
        'PercentSalaryHike': percent_salary_hike,
        'PerformanceRating': performance_rating,
        'RelationshipSatisfaction': relationship_satisfaction,
        'TotalWorkingYears': total_working_years,
        'TrainingTimesLastYear': training_times_last_year,
        'WorkLifeBalance': work_life_balance,
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_current_role,
        'YearsSinceLastPromotion': years_since_last_promotion,
        'YearsWithCurrManager': years_with_curr_manager,
        'Attrition': attrition
    })
    
    df['Attrition'] = df['Attrition'].map({0: 'No', 1: 'Yes'})
    return df

def preprocess_data(df):
    df_processed = df.copy()
    
    df_processed['Attrition'] = df_processed['Attrition'].map({'Yes': 1, 'No': 0})
    
    categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 
                        'JobRole', 'MaritalStatus', 'OverTime']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
    
    X = df_processed.drop('Attrition', axis=1)
    y = df_processed['Attrition']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_balanced, y_train_balanced = resample(
        X_train_scaled[y_train == 1],
        y_train[y_train == 1],
        replace=True,
        n_samples=len(y_train[y_train == 0]),
        random_state=42
    )
    
    X_train_balanced = np.vstack((X_train_scaled[y_train == 0], X_train_balanced))
    y_train_balanced = np.hstack((y_train[y_train == 0], y_train_balanced))
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train_balanced, y_train_balanced

def evaluate_model(y_true, y_pred, y_proba=None, model_name="Model"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    if y_proba is not None:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        auc = None
    
    print(f"\n{model_name} 评估结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精准率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1值: {f1:.4f}")
    print(f"Kappa系数: {kappa:.4f}")
    if auc is not None:
        print(f"AUC: {auc:.4f}")
    print("\n分类报告:")
    print(classification_report(y_true, y_pred))
    
    return {
        '模型': model_name,
        '准确率': accuracy,
        '精准率': precision,
        '召回率': recall,
        'F1值': f1,
        'Kappa': kappa,
        'AUC': auc
    }

if __name__ == '__main__':
    print("===== 员工离职预测分析 =====")
    
    print("\n1. 加载数据集...")
    df = load_data()
    print(f"数据集大小: {df.shape}")
    print(f"离职样本比例: {df['Attrition'].value_counts(normalize=True).round(3)}")
    
    print("\n2. 数据预处理...")
    X_train, X_test, y_train, y_test, X_train_balanced, y_train_balanced = preprocess_data(df)
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"平衡后训练集离职比例: {np.mean(y_train_balanced):.3f}")
    
    results = []
    
    print("\n===== 逻辑回归模型 =====")
    log_reg = LogisticRegression(random_state=42, class_weight='balanced')
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)
    y_proba_log = log_reg.predict_proba(X_test)
    result = evaluate_model(y_test, y_pred_log, y_proba_log, "逻辑回归")
    results.append(result)
    
    print("\n===== 随机森林模型 =====")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)
    result = evaluate_model(y_test, y_pred_rf, y_proba_rf, "随机森林")
    results.append(result)
    
    print("\n===== 随机森林(平衡数据) =====")
    rf_balanced = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_balanced.fit(X_train_balanced, y_train_balanced)
    y_pred_rf_bal = rf_balanced.predict(X_test)
    y_proba_rf_bal = rf_balanced.predict_proba(X_test)
    result = evaluate_model(y_test, y_pred_rf_bal, y_proba_rf_bal, "随机森林(平衡)")
    results.append(result)
    
    print("\n===== 模型对比 =====")
    df_results = pd.DataFrame(results)
    print(df_results.round(4))
    
    print("\n===== 特征重要性分析 =====")
    feature_names = df.drop('Attrition', axis=1).columns
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        '特征': feature_names,
        '重要性': importances
    }).sort_values('重要性', ascending=False)
    
    print(feature_importance_df.head(10))
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='重要性', y='特征', data=feature_importance_df.head(10))
    plt.title('员工离职预测 Top10 重要特征')
    plt.tight_layout()
    plt.savefig('./feature_importance.png')
    plt.close()
    print("特征重要性图已保存")
    
    print("\n===== 业务建议 =====")
    top_features = feature_importance_df.head(5)['特征'].tolist()
    print("根据模型分析，影响员工离职的主要因素:")
    print(f"1. {top_features[0]}: 该因素对离职影响最大，建议重点关注")
    print(f"2. {top_features[1]}: 建议优化相关政策")
    print(f"3. {top_features[2]}: 可作为离职预警指标")
    print(f"4. {top_features[3]}: 建议定期评估")
    print(f"5. {top_features[4]}: 可纳入员工关怀计划")
    
    print("\n===== 实验完成 =====")
    print("生成的文件:")
    print("- feature_importance.png: 特征重要性图")
    print("- employee_attrition_prediction.py: 源代码")