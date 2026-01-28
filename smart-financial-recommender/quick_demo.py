"""
快速演示版本 - 极不平衡样本多步推荐模型
针对无锡农商行社保卡客户转化场景
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("无锡农商行极不平衡样本多步推荐模型")
print("="*60)

# 1. 模拟无锡农商行数据（百万级社保卡客户，只有5%持有其他产品）
print("\n1. 创建模拟数据...")
n_customers = 50000  # 使用较小的数据集快速演示
n_other_products = int(n_customers * 0.05)  # 5%持有其他产品

np.random.seed(42)

# 客户数据
customers = pd.DataFrame({
    'cust_no': [f'WX_{i:08d}' for i in range(n_customers)],
    'age': np.random.normal(40, 15, n_customers).astype(int),
    'gender': np.random.choice(['M', 'F'], n_customers),
    'income': np.random.lognormal(10, 0.5, n_customers),
    'tenure_years': np.random.exponential(5, n_customers),
    'has_other_products': 0  # 默认只有储蓄
})

# 设置正样本（持有其他产品）
positive_indices = np.random.choice(n_customers, n_other_products, replace=False)
customers.loc[positive_indices, 'has_other_products'] = 1

print(f"   总客户数: {n_customers:,}")
print(f"   仅储蓄客户: {n_customers - n_other_products:,} ({(n_customers-n_other_products)/n_customers*100:.1f}%)")
print(f"   持有其他产品: {n_other_products:,} ({n_other_products/n_customers*100:.1f}%)")

# 2. 特征工程 - 考虑客户发展逻辑
print("\n2. 特征工程...")

# 生命周期特征
def get_life_stage(age):
    if age < 25: return 'student'
    elif age < 35: return 'young'
    elif age < 50: return 'family'
    elif age < 65: return 'mature'
    else: return 'retire'

customers['life_stage'] = customers['age'].apply(get_life_stage)

# 社会经济分层
def get_ses(income):
    if income < 40000: return 'low'
    elif income < 80000: return 'medium'
    else: return 'high'

customers['ses'] = customers['income'].apply(get_ses)

# 营销机会标签
customers['marketing_opportunity'] = np.where(
    (customers['age'] > 30) & (customers['age'] < 50) & (customers['income'] > 50000),
    'high',
    np.where(
        (customers['income'] > 30000),
        'medium',
        'low'
    )
)

print(f"   特征维度: {customers.shape[1]}")

# 3. 准备建模数据
print("\n3. 准备建模数据...")

# 选择特征
features = ['age', 'income', 'tenure_years']
X = customers[features]

# 编码分类特征
le_life = LabelEncoder()
le_ses = LabelEncoder()
le_opp = LabelEncoder()

customers['life_stage_enc'] = le_life.fit_transform(customers['life_stage'])
customers['ses_enc'] = le_ses.fit_transform(customers['ses'])
customers['marketing_opportunity_enc'] = le_opp.fit_transform(customers['marketing_opportunity'])

# 添加编码特征
X = pd.concat([X, customers[['life_stage_enc', 'ses_enc', 'marketing_opportunity_enc']]], axis=1)

y = customers['has_other_products']

# 4. 处理极不平衡数据
print("\n4. 处理极不平衡数据...")

# 统计不平衡比例
neg_count = (y == 0).sum()
pos_count = (y == 1).sum
print(f"   负样本: {neg_count:,}")
print(f"   正样本: {pos_count():,}")
print(f"   不平衡比例: {neg_count/pos_count():.1f}:1")

# 使用SMOTE过采样
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print(f"   SMOTE处理后:")
print(f"   负样本: {(y_res==0).sum():,}")
print(f"   正样本: {(y_res==1).sum():,}")

# 5. 多步推荐模型
print("\n5. 构建多步推荐模型...")

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# 步骤1: 初步推荐模型
print("\n   步骤1: 初步推荐模型")
model1 = GradientBoostingClassifier(n_estimators=100, random_state=42)
model1.fit(X_train, y_train)

y_pred1 = model1.predict(X_test)
y_prob1 = model1.predict_proba(X_test)[:, 1]

acc1 = accuracy_score(y_test, y_pred1)
f1_1 = f1_score(y_test, y_pred1)
auc1 = roc_auc_score(y_test, y_prob1)

print(f"      准确率: {acc1:.3f}")
print(f"      F1分数: {f1_1:.3f}")
print(f"      AUC: {auc1:.3f}")

# 步骤2: 基于营销机会的差异化模型
print("\n   步骤2: 差异化推荐模型")
# 在原数据上预测，只对高机会客户应用
high_opp_mask = customers['marketing_opportunity'] == 'high'
X_high_opp = X[high_opp_mask]
y_high_opp = y[high_opp_mask]

if len(X_high_opp) > 0:
    # 使用步骤1模型预测概率
    prob_high = model1.predict_proba(X_high_opp)[:, 1]
    # 对高概率客户进行更积极的推荐
    y_pred2 = (prob_high > 0.3).astype(int)

    # 评估
    if len(np.unique(y_high_opp)) > 1:
        f1_2 = f1_score(y_high_opp, y_pred2)
        auc2 = roc_auc_score(y_high_opp, prob_high)
        print(f"      高机会客户F1: {f1_2:.3f}")
        print(f"      高机会客户AUC: {auc2:.3f}")

# 6. 特征重要性分析
print("\n6. 特征重要性分析...")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model1.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   Top 5 重要特征:")
for _, row in feature_importance.head(5).iterrows():
    print(f"   - {row['feature']}: {row['importance']:.4f}")

# 7. 生成推荐
print("\n7. 生成推荐示例...")

# 选择只有储蓄的客户进行推荐
savings_only = customers[customers['has_other_products'] == 0].sample(100)
X_savings = savings_only[features]

# 添加编码特征
X_savings = pd.concat([
    X_savings,
    savings_only[['life_stage_enc', 'ses_enc', 'marketing_opportunity_enc']]
], axis=1)

# 预测购买其他产品的概率
probs = model1.predict_proba(X_savings)[:, 1]
savings_only['probability'] = probs

# 选择Top 10
top_customers = savings_only.nlargest(10, 'probability')

print("\n   Top 10 推荐客户:")
for _, customer in top_customers.iterrows():
    print(f"   - {customer['cust_no']}: {customer['probability']:.3f} "
          f"(年龄:{customer['age']}, 收入:{customer['income']:.0f}, "
          f"营销机会:{customer['marketing_opportunity']})")

# 8. 保存结果
print("\n8. 保存结果...")

# 保存推荐
top_customers[['cust_no', 'probability', 'age', 'income', 'marketing_opportunity']].to_csv(
    'recommendations_demo.csv', index=False
)
print("   [√] 推荐结果已保存: recommendations_demo.csv")

# 性能报告
performance = {
    'model_accuracy': acc1,
    'model_f1': f1_1,
    'model_auc': auc1,
    'imbalance_ratio': neg_count/pos_count(),
    'total_customers': n_customers,
    'other_products_holders': n_other_products
}

import json
with open('performance_demo.json', 'w', encoding='utf-8') as f:
    json.dump(performance, f, ensure_ascii=False, indent=2)

print("   [√] 性能报告已保存: performance_demo.json")

print("\n" + "="*60)
print("演示完成！")
print("="*60)
print("\n关键成果:")
print(f"1. 成功处理{neg_count/pos_count():.0f}:1的极不平衡样本")
print(f"2. 模型AUC达到{auc1:.3f}")
print(f"3. 识别出{len(top_customers)}个高潜力客户")
print(f"4. 实现了差异化营销推荐")