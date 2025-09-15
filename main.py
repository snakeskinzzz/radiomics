import os
import pandas as pd
import numpy as np
from radiomics import featureextractor
import SimpleITK as sitk
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, RFE
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score, confusion_matrix, 
                            roc_curve, auc, RocCurveDisplay)

from scipy.stats import levene, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

### 通过pyradiomics提取影像组学特征

def feature_select():
	kinds = ['TUMOR','HEALTHY']
	# 这个是特征处理配置文件，具体可以参考pyradiomics官网
	para_path = 'yaml/MR_1mm.yaml'

	extractor = featureextractor.RadiomicsFeatureExtractor(para_path) 
	dir = 'data/MyData/'

	for kind in kinds:
		print("{}:开始提取特征".format(kind))
		df = pd.DataFrame()
		path =  dir + kind
		# 使用配置文件初始化特征抽取器
		for index, folder in enumerate(os.listdir(path)):
			if folder == '.DS_Store':
				continue
			ori_path = os.path.join(path, folder, 'phiMap.nii')
			file_name = folder[:-4] if folder[-3:]=='ROI' else folder
			lab_path = os.path.join(path, folder, file_name + '.nii.gz')
			# for f in os.listdir(os.path.join(path, folder)):
			#     if 't1ce' in f:
			#         ori_path = os.path.join(path,folder, f)
			#         break
			# lab_path = ori_path.replace('t1ce','seg')
			features = extractor.execute(ori_path,lab_path)  #抽取特征
			#新增一列用来保存病例文件夹名字
			features = {'index': file_name, **features}
			df_add = pd.DataFrame.from_dict(features.values()).T
			df_add.columns = features.keys()
			df = pd.concat([df, df_add])
		df.to_csv('results/' +'{}.csv'.format(kind),index=0)
		print('提取特征完成')
    
# feature_select()
### 对提取出来的csv文件进一步处理，删除字符串的特征，并增加lable标记。
tumor_data = pd.read_csv('results/TUMOR.csv')
healthy_data = pd.read_csv('results/HEALTHY.csv')

tumor_data.insert(1,'label', 1) #插入标签
healthy_data.insert(1,'label', 0) #插入标签

#因为有些特征是字符串，直接删掉
cols=[x for i,x in enumerate(tumor_data.columns) if type(tumor_data.iat[1,i]) == str]
cols.remove('index')
tumor_data=tumor_data.drop(cols,axis=1)
cols=[x for i,x in enumerate(healthy_data.columns) if type(healthy_data.iat[1,i]) == str]
cols.remove('index')
healthy_data=healthy_data.drop(cols,axis=1)

#再合并成一个新的csv文件。
total_data = pd.concat([tumor_data, healthy_data])
total_data = shuffle(total_data)
total_data.to_csv('results/TotalOMICS.csv',index=False)

#简单查看数据的分布
# fig, ax = plt.subplots()
# sns.set()
# ax = sns.countplot(x='label',hue='label',data=total_data)
# plt.show()
# print(total_data['label'].value_counts())

### 划分数据
# 设置随机种子保证结果可重现
np.random.seed(888)

# 数据加载和预处理
# 划分训练集和测试集
train_data, test_data = train_test_split(total_data, test_size=0.2, random_state=888)
print(f'训练集样本数: {len(train_data)}')
print(f'测试集样本数: {len(test_data)}')

# 保存划分后的数据
train_data.to_csv("results/trainOmics.csv", index=False)
test_data.to_csv("results/testOmics.csv", index=False)

### 对单纯影像组学建模,先做T检验，再做Lasso回归进行对特征筛选

# t检验
def t_test(tData):
    # tData = pd.read_csv(file_path)
	# 分离两类数据
	df1 = tData[tData['label'] == 1]
	df0 = tData[tData['label'] == 0]

	# 获取数值列，供下面遍历
	numeric_columns = tData.select_dtypes(include=[np.number]).columns.tolist()
	if 'index' in numeric_columns:
		numeric_columns.remove('index')
	if 'label' in numeric_columns:
		numeric_columns.remove('label')

	columns_index = []
	for column_name in numeric_columns:
		if levene(df1[column_name], df0[column_name])[1] > 0.05:
			if ttest_ind(df1[column_name],df0[column_name],equal_var=True)[1] < 0.05:
				columns_index.append(column_name)
		else:
			if ttest_ind(df1[column_name],df0[column_name],equal_var=False)[1] < 0.05:
				columns_index.append(column_name)
	print(f"T检验筛选后剩下的特征数：{len(columns_index)}个")
	# 保存筛选后的数据
	selected_columns = ['index', 'label'] + columns_index
	tData_train_filtered = tData[selected_columns]
	tData_train_filtered.to_csv('./results/tData_train.csv', header=True, index=False, encoding="utf-8")
	return tData_train_filtered

def plot_anova_results(results_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # F值条形图
    ax1.barh(results_df['feature'], results_df['f_score'])
    ax1.set_xlabel('F-score')
    ax1.set_title('ANOVA特征重要性得分')
    
    # p值散点图（-log10转换）
    ax2.scatter(results_df['f_score'], -np.log10(results_df['p_value']))
    ax2.axhline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
    for i, feature in enumerate(results_df['feature']):
        ax2.annotate(feature, (results_df['f_score'].iloc[i], 
                              -np.log10(results_df['p_value'].iloc[i])))
    ax2.set_xlabel('F-score')
    ax2.set_ylabel('-log10(p-value)')
    ax2.set_title('F-score vs -log10(p-value)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_rfe_results(rfe, feature_names, title="RFE Result"):
    """
    可视化RFE结果
    """
    # 创建特征排名DataFrame
    results_df = pd.DataFrame({
        'feature': feature_names,
        'ranking': rfe.ranking_,
        'selected': rfe.support_
    }).sort_values('ranking')
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # 特征排名图
    colors = ['green' if sel else 'red' for sel in results_df['selected']]
    ax1.barh(results_df['feature'], results_df['ranking'], color=colors)
    ax1.set_xlabel('ranking(1=best)')
    ax1.set_title(f'{title} - ranking')
    ax1.axvline(x=1.5, color='blue', linestyle='--', alpha=0.7)
    
    # 选中特征比例图
    selected_count = sum(results_df['selected'])
    total_count = len(results_df)
    labels = ['selected', 'eliminated']
    sizes = [selected_count, total_count - selected_count]
    colors = ['lightgreen', 'lightcoral']
    
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')
    ax2.set_title(f'total: {total_count}')
    
    # plt.tight_layout()
    plt.show()
    
    return results_df

# LASSO特征选择
def lasso_feature_selection(data):
	print("进行LASSO特征选择...")
	# 准备数据
	X = data.drop(['index', 'label'], axis=1)
	y = data['label']
	X = X.apply(pd.to_numeric,errors = 'ignore') # transform the type of the data
	colNames = X.columns # to read the feature's name
	X = X.fillna(0)
	X = X.astype(np.float64)
	X = StandardScaler().fit_transform(X)
	X = pd.DataFrame(X)
	X.columns = colNames

	# LASSO回归进行特征选择
	alphas = np.logspace(-3,1,30)
	lasso_cv = LassoCV(alphas = alphas, cv=10, random_state=888, max_iter=10000).fit(X, y)

	# 选择非零系数特征
	model_lassoCV = LogisticRegression(penalty='l1', solver='liblinear', C=1/lasso_cv.alpha_, random_state=888).fit(X, y)

	# 获取选择的特征
	# coef = pd.Series(model_lassoCV.coef_, index = X.columns)
	# print(f"选择的特征: {coef}")

	selected_features_mask = model_lassoCV.coef_[0] != 0
	selected_features = X.columns[selected_features_mask]
	selected_coefficients = model_lassoCV.coef_[0][selected_features_mask]

	print("LASSO选择的特征及其系数:")
	for feature, coef in zip(selected_features, selected_coefficients):
		print(f"{feature}: {coef:.4f}")

# ANOVA特征选择
def anova_feature_selection(data, alpha=0.05, top_k=None):
	"""
    自定义ANOVA特征选择函数
	适用场景：ANOVA适用于分类问题中的连续特征选择
	数据假设：
	特征应该是连续型的
	数据应该近似正态分布
	各组方差应该相等（方差齐性）
    
    参数:
    X: 特征DataFrame
    y: 目标变量
    alpha: 显著性水平
    top_k: 选择前k个特征，如果为None则使用p值阈值
    
    返回:
    selected_features: 选中的特征列表
    results_df: 详细结果DataFrame
    """
	print("进行ANOVA特征选择...")
	# 准备数据
	X = data.drop(['index', 'label'], axis=1)
	y = data['label']
	X = X.apply(pd.to_numeric,errors = 'ignore') # transform the type of the data
	colNames = X.columns # to read the feature's name
	X = X.fillna(0)
	X = X.astype(np.float64)
	X = StandardScaler().fit_transform(X)
	X = pd.DataFrame(X)
	X.columns = colNames

	# k_best = 2
	# selector = SelectKBest(score_func=f_classif, k=k_best)
	# selector.fit(X, y)
	# feature_index = selector.get_support(True)
	# f_value, p_value = f_classif(X, y)
    # 计算ANOVA得分
	f_scores, p_values = f_classif(X, y)
    
    # 创建结果表格
	results_df = pd.DataFrame({
        'feature': X.columns,
        'f_score': f_scores,
        'p_value': p_values,
        'significant': p_values < alpha
    }).sort_values('f_score', ascending=False)
    
    # 选择特征
	if top_k is not None:
        # 选择前k个特征
		selected_features = results_df.head(top_k)['feature'].tolist()
	else:
        # 基于p值阈值选择
		selected_features = results_df[results_df['significant']]['feature'].tolist()
	print(f'ANOVA特征选择: {selected_features}')
	plot_anova_results(results_df)
	return selected_features, results_df

def RFE__feature_selection(data):
	print("进行RFE特征选择...")
	# 准备数据
	X = data.drop(['index', 'label'], axis=1)
	y = data['label']
	X = X.apply(pd.to_numeric,errors = 'ignore') # transform the type of the data
	colNames = X.columns # to read the feature's name
	X = X.fillna(0)
	X = X.astype(np.float64)
	X = StandardScaler().fit_transform(X)
	X = pd.DataFrame(X)
	X.columns = colNames
	# estimator = LogisticRegression(max_iter=1000, random_state=42)
	# estimator = RandomForestClassifier(random_state=42)
	estimator = RFE(
		SVC(kernel='linear', random_state=42),
		n_features_to_select = 5,# 要选择的特征数量
		step=0.05,# 每次迭代移除的特征数量
		verbose=1# 显示进度
		)
	estimator.fit(X, y)
	# print("特征排名 (1表示被选中):", estimator.ranking_)
	# print("特征支持掩码:", estimator.support_)
	print("选中的特征:", X.columns[estimator.support_].tolist())
	plot_rfe_results(estimator, X.columns)

tData_train_filtered = t_test(train_data)

# lasso_feature_selection(tData_train_filtered)

# anova_feature_selection(tData_train_filtered, top_k=2)

RFE__feature_selection(tData_train_filtered)