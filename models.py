import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, confusion_matrix
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
import statsmodels.api as sm
import seaborn as sns
import os
import matplotlib
from sklearn.calibration import calibration_curve
import warnings  # 用于屏蔽警告

# 屏蔽警告
warnings.filterwarnings('ignore', category=UserWarning, module='LightGBM')

# 设置中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 数据路径
train_file_paths = [
    'csv/trainOmics.csv'
]

valid_file_paths = [
    'csv/testOmics.csv'
]

data_names = ['ALN']
all_results = []

# 混淆矩阵绘制函数
# 混淆矩阵绘制函数
def plot_confusion_matrix(cm, title, save_path):
    # 打印混淆矩阵以调试数据问题
    print("Confusion Matrix Data:")
    print(cm)
    plt.figure(figsize=(8, 6))
    # 绘制热力图，调整 vmin 和 vmax 使得小的数值也能显示
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 16}, vmin=0, vmax=np.max(cm))
    # 设置 xticks 和 yticks 使其与矩阵的大小匹配
    ax.set_xticks(np.arange(cm.shape[1]) + 0.5)  # 为每列设置一个刻度位置
    ax.set_yticks(np.arange(cm.shape[0]) + 0.5)  # 为每行设置一个刻度位置
    # 设置标签，确保它们与矩阵的维度一致
    ax.set_xticklabels(np.arange(cm.shape[1]), fontsize=12)
    ax.set_yticklabels(np.arange(cm.shape[0]), fontsize=12)
    plt.title(title, fontsize=18)
    plt.ylabel('Actual', fontsize=14)
    plt.xlabel('Predicted', fontsize=14)
    # 调整子图之间的间隔，避免内容重叠
    plt.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(save_path, dpi=300)
    plt.close()

# 模型字典，用于模型迭代
models = {
    'LR': LogisticRegression(class_weight='balanced'),
    'SVM': SVC(probability=True, class_weight='balanced'),
    'SGD': SGDClassifier(max_iter=1000, tol=1e-3),
    # 'KNN': KNeighborsClassifier(),
    # 'RF': RandomForestClassifier(),
    # 'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    # 'LightGBM': LGBMClassifier()
}

# 参数字典，用于超参数搜索
param_grids = {
    'LR': {
        'C': [0.00001, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.07,  0.1, 0.5, 1, 2, 5, 10,20, 30, 40, 45, 50, 55, 60, 100],  # 扩大C的范围
        'max_iter': [0.5, 1, 5, 7, 8, 9, 10, 12, 15, 20, 25, 30, 50, 60, 70, 100]
    },
    'SVM': {
        'C': [0.0001, 0.001, 0.05, 0.01, 0.1, 1, 5, 10, 50, 100, 1000],  # 扩展C范围
        'kernel': ['linear', 'rbf'],
        'gamma': [0.0001, 0.001, 0.01, 0.05, 0.1, 1],  # 增加gamma范围
    },
    'SGD': {
        'loss': ['log_loss'],
        'penalty': ['l2', 'elasticnet'],
        'alpha': [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1, 2,  5, 10]  # 加大 alpha 范围
    }
    # 'KNN': {
    #     'n_neighbors': list(range(2, 51)),  # 减小 n_neighbors 的范围，避免过拟合
    #     'weights': ['uniform', 'distance'],
    #     'algorithm': ['auto']
    # },
    # 'RF': {
    #     'n_estimators': [10, 50, 100, 200, 300, 500],  # 扩展 n_estimators 的范围
    #     'max_depth': [3, 5, 7, 10, 15, 20, 25, 50],  # 增加 max_depth 的选择范围
    #     'min_samples_split': [2, 5, 10, 15, 25, 50],  # 增加 min_samples_split 的选择范围
    #     'min_samples_leaf': [1, 5, 7, 10, 12, 15, 17, 20]  # 减少 min_samples_leaf 的范围
    # },
    # 'XGBoost': {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [3, 5, 7],
    #     'learning_rate': [0.0001, 0.001, 0.01, 0.1],  # 更低的学习率
    #     'reg_alpha': [0.1, 0.5, 1.0],  # 试探更大的正则化
    #     'reg_lambda': [0.5, 1.0, 2.0]  # 增加正则化
    # },
    # 'LightGBM': {
    #     'n_estimators': [50, 100, 200, 300],
    #     'max_depth': [5, 7],
    #     'learning_rate': [0.01, 0.05],  # 更低的学习率
    #     'num_leaves': [15, 31],  # 更细粒度的叶子节点数选择
    #     'colsample_bytree': [0.6, 0.7, 0.8],
    #     'subsample': [0.6, 0.7, 0.8],
    #     'min_child_weight': [0.1, 0.5, 1.0],
    #     'reg_alpha': [1.0, 2.0],  # 增大正则化
    #     'reg_lambda': [1.0, 2.0],  # 增大正则化
    #     'min_child_samples': [20, 50]  # 增加子样本限制
    # }
}


# 创建用于保存ROC曲线的图像
fig_train, ax_train = plt.subplots(figsize=(10, 8))
fig_valid, ax_valid = plt.subplots(figsize=(10, 8))

# 存储所有患者的预测概率
all_patient_predictions = pd.DataFrame()

# 存储已训练好的模型
models_fitted = {}

# 遍历每个数据集
for index, (train_file_path, valid_file_path) in enumerate(zip(train_file_paths, valid_file_paths)):
    # 加载训练和验证数据集
    train_data = pd.read_csv(train_file_path)
    X_train = sm.add_constant(train_data.drop(['label', 'index'], axis=1))
    y_train = train_data['label']
    valid_data = pd.read_csv(valid_file_path)
    X_valid = sm.add_constant(valid_data.drop(['label', 'index'], axis=1))
    y_valid = valid_data['label']

    patient_ids_train = train_data['index']
    patient_ids_valid = valid_data['index']

    # 对每个模型进行训练和评估
    for model_name, model in models.items():
        print(f"Processing {model_name} on {data_names[index]} dataset...")

        # 使用交叉验证进行参数搜索
        param_grid = param_grids[model_name]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=50)
        grid_search = RandomizedSearchCV(model, param_grid, scoring='roc_auc', cv=cv, n_iter=100, random_state=50)
        grid_search.fit(X_train, y_train)

        # 获取最佳模型
        best_model = grid_search.best_estimator_
        # 保存训练好的模型到字典中
        models_fitted[model_name] = best_model

        # 打印最佳参数
        print(f"最佳参数: {grid_search.best_params_}")

        # 获取最佳模型并进行预测
        best_model = grid_search.best_estimator_
        train_predictions_proba = best_model.predict_proba(X_train)[:, 1]
        valid_predictions_proba = best_model.predict_proba(X_valid)[:, 1]

        # 计算ROC曲线和AUC值
        fpr_train, tpr_train, thresholds_train = roc_curve(y_train, train_predictions_proba)
        fpr_valid, tpr_valid, thresholds_valid = roc_curve(y_valid, valid_predictions_proba)
        roc_auc_train = auc(fpr_train, tpr_train)
        roc_auc_valid = auc(fpr_valid, tpr_valid)

        # 计算混淆矩阵
        optimal_threshold_train = thresholds_train[np.argmax(tpr_train - fpr_train)]
        optimal_threshold_valid = thresholds_valid[np.argmax(tpr_valid - fpr_valid)]

        train_predictions = (train_predictions_proba >= optimal_threshold_train).astype(int)
        valid_predictions = (valid_predictions_proba >= optimal_threshold_valid).astype(int)

        cm_train = confusion_matrix(y_train, train_predictions)
        cm_valid = confusion_matrix(y_valid, valid_predictions)

        sensitivity_train = cm_train[1, 1] / (cm_train[1, 1] + cm_train[1, 0])
        specificity_train = cm_train[0, 0] / (cm_train[0, 0] + cm_train[0, 1])

        sensitivity_valid = cm_valid[1, 1] / (cm_valid[1, 1] + cm_valid[1, 0])
        specificity_valid = cm_valid[0, 0] / (cm_valid[0, 0] + cm_valid[0, 1])

        from sklearn.utils import resample
        import numpy as np


        def compute_ci_only(auc_value, y_true, y_proba, n_bootstraps=1000, ci_level=0.95):
            aucs = []
            rng = np.random.RandomState(42)
            for i in range(n_bootstraps):
                # 使用bootstrap重采样
                indices = rng.randint(0, len(y_true), len(y_true))
                if len(np.unique(y_true[indices])) < 2:
                    # 如果采样后的数据集没有两个类别，则跳过
                    continue
                fpr, tpr, _ = roc_curve(y_true[indices], y_proba[indices])
                aucs.append(auc(fpr, tpr))

            # 计算置信区间
            sorted_aucs = np.sort(aucs)
            lower_bound = np.percentile(sorted_aucs, (1 - ci_level) / 2 * 100)
            upper_bound = np.percentile(sorted_aucs, (1 + ci_level) / 2 * 100)

            # 仅返回给定AUC的置信区间，不计算均值
            return lower_bound, upper_bound


        # 传入已经计算好的AUC值，并只计算CI
        train_lower_auc_bound, train_upper_auc_bound = compute_ci_only(roc_auc_train, y_train, train_predictions_proba)
        valid_lower_auc_bound, valid_upper_auc_bound = compute_ci_only(roc_auc_valid, y_valid, valid_predictions_proba)

        # 存储结果到列表
        all_results.append({
            'Feature': data_names[index],
            'Model': model_name,
            'AUC_train': roc_auc_train,
            'train 95% CI Lower': train_lower_auc_bound,
            'train 95% CI Upper': train_upper_auc_bound,
            'Sensitivity_train': sensitivity_train,
            'Specificity_train': specificity_train,
            'AUC_valid': roc_auc_valid,
            'valid 95% CI Lower': valid_lower_auc_bound,
            'valid 95% CI Upper': valid_upper_auc_bound,
            'Sensitivity_valid': sensitivity_valid,
            'Specificity_valid': specificity_valid,
        })

        # 绘制训练集ROC曲线，并在图例中加入AUC及其95%置信区间
        ax_train.plot(fpr_train, tpr_train,
                      label=f'{model_name} (AUC: {roc_auc_train:.2f}, 95% CI: [{train_lower_auc_bound:.2f}, {train_upper_auc_bound:.2f}])',
                      linewidth=2)

        # 绘制验证集ROC曲线，并在图例中加入AUC及其95%置信区间
        ax_valid.plot(fpr_valid, tpr_valid,
                      label=f'{model_name} (AUC: {roc_auc_valid:.2f}, 95% CI: [{valid_lower_auc_bound:.2f}, {valid_upper_auc_bound:.2f}])',
                      linewidth=2)

        # 生成混淆矩阵图像并保存
        save_dir = 'RESULTS'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path_cm_train = os.path.join(save_dir, f'{data_names[index]}_{model_name}_confusion_matrix_train.jpg')
        save_path_cm_valid = os.path.join(save_dir, f'{data_names[index]}_{model_name}_confusion_matrix_valid.jpg')
        plot_confusion_matrix(cm_train, f'{model_name} Training Set Confusion Matrix', save_path_cm_train)
        plot_confusion_matrix(cm_valid, f'{model_name} Validation Set Confusion Matrix', save_path_cm_valid)

        # 存储所有患者的预测概率
        patient_predictions = pd.DataFrame({
            'index': np.concatenate([patient_ids_train, patient_ids_valid]),
            'Set': ['Train'] * len(patient_ids_train) + ['Valid'] * len(patient_ids_valid),
            f'{data_names[index]}_{model_name}': np.concatenate([train_predictions_proba, valid_predictions_proba])
        })

        if all_patient_predictions.empty:
            all_patient_predictions = patient_predictions
        else:
            all_patient_predictions = all_patient_predictions.merge(patient_predictions, on=['index', 'Set'], how='outer')

# 设置轴标签和标题的字体大小
ax_train.set_xlabel('False Positive Rate', fontsize=12)
ax_train.set_ylabel('True Positive Rate', fontsize=12)
ax_train.set_title('training set', fontsize=14)

ax_valid.set_xlabel('False Positive Rate', fontsize=12)
ax_valid.set_ylabel('True Positive Rate', fontsize=12)
ax_valid.set_title('validation set', fontsize=14)

# 添加对角线和图例
ax_train.plot([0, 1], [0, 1], linestyle='--', color='gray')
ax_valid.plot([0, 1], [0, 1], linestyle='--', color='gray')

ax_train.legend(loc="lower right", prop={'size': 14})
ax_valid.legend(loc="lower right", prop={'size': 14})

# 保存ROC曲线
save_dir = 'RESULTS'
fig_train.savefig(os.path.join(save_dir, 'roc_curve_train.jpg'), dpi=300)
fig_valid.savefig(os.path.join(save_dir, 'roc_curve_valid.jpg'), dpi=300)
plt.close(fig_train)
plt.close(fig_valid)

# 保存所有患者的预测概率
predictions_excel_path = os.path.join(save_dir, 'predictions.xlsx')
all_patient_predictions.to_excel(predictions_excel_path, index=False)
print(f"预测概率Excel文件已保存到: {predictions_excel_path}")

# 将all_results列表转换为DataFrame并保存到Excel
df_results = pd.DataFrame(all_results)
excel_path = os.path.join(save_dir, 'auc_results.xlsx')
df_results.to_excel(excel_path, index=False)
print(f"AUC结果Excel文件已保存到: {excel_path}")

def calculate_std_err(y_test, y_score, prob_pred):
    # 获取bin的边界
    bin_edges = np.unique([0, *prob_pred, 1])
    # 计算属于每个bin的样本的索引
    bin_indices = np.digitize(y_score, bin_edges[1:-1])
    # 初始化列表存储每个bin的标准误差
    std_errs = []
    for bin_index in range(len(prob_pred)):
        # 获取当前bin的实际值
        actual_values = y_test[bin_indices == bin_index]
        # 只有当 bin 中有值时才计算标准误差
        if len(actual_values) > 0:
            std_err = np.std(actual_values) / np.sqrt(len(actual_values))
            std_errs.append(std_err)
        else:
            std_errs.append(0)

    return std_errs


# 修改后的 predict_models 函数
def predict_models(models_fitted, test_sets):
    # 初始化一个列表来存储每个模型的预测结果
    y_scores = []

    # 对于每个已训练好的模型，在测试集上进行预测，并将预测概率添加到列表中
    for model, test_set in zip(models_fitted, test_sets):
        X_test, _ = test_set
        y_score = model.predict_proba(X_test)[:, 1]
        y_scores.append(y_score)

    return y_scores


# 获取所有的测试集和训练集
test_sets = [(X_train, y_train), (X_valid, y_valid)]
for test_set_index, test_set in enumerate(test_sets):
    # 获取模型预测结果
    # 使用训练好的模型进行预测
    y_scores = predict_models(models_fitted.values(), [test_set] * len(models_fitted))

    # 初始化图像
    plt.figure(figsize=(8, 8))
    # 获取颜色映射
    cmap = matplotlib.colormaps['tab10']  # 使用 Matplotlib 3.7 以后的新方式

    # 对每个模型进行绘图
    for index, y_score in enumerate(y_scores):
        _, y_test = test_set
        prob_true, prob_pred = calibration_curve(y_test, y_score, n_bins=5)
        std_errs = calculate_std_err(y_test, y_score, prob_pred)
        scaled_std_errs = [err * 0.5 for err in std_errs]
        # 获取相同的颜色
        color = cmap(index)

        # 使用相同颜色绘制误差棒和校准曲线
        plt.errorbar(prob_pred, prob_true, yerr=scaled_std_errs, fmt='o', label=list(models_fitted.keys())[index],
                     color=color, elinewidth=1, capsize=3)
        # 确保同样的颜色用于校准曲线
        plt.plot(prob_pred, prob_true, color=color, linewidth=2)
        # 绘制理想的校准曲线
    plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')

    # 添加图例和标签
    plt.legend(loc="upper left")
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Probability')

    if test_set_index == 0:
        # 保存训练集的图像
        plt.savefig(r'T\校准曲线_训练集.jpg',
                    format='jpg', dpi=300)
    else:
        # 保存验证集的图像
        plt.savefig(r'\校准曲线_验证集.jpg',
                    format='jpg', dpi=300)


def plot_DCA(ax, thresh_group, models, X, y, model_names, set_name):
    colors = ['crimson', 'blue', 'green', 'purple', 'orange', 'black', 'dodgerblue']

    # 绘制无操作（净效益为0）的曲线
    ax.plot(thresh_group, [0] * len(thresh_group), color='gray', label='None', linestyle='-')

    # 先绘制参考曲线（假设所有样本都是正类）
    net_benefit_all = calculate_net_benefit_all(thresh_group, y)
    ax.plot(thresh_group, net_benefit_all, color='black', label='All Positive', linestyle='--')

    for model, model_name in zip(models, model_names):
        model_color = colors[model_names.index(model_name)]
        net_benefit = calculate_net_benefit_model(thresh_group, model, X, y)
        ax.plot(thresh_group, net_benefit, color=model_color, label=f'{model_name}', linewidth=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, 0.6)
    ax.set_xlabel('High Risk Threshold', fontdict={'family': 'Times New Roman', 'fontsize': 13})
    ax.set_ylabel('Net Benefit', fontdict={'family': 'Times New Roman', 'fontsize': 13})
    ax.grid('off')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc='upper right')
    ax.set_title(f'DCA Curve for Models on {set_name} Set', fontsize=16)


def calculate_net_benefit_model(thresh_group, model, X_test, y_test):
    y_scores = model.predict_proba(X_test)[:, 1]
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred = (y_scores > thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        tn = np.sum((y_pred == 0) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        total = len(y_test)
        net_benefit = (tp / total) - (fp / total) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


def calculate_net_benefit_all(thresh_group, y_test):
    net_benefit_all = np.array([])
    tp = np.sum(y_test == 1)
    fp = np.sum(y_test == 0)
    total = len(y_test)
    for thresh in thresh_group:
        net_benefit = (tp / total) - (fp / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all


# 定义阈值组
thresh_group = np.arange(0, 1, 0.01)


# 设置保存文件夹路径
save_dir = 'RESULTS'

# 检查文件夹是否存在，不存在则创建
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 为训练集绘制DCA图
fig_train, ax_train = plt.subplots(figsize=(10, 8))
plot_DCA(ax_train, thresh_group, list(models_fitted.values()), X_train, y_train, list(models_fitted.keys()), "Training")

# 保存训练集的DCA图像
fig_train.savefig(os.path.join(save_dir, '决策曲线_训练集.jpg'), dpi=300)
plt.close(fig_train)  # 关闭图像，释放内存

# 为验证集绘制DCA图
fig_valid, ax_valid = plt.subplots(figsize=(10, 8))
plot_DCA(ax_valid, thresh_group, list(models_fitted.values()), X_valid, y_valid, list(models_fitted.keys()), "Validation")

# 保存验证集的DCA图像
fig_valid.savefig(os.path.join(save_dir, '决策曲线_验证集.jpg'), dpi=300)
plt.close(fig_valid)  # 关闭图像，释放内存



# 确保 auc_scores 被填充
auc_scores = {model_name: {} for model_name in models}

# 在模型训练和评估的过程中填充 auc_scores 字典
for index, (train_file_path, valid_file_path) in enumerate(zip(train_file_paths, valid_file_paths)):
    # 加载数据
    train_data = pd.read_csv(train_file_path)
    X_train = sm.add_constant(train_data.drop(['label', 'index'], axis=1))
    y_train = train_data['label']
    valid_data = pd.read_csv(valid_file_path)
    X_valid = sm.add_constant(valid_data.drop(['label', 'index'], axis=1))
    y_valid = valid_data['label']

    # 训练和评估每个模型
    for model_name, model in models.items():
        print(f"Processing {model_name} on {data_names[index]} dataset...")

        # 交叉验证并获取最佳模型
        param_grid = param_grids[model_name]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = RandomizedSearchCV(model, param_grid, scoring='roc_auc', cv=cv, n_iter=20, random_state=42)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # 获取模型预测概率并计算AUC
        train_predictions_proba = best_model.predict_proba(X_train)[:, 1]
        valid_predictions_proba = best_model.predict_proba(X_valid)[:, 1]
        fpr_train, tpr_train, thresholds_train = roc_curve(y_train, train_predictions_proba)
        fpr_valid, tpr_valid, thresholds_valid = roc_curve(y_valid, valid_predictions_proba)
        roc_auc_train = auc(fpr_train, tpr_train)
        roc_auc_valid = auc(fpr_valid, tpr_valid)

        # 将 AUC 分数添加到 auc_scores 字典
        auc_scores[model_name][data_names[index]] = roc_auc_valid  # 存储验证集的 AUC 分数
