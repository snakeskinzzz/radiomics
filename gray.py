import os
import pandas as pd
import numpy as np
from radiomics import featureextractor, imageoperations
import nibabel as nib
import SimpleITK as sitk
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, RFE, VarianceThreshold
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score, confusion_matrix, 
                            roc_curve, auc, RocCurveDisplay)

from scipy.stats import levene, ttest_ind, pearsonr, skew, kurtosis
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')



# kinds = ['TUMOR','HEALTHY']
path = 'data/MyData/TUMOR'
data = pd.read_excel('tumor.xlsx', index_col=0)
# data = pd.read_excel('/Users/headsnakeyu/Desktop/test2.xls', sheet_name='healthy', index_col=0)
# data['c_mean'] = None
# data['c_max'] = None
# data['c_min'] = None
# data['c_p10'] = None
# data['c_p25'] = None
# data['c_p50'] = None
# data['c_p75'] = None
# data['c_p90'] = None
# data['c_p95'] = None
# data['c_var'] = None
# data['c_skew'] = None
# data['c_ptp'] = None
# data['c_kurtosis'] = None
# data['c_std'] = None
# data['c_median'] = None

data['phi_mean'] = None
data['phi_max'] = None
data['phi_min'] = None
data['phi_p10'] = None
data['phi_p25'] = None
data['phi_p50'] = None
data['phi_p75'] = None
data['phi_p90'] = None
data['phi_p95'] = None
data['phi_var'] = None
data['phi_skew'] = None
data['phi_ptp'] = None
data['phi_kurtosis'] = None
data['phi_std'] = None
data['phi_median'] = None

for index, folder in enumerate(os.listdir(path)):
	if folder == '.DS_Store':
		continue
	origin_path = os.path.join(path, folder, 'phiMap.nii')
	file_name = folder[:-4] if folder[-3:]=='ROI' else folder
	roi_path = os.path.join(path, folder, file_name + '.nii.gz')
	print(file_name)
	image = sitk.ReadImage(origin_path)
	roi_mask = sitk.ReadImage(roi_path)
	original_array = sitk.GetArrayFromImage(image)
	roi_array = sitk.GetArrayFromImage(roi_mask)
	original_slice = original_array.flatten()
	roi_slice = roi_array.flatten()
	roi_binary = (roi_slice > 0).astype(np.uint8)
	roi_pixels = original_slice[roi_binary > 0]
	# statistics = {
	# 	'平均值': np.mean(roi_pixels),
	# 	'中位数': np.median(roi_pixels),
	# 	'标准差': np.std(roi_pixels),
	# 	'最小值': np.min(roi_pixels),
	# 	'最大值': np.max(roi_pixels),
	# 	'范围': np.ptp(roi_pixels),
	# 	'偏度': skew(roi_pixels),
	# 	'峰度': kurtosis(roi_pixels),
	# 	'第10百分位数': np.percentile(roi_pixels, 10),
	# 	'第25百分位数': np.percentile(roi_pixels, 25),
	# 	'第50百分位数': np.percentile(roi_pixels, 50),
	# 	'第75百分位数': np.percentile(roi_pixels, 75),
	# 	'第90百分位数': np.percentile(roi_pixels, 90),
	# 	'第95百分位数': np.percentile(roi_pixels, 95),
	# 	'方差': np.var(roi_pixels)
	# 	}
	# data.at[file_name, 'c_mean'] = np.mean(roi_pixels)
	# data.at[file_name, 'c_max'] = np.max(roi_pixels)
	# data.at[file_name, 'c_min'] = np.min(roi_pixels)
	# data.at[file_name, 'c_p10'] = np.percentile(roi_pixels,10)
	# data.at[file_name, 'c_p25'] = np.percentile(roi_pixels,25)
	# data.at[file_name, 'c_p50'] = np.percentile(roi_pixels,50)
	# data.at[file_name, 'c_p75'] = np.percentile(roi_pixels,75)
	# data.at[file_name, 'c_p90'] = np.percentile(roi_pixels,90)
	# data.at[file_name, 'c_p95'] = np.percentile(roi_pixels,95)
	# data.at[file_name, 'c_var'] = np.var(roi_pixels)
	# data.at[file_name, 'c_skew'] = skew(roi_pixels)
	# data.at[file_name, 'c_ptp'] = np.ptp(roi_pixels)
	# data.at[file_name, 'c_kurtosis'] = kurtosis(roi_pixels)
	# data.at[file_name, 'c_std'] = np.std(roi_pixels)
	# data.at[file_name, 'c_median'] = np.median(roi_pixels)

	data.at[file_name, 'phi_mean'] = np.mean(roi_pixels)
	data.at[file_name, 'phi_max'] = np.max(roi_pixels)
	data.at[file_name, 'phi_min'] = np.min(roi_pixels)
	data.at[file_name, 'phi_p10'] = np.percentile(roi_pixels,10)
	data.at[file_name, 'phi_p25'] = np.percentile(roi_pixels,25)
	data.at[file_name, 'phi_p50'] = np.percentile(roi_pixels,50)
	data.at[file_name, 'phi_p75'] = np.percentile(roi_pixels,75)
	data.at[file_name, 'phi_p90'] = np.percentile(roi_pixels,90)
	data.at[file_name, 'phi_p95'] = np.percentile(roi_pixels,95)
	data.at[file_name, 'phi_var'] = np.var(roi_pixels)
	data.at[file_name, 'phi_skew'] = skew(roi_pixels)
	data.at[file_name, 'phi_ptp'] = np.ptp(roi_pixels)
	data.at[file_name, 'phi_kurtosis'] = kurtosis(roi_pixels)
	data.at[file_name, 'phi_std'] = np.std(roi_pixels)
	data.at[file_name, 'phi_median'] = np.median(roi_pixels)

data.to_excel('tumor_all.xlsx')