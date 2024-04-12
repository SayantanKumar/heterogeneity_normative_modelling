import numpy as np
import pandas as pd
import os
import warnings
import ggseg_python
import re
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#import neuroHarmonize
from neuroHarmonize import harmonizationLearn, harmonizationApply, loadHarmonizationModel, saveHarmonizationModel

import seaborn as sns
from functools import reduce
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold, StratifiedKFold

import scipy
import statsmodels
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.multitest import fdrcorrection_twostage

from data_utils import *
from ADNI_KARI_merge_compare import *
from harmonize_combat import *

from dataloaders import *
from multimodal_VAE import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torchvision import datasets, transforms
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer

#--------------------------------------------------------------
#--------------------------------------------------------------

a_n_merged_harm = pd.read_csv('./saved_dataframes/a_n_merged_harm.csv')
a_n_merged = pd.read_csv('./saved_dataframes/a_n_merged.csv')

roi_demo_both = a_n_merged_harm.loc[a_n_merged_harm.tau_present == 'yes'].reset_index(drop = True)

roi_demo_both.loc[roi_demo_both.Sex == 0, 'Sex'] = 'Female'
roi_demo_both.loc[roi_demo_both.Sex == 1, 'Sex'] = 'Male'
roi_demo_both['Age'] = round(roi_demo_both['Age'])

age_df = pd.DataFrame(one_hot_encoding(roi_demo_both, 'Age'), index = roi_demo_both.index.values)
age_mat = one_hot_encoding(roi_demo_both, 'Age')

sex_df = pd.DataFrame(one_hot_encoding(roi_demo_both, 'Sex'), index = roi_demo_both.index.values)
sex_mat = one_hot_encoding(roi_demo_both, 'Sex')

site_df = pd.DataFrame(one_hot_encoding(roi_demo_both, 'dataset'), index = roi_demo_both.index.values)
site_mat = one_hot_encoding(roi_demo_both, 'dataset')

combine_arr = []
age_sex = []
for i in range(roi_demo_both.shape[0]):
    age_sex_val = np.matmul(sex_mat[i].reshape(-1,1), age_mat[i].reshape(-1,1).T).flatten().reshape(-1,1)
    age_sex.append(age_sex_val)
    
    age_sex_site_val = np.matmul(age_sex[i], site_mat[i].reshape(-1,1).T).flatten().reshape(-1,1)
    combine_arr.append(age_sex_site_val)

combine_mat = np.hstack(np.array(combine_arr)).T

age_sex_df = pd.DataFrame(np.hstack(np.array(age_sex)).T, index = roi_demo_both.index.values)


#age_sex_site_df = pd.DataFrame(combine_mat, index = roi_demo_both.index.values)

age_sex_site_df = pd.DataFrame(np.concatenate((age_mat, sex_mat, site_mat), axis=1), index = roi_demo_both.index.values)

#------------------------------------------------

only_CN = roi_demo_both.loc[roi_demo_both.stage == 'cdr = 0 amyloid negative']
rest = roi_demo_both.loc[roi_demo_both['stage'] != 'cdr = 0 amyloid negative']

y_CN = only_CN['stage']
y_rest = rest['stage']

only_CN_test = only_CN.sample(n=round(0.15*len(only_CN)), random_state=1)
CN_model, CN_held_val = train_test_split(only_CN.loc[~only_CN.ID.isin(only_CN_test.ID.values)], test_size=0.2, shuffle = False, random_state = 1000)

X_test_org = pd.concat([rest, only_CN_test]).copy().reset_index(drop = True)

print('Number of CN used for model training/val: {}'.format(len(CN_model)))
print('Number of CN used for normalization: {}'.format(len(CN_held_val)))
print('Number of CN used in test set: {}'.format(len(only_CN_test)))
print('Number of disease patients used in test set: {}'.format(len(rest)))

#----------------------------------------------------------------------
#######################################################################
#----------------------------------------------------------------------

with open("./saved_dataframes/MRI_vol_cols", "rb") as fp: 
    MRI_vol_cols = pickle.load(fp)
    
with open("./saved_dataframes/amyloid_SUVR_cols", "rb") as fp: 
    amyloid_SUVR_cols = pickle.load(fp)
    
with open("./saved_dataframes/tau_SUVR_cols", "rb") as fp: 
    tau_SUVR_cols = pickle.load(fp)


k = 1
params = {'batch_size':64, 'lr':5e-5, 'latent_dim':32, 'alpha_1':1, 'alpha_2':1, 'alpha_3':1, 'beta':1, 'epochs':1000}

common_cols = ['ID', 'Age', 'Sex', 'tau_present', 'amyloid_positive', 'amyloid_centiloid', 'cdr', 'stage', 'dataset']

model, train_loss, val_loss, X_org_val, X_pred_val, dev_mvae, X_valho_org, X_pred_valho, X_pred_test, X_test_total = complete_training_prediction(CN_model, MRI_vol_cols, amyloid_SUVR_cols, tau_SUVR_cols, common_cols, age_sex_site_df, params)


# #----------------------------------------------
# cat = ['cdr = 0 amyloid negative', 'cdr = 0 amyloid positive', 'cdr = 0.5', 'cdr >= 1']
# slope = calculate_slope_model(cat, dev_mvae)

# temp_dev_bvae = dev_mvae.copy()
# temp_dev_bvae['label'] = 'mmVAE' + '(slope = ' + str(slope) + ')'
# plt.figure(figsize = (10,5))
# sns.boxplot(x = 'stage', y = 'mean_dev_all', hue = 'label', data = temp_dev_bvae, order = ['cdr = 0 amyloid negative', 'cdr = 0 amyloid positive', 'cdr = 0.5', 'cdr >= 1'], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
# plt.xlabel('Disease category', fontsize = 16)
# plt.ylabel('Mean deviation (Z-scores) \n across all brain regions', fontsize = 16)
# plt.yticks(fontsize = 16)
# plt.xticks(ticks = [0,1,2,3], labels=['cdr = 0 \n amyloid negative', 'cdr = 0 \n amyloid positive', 'cdr = 0.5', 'cdr >= 1'], fontsize = 14)
# plt.legend(fontsize = 16)
# plt.title('Disease staging (multimodal)', fontsize = 18)

# fs_cols = MRI_vol_cols + amyloid_SUVR_cols + tau_SUVR_cols
# recon['mean_recon_all'] = (abs((recon[fs_cols])).sum(axis = 1).to_numpy()/recon[fs_cols].shape[1])
# plt.figure(figsize = (10,4))
# sns.boxplot(x = 'stage', y = 'mean_recon_all', data = recon, order = ['cdr = 0 amyloid negative', 'cdr = 0 amyloid positive', 'cdr = 0.5', 'cdr >= 1'], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
# plt.xlabel('Disease category', fontsize = 16)
# plt.ylabel('Mean reconstruction loss \n for all modalities', fontsize = 16)
# plt.yticks(fontsize = 16)
# plt.xticks(ticks = [0,1,2,3], labels=['cdr = 0 \n amyloid negative', 'cdr = 0 \n amyloid positive', 'cdr = 0.5', 'cdr >= 1'], fontsize = 14)
# plt.title('Mean reconstruction loss \n (all modalities)', fontsize = 18)

plt.figure(figsize = (7,5))
epochs = range(len(train_loss))
plt.plot(epochs, train_loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation loss')
plt.legend()


dev_mvae.to_csv('./saved_dataframes/multimodal_deviations.csv')