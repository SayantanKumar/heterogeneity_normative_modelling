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

#-------------------------------------------------------
############### -------------------------
#-------------------------------------------------------

reject_corr = pd.read_csv('./saved_dataframes/sig_deviations.csv')
temp_dev_mmvae = pd.read_csv('./saved_dataframes/deviations_mmvae.csv')
all_atn_adni = pd.read_csv('./saved_dataframes/all_atn_adni.csv')

with open("./saved_dataframes/MRI_vol_cols", "rb") as fp: 
    MRI_vol_cols = pickle.load(fp)
    
with open("./saved_dataframes/amyloid_SUVR_cols", "rb") as fp: 
    amyloid_SUVR_cols = pickle.load(fp)
    
with open("./saved_dataframes/tau_SUVR_cols", "rb") as fp: 
    tau_SUVR_cols = pickle.load(fp)

    
#-----------------------------------------------------------------
############Association between ADNI cognition and mean deviations.
############Regression lines stratified by disease categories 
#----------------------------------------------------------------

# Correlation of deviation maps with cog
#cog_adni_test = prepare_adni_cognition(tadpole_challenge_path, temp_dev_mmvae)

# dev_bvae_adni_cog = cog_correlation_adni(temp_dev_mmvae, cog_adni_test)

# temp_dev_mmvae['mean_dev_mri_true'] = abs(temp_dev_mmvae[MRI_vol_cols].apply(lambda row: row[lambda x: x < 0].mean(), axis=1))
# temp_dev_mmvae['mean_dev_amyloid_true'] = temp_dev_mmvae[MRI_vol_cols].apply(lambda row: row[lambda x: x > 0].mean(), axis=1)
# temp_dev_mmvae['mean_dev_tau_true'] = temp_dev_mmvae[MRI_vol_cols].apply(lambda row: row[lambda x: x > 0].mean(), axis=1)

comp_cog_adni = pd.read_csv('/Users/sayantankumar/Desktop/Aris_Work/Data/ADNI/UWNPSYCHSUM_01_23_23_13May2023.csv')
cog_adni_comp = pd.merge(comp_cog_adni, all_atn_adni[['RID', 'mri_date', 'amyloid_date', 'tau_date']], on = 'RID', how = 'right').sort_values(by = ['RID', 'EXAMDATE']).reset_index(drop = True)
cog_adni_comp['date_diff'] = abs(pd.to_datetime(cog_adni_comp['mri_date']) - pd.to_datetime(cog_adni_comp['EXAMDATE'])).dt.days

cog_adni_comp = cog_adni_comp.sort_values(by = ['RID', 'date_diff']).drop_duplicates(subset = 'RID', keep = 'first')
cog_adni_test_comp = cog_adni_comp.loc[cog_adni_comp.RID.isin(temp_dev_mmvae.ID.values)].reset_index(drop = True)

cog_adni_test_comp['ADNI_MEM'] = cog_adni_test_comp['ADNI_MEM'].fillna(cog_adni_test_comp['ADNI_MEM'].mean())
cog_adni_test_comp['ADNI_EF'] = cog_adni_test_comp['ADNI_EF'].fillna(cog_adni_test_comp['ADNI_EF'].mean())
cog_adni_test_comp['ADNI_LAN'] = cog_adni_test_comp['ADNI_LAN'].fillna(cog_adni_test_comp['ADNI_LAN'].mean())
cog_adni_test_comp['ADNI_VS'] = cog_adni_test_comp['ADNI_VS'].fillna(cog_adni_test_comp['ADNI_VS'].mean())


from scipy.stats import pearsonr
from scipy.stats import spearmanr
    
dev_bvae_adni = reject_corr.loc[temp_dev_mmvae.dataset == 'ADNI'].reset_index(drop = True)
dev_bvae_adni_cog = pd.merge(dev_bvae_adni, cog_adni_test_comp[['RID', 'ADNI_MEM', 'ADNI_EF', 'ADNI_LAN', 'ADNI_VS']], left_on = 'ID', right_on = 'RID', how = 'inner')

cog_cols = ['ADNI_MEM', 'ADNI_EF', 'ADNI_LAN', 'ADNI_VS']


pearson_all_dev = []
pearson_mri_dev = []
pearson_amyloid_dev = []
pearson_tau_dev = []

pearson_all_toc = []
pearson_mri_toc = []
pearson_amyloid_toc = []
pearson_tau_toc = []

for cog in cog_cols:
    
    corr_p_all_dev, _ = pearsonr(dev_bvae_adni_cog[cog], dev_bvae_adni_cog['mean_dev_all_sig_fdr'])
    corr_p_mri_dev, _ = pearsonr(dev_bvae_adni_cog[cog], dev_bvae_adni_cog['mean_dev_mri_sig_fdr'])
    corr_p_amyloid_dev, _ = pearsonr(dev_bvae_adni_cog[cog], dev_bvae_adni_cog['mean_dev_amyloid_sig_fdr'])
    corr_p_tau_dev, _ = pearsonr(dev_bvae_adni_cog[cog], dev_bvae_adni_cog['mean_dev_tau_sig_fdr'])

    pearson_all_dev.append(corr_p_all_dev)
    pearson_mri_dev.append(corr_p_mri_dev)
    pearson_amyloid_dev.append(corr_p_amyloid_dev)
    pearson_tau_dev.append(corr_p_tau_dev)

    R2_pearson_all_dev = [i**2 for i in pearson_all_dev]
    R2_pearson_mri_dev = [i**2 for i in pearson_mri_dev]
    R2_pearson_amyloid_dev = [i**2 for i in pearson_amyloid_dev]
    R2_pearson_tau_dev = [i**2 for i in pearson_tau_dev]
    
fig, axs = plt.subplots(len(cog_cols), 4, figsize = (40, 36))
j = 0
for cog in cog_cols:

    #------------ All modalities ------------------------
    m, b = np.polyfit(dev_bvae_adni_cog[cog], dev_bvae_adni_cog['mean_dev_all_sig_fdr'], 1)
    axs[j,0].plot(dev_bvae_adni_cog[cog], m*dev_bvae_adni_cog[cog] + b, color = 'red')

    axs[j,0].scatter(dev_bvae_adni_cog[cog], dev_bvae_adni_cog['mean_dev_all_sig_fdr'])
    axs[j,0].tick_params(axis='x', labelsize=24)
    axs[j,0].tick_params(axis='y', labelsize=24)
    axs[j,0].set_xlabel(cog, fontsize = 24)
    axs[j,0].set_ylabel('Disease severity \n all modalities', fontsize = 24)
    axs[j,0].set_title('All modalities \n $r$ = {}'.format(round(pearson_all_dev[j], 3)), fontsize = 32)

    #------------ MRI ------------------------
    m, b = np.polyfit(dev_bvae_adni_cog[cog], dev_bvae_adni_cog['mean_dev_mri_sig_fdr'], 1)
    axs[j,1].plot(dev_bvae_adni_cog[cog], m*dev_bvae_adni_cog[cog] + b, color = 'red')

    axs[j,1].scatter(dev_bvae_adni_cog[cog], dev_bvae_adni_cog['mean_dev_mri_sig_fdr'])
    axs[j,1].set_xlabel(cog, fontsize = 24)
    axs[j,1].tick_params(axis='x', labelsize=24)
    axs[j,1].tick_params(axis='y', labelsize=24)
    axs[j,1].set_ylabel('Disease severity \n MRI', fontsize = 24)
    axs[j,1].set_title('MRI $r$ = {}'.format(round(pearson_mri_dev[j], 3)), fontsize = 32)

    #------------ Amyloid ------------------------
    m, b = np.polyfit(dev_bvae_adni_cog[cog], dev_bvae_adni_cog['mean_dev_amyloid_sig_fdr'], 1)
    axs[j,2].plot(dev_bvae_adni_cog[cog], m*dev_bvae_adni_cog[cog] + b, color = 'red')

    axs[j,2].scatter(dev_bvae_adni_cog[cog], dev_bvae_adni_cog['mean_dev_amyloid_sig_fdr'])
    axs[j,2].set_xlabel(cog, fontsize = 24)
    axs[j,2].tick_params(axis='x', labelsize=24)
    axs[j,2].tick_params(axis='y', labelsize=24)
    axs[j,2].set_ylabel('Disease severity \n amyloid', fontsize = 24)
    axs[j,2].set_title('Amyloid $r$ = {}'.format(round(pearson_amyloid_dev[j], 3)), fontsize = 32)

    #------------ Tau ------------------------
    m, b = np.polyfit(dev_bvae_adni_cog[cog], dev_bvae_adni_cog['mean_dev_tau_sig_fdr'], 1)
    axs[j,3].plot(dev_bvae_adni_cog[cog], m*dev_bvae_adni_cog[cog] + b, color = 'red')

    axs[j,3].scatter(dev_bvae_adni_cog[cog], dev_bvae_adni_cog['mean_dev_tau_sig_fdr'])
    axs[j,3].set_xlabel(cog, fontsize = 24)
    axs[j,3].tick_params(axis='x', labelsize=24)
    axs[j,3].tick_params(axis='y', labelsize=24)
    axs[j,3].set_ylabel('Disease severity \n tau', fontsize = 24)
    axs[j,3].set_title('Tau $r$ = {}'.format(round(pearson_tau_dev[j], 3)), fontsize = 32)


    if cog == 'RAVLT_immediate':
        axs[j,0].set_xlabel('RAVLT', fontsize = 24)
        axs[j,1].set_xlabel('RAVLT', fontsize = 24)
        axs[j,2].set_xlabel('RAVLT', fontsize = 24)
        axs[j,3].set_xlabel('RAVLT', fontsize = 24)

    #fig.suptitle('Pearson Correlation Coeffient between mean deviation and cognitive scores', fontsize = 24)

    j = j + 1

plt.tight_layout()
plt.subplots_adjust(top = 0.9, hspace= 0.7, wspace = 0.3)
plt.suptitle('Pearson Correlation Coeffient between mean deviation and cognitive scores', fontsize = 40)
plt.savefig('./Plots/cognition_deviation_association.pdf', bbox_inches = 'tight', dpi = 600)

#-----------------------------------------------------------------
######## Generic function to fit regression and estimate correlation 
######### between cognition and severity values
#----------------------------------------------------------------

def plot_cognition_correlation(dev_bvae_adni_cog, modality, x_axis = '', y_axis = ''):
    
    #plt.figure(figsize = (6,4))
    
    pearson = []
    corr_p, _ = pearsonr(dev_bvae_adni_cog[x_axis], dev_bvae_adni_cog[y_axis])
    pearson.append(corr_p)
    R2_pearson = [i**2 for i in pearson]
    
    m, b = np.polyfit(dev_bvae_adni_cog[x_axis], dev_bvae_adni_cog[y_axis], 1)
    plt.plot(dev_bvae_adni_cog[x_axis], m*dev_bvae_adni_cog[x_axis] + b, color = 'red')

    plt.scatter(dev_bvae_adni_cog[x_axis], dev_bvae_adni_cog[y_axis])
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.xlabel(x_axis, fontsize = 18)
    plt.ylabel('Disease severity \n {}'.format(modality), fontsize = 18)
    plt.title('{} $r$ = {}'.format(modality, round(corr_p, 3)), fontsize = 18)

plt.subplots(figsize = (16,4))
plt.subplot(1,2,1)
plot_cognition_correlation(dev_bvae_adni_cog, 'Amyloid', x_axis = 'ADNI_MEM', y_axis = 'mean_dev_amyloid_sig_fdr')
plt.subplot(1,2,2)
plot_cognition_correlation(dev_bvae_adni_cog, 'Amyloid', x_axis = 'ADNI_EF', y_axis = 'mean_dev_amyloid_sig_fdr')

plt.tight_layout()
plt.subplots_adjust(top = 0.9, hspace= 0.7, wspace = 0.3)
plt.show()
#fig.suptitle('Pearson Correlation Coeffient between mean deviation and cognitive scores', fontsize = 60)


def calculate_regression_coefficients(df, cog_col, severity_col):
    
    import statsmodels.api as sm

    #df = dev_bvae_adni_cog.copy()
    df['intercept'] = 1

    X = df[['intercept', cog_col]]
    y = df[severity_col]

    model = sm.OLS(y, X)
    results = model.fit()

    beta = results.params[1]
    p_value = results.pvalues[1]

    print("Beta coefficient:", beta)
    print("P-value:", p_value)
    
    return beta, p_value

#Example
calculate_regression_coefficients(dev_bvae_adni_cog, 'ADNI_MEM', 'mean_dev_tau_sig_fdr')