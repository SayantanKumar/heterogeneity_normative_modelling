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
############### Relevant functions -------------------------
#-------------------------------------------------------

def count_true_deviations(temp_dev_mmvae, MRI_vol_cols, amyloid_SUVR_cols, tau_SUVR_cols):
    
    temp_dev_mmvae_ADNI = temp_dev_mmvae.loc[temp_dev_mmvae.dataset == 'ADNI'].reset_index(drop = True)
    temp_dev_mmvae_KARI = temp_dev_mmvae.loc[temp_dev_mmvae.dataset == 'KARI'].reset_index(drop = True)
    
    def count_true_dev(temp_dev_mmvae):
        count_mri_true_dev = pd.DataFrame(temp_dev_mmvae[MRI_vol_cols].apply(lambda x: (x < 0).sum(), axis=0), index = MRI_vol_cols).rename(columns = {0:'mri_true_dev'})
        count_mri_true_dev = (count_mri_true_dev/len(temp_dev_mmvae[MRI_vol_cols]))*100
        count_mri_true_dev.index = count_mri_true_dev.index.str.replace(r'_Nvol', '')

        count_amyloid_true_dev = pd.DataFrame(temp_dev_mmvae[amyloid_SUVR_cols].apply(lambda x: (x > 0).sum(), axis=0), index = amyloid_SUVR_cols).rename(columns = {0:'amyloid_true_dev'})
        count_amyloid_true_dev = (count_amyloid_true_dev/len(temp_dev_mmvae[amyloid_SUVR_cols]))*100
        count_amyloid_true_dev.index = count_amyloid_true_dev.index.str.replace(r'_Asuvr', '')

        count_tau_true_dev = pd.DataFrame(temp_dev_mmvae[tau_SUVR_cols].apply(lambda x: (x > 0).sum(), axis=0), index = tau_SUVR_cols).rename(columns = {0:'tau_true_dev'})
        count_tau_true_dev = (count_tau_true_dev/len(temp_dev_mmvae[tau_SUVR_cols]))*100
        count_tau_true_dev.index = count_mri_true_dev.index.str.replace(r'_Tsuvr', '')

        count_all_true_dev = pd.concat([count_mri_true_dev, count_amyloid_true_dev, count_tau_true_dev], axis = 1)
        
        return count_all_true_dev

    count_all_true_dev_ADNI = count_true_dev(temp_dev_mmvae_ADNI)
    count_all_true_dev_KARI = count_true_dev(temp_dev_mmvae_KARI)
    
    for dataset in ['ADNI','KARI']:
        
        df = temp_dev_mmvae.loc[temp_dev_mmvae.dataset == dataset].reset_index(drop = True)
        
        plt.subplots(figsize = (24,5))
        plt.subplot(1,3,1)
        plt.hist((df[MRI_vol_cols].apply(lambda x: (x < 0).sum(), axis=1))/90*100, bins=100, color='blue', alpha=0.5, label='MRI')
        plt.xlabel('% of mri regions', fontsize = 18)
        plt.ylabel('Frequency', fontsize = 18)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.title('Percentage of {} MRI regions for each patient \n with positive deviations'.format(dataset), fontsize = 20)

        plt.subplot(1,3,2)
        plt.hist((df[amyloid_SUVR_cols].apply(lambda x: (x > 0).sum(), axis=1))/90*100, bins=100, color='red', alpha=0.5, label='Amyloid')
        plt.xlabel('% of amyloid regions', fontsize = 18)
        plt.ylabel('Frequency', fontsize = 18)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.title('Percentage of {} amyloid regions for each patient \n with positive deviations'.format(dataset), fontsize = 20)

        plt.subplot(1,3,3)
        plt.hist((df[tau_SUVR_cols].apply(lambda x: (x > 0).sum(), axis=1))/90*100, bins=100, color='green', alpha=0.5, label='Tau')
        plt.xlabel('% of tau regions', fontsize = 18)
        plt.ylabel('Frequency', fontsize = 18)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.title('Percentage of {} tau regions for each patient \n with positive deviations'.format(dataset), fontsize = 20)

        plt.tight_layout()
        plt.subplots_adjust(top = 0.9, hspace= 0.5, wspace = 0.3)
        #plt.suptitle('Pearson Correlation Coeffient between mean deviation and cognitive scores', fontsize = 40)
        #plt.show()
        
        if dataset == 'ADNI':
            plt.savefig('./Plots/count_true_deviations_ADNI.pdf', bbox_inches = 'tight', dpi = 600)

        if dataset == 'KARI':
            plt.savefig('./Plots/count_true_deviations_KARI.pdf', bbox_inches = 'tight', dpi = 600)


#-----------------------------------------------------------
#-----------------------------------------------------------


#-------------------------------------------------------
############### Main functions -------------------------
#-------------------------------------------------------

dev_mvae = pd.read_csv('./saved_dataframes/multimodal_deviations.csv')
temp_dev_mmvae = dev_mvae.copy()

with open("./saved_dataframes/MRI_vol_cols", "rb") as fp: 
    MRI_vol_cols = pickle.load(fp)
    
with open("./saved_dataframes/amyloid_SUVR_cols", "rb") as fp: 
    amyloid_SUVR_cols = pickle.load(fp)
    
with open("./saved_dataframes/tau_SUVR_cols", "rb") as fp: 
    tau_SUVR_cols = pickle.load(fp)


#--- Percentage of true deviations (negative for MRI and positive for amyloid/tau)
count_true_deviations(temp_dev_mmvae, MRI_vol_cols, amyloid_SUVR_cols, tau_SUVR_cols)


#------------------ Disease severity across disease stages ------------------
#---------------------------------------------------------------

from scipy.stats import norm

temp_dev_mmvae = dev_mvae.copy()
temp_dev_mmvae = temp_dev_mmvae.loc[(temp_dev_mmvae.stage == 'cdr = 0 amyloid negative') | (temp_dev_mmvae.amyloid_positive == 1)].reset_index(drop = True)
temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'cdr = 0 amyloid negative', 'stage'] = 'HC'
temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'cdr = 0 amyloid positive', 'stage'] = 'preclinical'

fs_cols = MRI_vol_cols + amyloid_SUVR_cols + tau_SUVR_cols

p_values_mri = pd.DataFrame(norm.cdf(temp_dev_mmvae[MRI_vol_cols]), columns = MRI_vol_cols)
p_values_amyloid = pd.DataFrame(1-norm.cdf(temp_dev_mmvae[amyloid_SUVR_cols]), columns = amyloid_SUVR_cols)
p_values_tau = pd.DataFrame(1-norm.cdf(temp_dev_mmvae[tau_SUVR_cols]), columns = tau_SUVR_cols)

p_values_all = pd.concat([p_values_mri, p_values_amyloid, p_values_tau], axis = 1)

adjusted_p_values = np.zeros_like(p_values_all)
for i in range(p_values_all.shape[1]):
    #adjusted_p_values[:, i] = multipletests(p_values_all.to_numpy()[:, i], method='fdr_bh')[1]
    adjusted_p_values[:, i] = fdrcorrection(p_values_all.to_numpy()[:, i], method='indep', alpha = 0.1)[1]
    
    
# rejected, adjusted_p_values = fdrcorrection(p_values_all.to_numpy().flatten(), alpha=0.1)
# adjusted_p_values = adjusted_p_values.reshape(p_values_all.shape)

adjusted_p_values = pd.DataFrame(adjusted_p_values, columns = fs_cols)

reject_corr = adjusted_p_values[fs_cols].copy()
for idx in adjusted_p_values.index.values:
    for col in fs_cols:
        if adjusted_p_values[fs_cols].loc[idx, col] < 0.1:
            reject_corr.loc[idx, col] = 1
        else :
            reject_corr.loc[idx, col] = 0

temp_dev_mmvae['mean_dev_mri_sig_fdr'] = abs(temp_dev_mmvae[MRI_vol_cols].mul(reject_corr[MRI_vol_cols]).mean(axis = 1))
temp_dev_mmvae['mean_dev_amyloid_sig_fdr'] = temp_dev_mmvae[amyloid_SUVR_cols].mul(reject_corr[amyloid_SUVR_cols]).mean(axis = 1)
temp_dev_mmvae['mean_dev_tau_sig_fdr'] = temp_dev_mmvae[tau_SUVR_cols].mul(reject_corr[tau_SUVR_cols]).mean(axis = 1)
temp_dev_mmvae['mean_dev_all_sig_fdr'] = (temp_dev_mmvae['mean_dev_mri_sig_fdr'] + temp_dev_mmvae['mean_dev_amyloid_sig_fdr'] + temp_dev_mmvae['mean_dev_tau_sig_fdr'])/3

for dataset in ['ADNI', 'KARI']:
    
    df = temp_dev_mmvae.loc[temp_dev_mmvae.dataset == dataset].reset_index(drop = True)
    
    plt.subplots(figsize = (18, 12))

    plt.subplot(2,2,1)
    sns.boxplot(x = 'stage', y = 'mean_dev_mri_sig_fdr', data = df, order = ['HC', 'preclinical', 'cdr = 0.5', 'cdr >= 1'], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black', 'markersize': 10})
#     add_stat_annotation(ax=plt.gca(), data=df, x='stage', y='mean_dev_mri_sig_fdr', order = ['HC', 'preclinical', 'cdr = 0.5', 'cdr >= 1'],
#                     box_pairs=[("HC", "preclinical"), ("preclinical", "cdr = 0.5"), ("cdr = 0.5", "cdr >= 1")],
#                     test='Kruskal', text_format='star', loc='inside', verbose=1)

    plt.xlabel('Disease stages', fontsize = 18)
    plt.ylabel('severity (Z-scores)', fontsize = 18)
    plt.yticks(fontsize = 16)
    plt.xticks(ticks = [0,1,2,3], labels=['CU', 'preclinical', 'cdr = 0.5', 'cdr >= 1'], fontsize = 16)
    #plt.legend(fontsize = 20)
    plt.title('{} disease severity (MRI atrophy)'.format(dataset), fontsize = 20)

    plt.subplot(2,2,2)
    sns.boxplot(x = 'stage', y = 'mean_dev_amyloid_sig_fdr', data = df, order = ['HC', 'preclinical', 'cdr = 0.5', 'cdr >= 1'], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black', 'markersize': 10})
#     add_stat_annotation(ax=plt.gca(), data=df, x='stage', y='mean_dev_amyloid_sig_fdr', order = ['HC', 'preclinical', 'cdr = 0.5', 'cdr >= 1'],
#                     box_pairs=[("HC", "preclinical"), ("preclinical", "cdr = 0.5"), ("cdr = 0.5", "cdr >= 1")],
#                     test='Kruskal', text_format='star', loc='inside', verbose=1)

    plt.xlabel('Disease stages', fontsize = 18)
    plt.ylabel('severity (Z-scores)', fontsize = 18)
    plt.yticks(fontsize = 16)
    #plt.legend(fontsize = 20)
    plt.xticks(ticks = [0,1,2,3], labels=['CU', 'preclinical', 'cdr = 0.5', 'cdr >= 1'], fontsize = 16)
    plt.title('{} disease severity (amyloid deposition)'.format(dataset), fontsize = 20)

    plt.subplot(2,2,3)
    sns.boxplot(x = 'stage', y = 'mean_dev_tau_sig_fdr', data = df, order = ['HC', 'preclinical', 'cdr = 0.5', 'cdr >= 1'], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black', 'markersize': 10})
#     add_stat_annotation(ax=plt.gca(), data=df, x='stage', y='mean_dev_tau_sig_fdr', order = ['HC', 'preclinical', 'cdr = 0.5', 'cdr >= 1'],
#                     box_pairs=[("HC", "preclinical"), ("preclinical", "cdr = 0.5"), ("cdr = 0.5", "cdr >= 1")],
#                     test='Kruskal', text_format='star', loc='inside', verbose=1)

    plt.xlabel('Disease stages', fontsize = 18)
    plt.ylabel('severity (Z-scores)', fontsize = 18)
    plt.yticks(fontsize = 16)
    #plt.legend(fontsize = 20)
    plt.xticks(ticks = [0,1,2,3], labels=['CU', 'preclinical', 'cdr = 0.5', 'cdr >= 1'], fontsize = 16)
    plt.title('{} disease severity (tau deposition)'.format(dataset), fontsize = 20)

    plt.subplot(2,2,4)
    sns.boxplot(x = 'stage', y = 'mean_dev_all_sig_fdr', data = df, order = ['HC', 'preclinical', 'cdr = 0.5', 'cdr >= 1'], showmeans=True,  meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black', 'markersize': 10})
#     add_stat_annotation(ax=plt.gca(), data=df, x='stage', y='mean_dev_all_sig_fdr', order = ['HC', 'preclinical', 'cdr = 0.5', 'cdr >= 1'],
#                     box_pairs=[("HC", "preclinical"), ("preclinical", "cdr = 0.5"), ("cdr = 0.5", "cdr >= 1")],
#                     test='Kruskal', text_format='star', loc='inside', verbose=1)

    plt.xlabel('Disease stages', fontsize = 18)
    plt.ylabel('severity (Z-scores)', fontsize = 18)
    plt.yticks(fontsize = 16)
    #plt.legend(fontsize = 20)
    plt.xticks(ticks = [0,1,2,3], labels=['CU', 'preclinical', 'cdr = 0.5', 'cdr >= 1'], fontsize = 16)
    plt.title('{} disease severity (across all modalities)'.format(dataset), fontsize = 20)

    plt.tight_layout()
    plt.suptitle('Patient-level disease severity across {} groups'.format(dataset), fontsize = 24)
    plt.subplots_adjust(top = 0.9, hspace= 0.4, wspace = 0.2)
    #plt.show()
    
    if dataset == 'ADNI':
        plt.savefig('./Plots/Patient_severity_ADNI.pdf', bbox_inches = 'tight', dpi = 600)

    if dataset == 'KARI':
        plt.savefig('./Plots/Patient_severity_KARI.pdf', bbox_inches = 'tight', dpi = 600)


#---------------------------------------------------------------
#-------- Compare disease severity across different AD groups
#---------------------------------------------------------------

from scipy import stats
from scipy.stats import f_oneway

def calculate_stats(array):
    
    median = np.mean(array)
    
    q75, q25 = np.percentile(array, [75 ,25])
    iqr = q75 - q25

    ci = stats.t.interval(0.95, len(array)-1, loc=np.mean(array), scale=stats.sem(array))
    
    print('Mean = {}, IQR = {}, 95% CI = {}'.format(median, iqr, ci))
    #return median, iqr, ci

calculate_stats(temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'HC'].mean_dev_all_sig_fdr)
calculate_stats(temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'preclinical'].mean_dev_all_sig_fdr)
calculate_stats(temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'cdr = 0.5'].mean_dev_all_sig_fdr)
calculate_stats(temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'cdr >= 1'].mean_dev_all_sig_fdr)

def compare_group_means(*groups):
    f_value, p_value = f_oneway(*groups)
    return f_value, p_value

f_value_mri, p_value_mri = f_oneway(temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'HC']['mean_dev_mri_sig_fdr'].values, temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'preclinical']['mean_dev_mri_sig_fdr'].values, temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'cdr = 0.5']['mean_dev_mri_sig_fdr'].values, temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'cdr >= 1']['mean_dev_mri_sig_fdr'].values)
f_value_amyloid, p_value_amyloid = f_oneway(temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'HC']['mean_dev_amyloid_sig_fdr'].values, temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'preclinical']['mean_dev_amyloid_sig_fdr'].values, temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'cdr = 0.5']['mean_dev_amyloid_sig_fdr'].values, temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'cdr >= 1']['mean_dev_amyloid_sig_fdr'].values)
f_value_tau, p_value_tau = f_oneway(temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'HC']['mean_dev_tau_sig_fdr'].values, temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'preclinical']['mean_dev_tau_sig_fdr'].values, temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'cdr = 0.5']['mean_dev_tau_sig_fdr'].values, temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'cdr >= 1']['mean_dev_tau_sig_fdr'].values)


#f_value_mri, p_value_mri = f_oneway(temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'HC']['mean_dev_amyloid_sig_fdr'].values, temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'cdr = 0.5']['mean_dev_amyloid_sig_fdr'].values)

#print("F-value MRI:", f_value_mri)
print("P-value MRI:", p_value_mri)
print("P-value amyloid:", p_value_amyloid)
print("P-value tau:", p_value_tau)



#---------------------------------------------------------------
#Total outlier count (Number of regions for each patient with statistically signficant deviations)
#---------------------------------------------------------------

fs_cols = MRI_vol_cols + amyloid_SUVR_cols + tau_SUVR_cols

#temp_dev_mmvae = dev_bvae.copy()

temp_dev_mmvae['mri_toc'] = temp_dev_mmvae[MRI_vol_cols].apply(lambda x: (norm.cdf(x) < 0.1).sum(), axis=1)/90*100
temp_dev_mmvae['amyloid_toc'] = temp_dev_mmvae[amyloid_SUVR_cols].apply(lambda x: (1 - norm.cdf(x) < 0.1).sum(), axis=1)/90*100
temp_dev_mmvae['tau_toc'] = temp_dev_mmvae[tau_SUVR_cols].apply(lambda x: (1 - norm.cdf(x) < 0.1).sum(), axis=1)/90*100
temp_dev_mmvae['all_toc'] = (temp_dev_mmvae['mri_toc'] + temp_dev_mmvae['amyloid_toc'] + temp_dev_mmvae['tau_toc'])/3

#-----------------------------------------------

# p_values_all = pd.DataFrame(1 - norm.cdf(abs(temp_dev_mmvae[fs_cols]).to_numpy()), columns = fs_cols)
# adjusted_p_values = np.zeros_like(p_values_all)
# for i in range(p_values_all.shape[1]):
#     #adjusted_p_values[:, i] = multipletests(p_values_all.to_numpy()[:, i], method='fdr_bh')[1]
#     adjusted_p_values[:, i] = fdrcorrection(p_values_all.to_numpy()[:, i], method='indep')[1]
    
# adjusted_p_values = pd.DataFrame(adjusted_p_values, columns = fs_cols)

# reject_corr = adjusted_p_values[fs_cols].copy()
# for idx in adjusted_p_values.index.values:
#     for col in fs_cols:
#         if adjusted_p_values[fs_cols].loc[idx, col] < 0.05:
#             reject_corr.loc[idx, col] = 1
#         else :
#             reject_corr.loc[idx, col] = np.nan
            
reject_corr['mri_toc_fdr'] = reject_corr[MRI_vol_cols].apply(lambda x: (x == 1).sum(), axis=1)/90*100
reject_corr['amyloid_toc_fdr'] = reject_corr[amyloid_SUVR_cols].apply(lambda x: (x == 1).sum(), axis=1)/90*100
reject_corr['tau_toc_fdr'] = reject_corr[tau_SUVR_cols].apply(lambda x: (x == 1).sum(), axis=1)/90*100
#reject_corr['all_toc_fdr'] = (reject_corr['mri_toc_fdr'] + reject_corr['mri_toc_fdr'] + reject_corr['mri_toc_fdr'])/3


# reject_corr['mri_toc_fdr'] = (reject_corr[MRI_vol_cols].sum(axis = 1))/90*100
# reject_corr['amyloid_toc_fdr'] = (reject_corr[amyloid_SUVR_cols].sum(axis = 1))/90*100
# reject_corr['tau_toc_fdr'] = (reject_corr[tau_SUVR_cols].sum(axis = 1))/90*100
reject_corr['all_toc_fdr'] = (reject_corr[fs_cols].sum(axis = 1))/270*100


non_roi_cols = [col for col in temp_dev_mmvae.columns if col not in fs_cols]
reject_corr[non_roi_cols] = temp_dev_mmvae[non_roi_cols]

# mri_roi_sig = pd.DataFrame(temp_dev_mmvae[MRI_vol_cols].apply(lambda x: (scipy.stats.norm.sf(abs(x)) < 0.1).sum(), axis=0), index = MRI_vol_cols).rename(columns = {0:'sig_count_mri'})
# amyloid_roi_sig = pd.DataFrame(temp_dev_mmvae[amyloid_SUVR_cols].apply(lambda x: (scipy.stats.norm.sf(abs(x)) < 0.1).sum(), axis=0), index = amyloid_SUVR_cols).rename(columns = {0:'sig_count_amyloid'})
# tau_roi_sig = pd.DataFrame(temp_dev_mmvae[tau_SUVR_cols].apply(lambda x: (scipy.stats.norm.sf(abs(x)) < 0.1).sum(), axis=0), index = tau_SUVR_cols).rename(columns = {0:'sig_count_tau'})

#total_outlier_count(temp_dev_mmvae, reject_corr, MRI_vol_cols, amyloid_SUVR_cols, tau_SUVR_cols)

for dataset in ['ADNI', 'KARI']:
    
    df = temp_dev_mmvae.loc[temp_dev_mmvae.dataset == dataset].reset_index(drop = True)
    plt.subplots(figsize = (20, 12))

    plt.subplot(2,2,1)
    sns.boxplot(x = 'stage', y = 'mri_toc', data = df, order = ['HC', 'preclinical', 'cdr = 0.5', 'cdr >= 1'], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
    plt.xlabel('CDR category', fontsize = 18)
    plt.ylabel('Proportion (%) of outlier \n MRI regions', fontsize = 18)
    plt.yticks(fontsize = 16)
    plt.xticks(ticks = [0,1,2,3], labels=['HC', 'preclinical', 'cdr = 0.5', 'cdr >= 1'], fontsize = 16)
    plt.title('Proportion (%) of regions with significant MRI deviations', fontsize = 18)
    
    plt.subplot(2,2,2)
    sns.boxplot(x = 'stage', y = 'amyloid_toc', data = df, order = ['HC', 'preclinical', 'cdr = 0.5', 'cdr >= 1'], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
    plt.xlabel('CDR category', fontsize = 18)
    plt.ylabel('Proportion (%) of outlier \n amyloid regions', fontsize = 18)
    plt.yticks(fontsize = 16)
    plt.xticks(ticks = [0,1,2,3], labels=['HC', 'preclinical', 'cdr = 0.5', 'cdr >= 1'], fontsize = 16)
    plt.title('Proportion (%) of regions with significant amyloid deviations', fontsize = 18) 

    plt.subplot(2,2,3)
    sns.boxplot(x = 'stage', y = 'tau_toc', data = df, order = ['HC', 'preclinical', 'cdr = 0.5', 'cdr >= 1'], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
    plt.xlabel('CDR category', fontsize = 18)
    plt.ylabel('Proportion (%) of outlier \n tau regions', fontsize = 18)
    plt.yticks(fontsize = 16)
    plt.xticks(ticks = [0,1,2,3], labels=['HC', 'preclinical', 'cdr = 0.5', 'cdr >= 1'], fontsize = 16)
    plt.title('Proportion (%) of regions with significant tau deviations', fontsize = 18) 

    plt.subplot(2,2,4)
    sns.boxplot(x = 'stage', y = 'all_toc', data = df, order = ['HC', 'preclinical', 'cdr = 0.5', 'cdr >= 1'], showmeans=True, meanprops={'marker':'*', 'markerfacecolor':'white', 'markeredgecolor':'black'})
    plt.xlabel('CDR category', fontsize = 18)
    plt.ylabel('Proportion (%) of outlier \n regions (all modalities)', fontsize = 18)
    plt.yticks(fontsize = 16)
    plt.xticks(ticks = [0,1,2,3], labels=['HC', 'preclinical', 'cdr = 0.5', 'cdr >= 1'], fontsize = 16)
    plt.title('Proportion (%) of regions with significant deviations (all modalities)', fontsize = 18) 

    plt.tight_layout()
    plt.suptitle('Number of regions (all modalities) with statistically signficant deviations before and after FDR correction ({})'.format(dataset), fontsize = 24)
    plt.subplots_adjust(top = 0.9, hspace= 0.4, wspace = 0.2)
    #plt.show()
    
    if dataset == 'ADNI':
        plt.savefig('./Plots/toc_ADNI.pdf', bbox_inches = 'tight', dpi = 600)

    if dataset == 'KARI':
        plt.savefig('./Plots/toc_KARI.pdf', bbox_inches = 'tight', dpi = 600)


    
temp_dev_mmvae.to_csv('./saved_dataframes/deviations_mmvae.csv')
reject_corr.to_csv('./saved_dataframes/sig_deviations.csv')