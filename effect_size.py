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

def plot_effect_size(temp_dev_mmvae, modality_cols, modality, strip_suffix_col, dataset, value_range):
    
    from scipy.stats import f_oneway

    temp_dev_mmvae = temp_dev_mmvae.loc[temp_dev_mmvae.dataset == dataset].reset_index(drop = True)

    sig_cohen_hc_precl, sig_cohen_hc_cdr05, sig_cohen_hc_cdr1 = calculate_effect_size(temp_dev_mmvae, modality_cols)

    sig_cohen_hc_precl = [abs(x) for x in sig_cohen_hc_precl]
    sig_cohen_hc_cdr05 = [abs(x) for x in sig_cohen_hc_cdr05]
    sig_cohen_hc_cdr1 = [abs(x) for x in sig_cohen_hc_cdr1]
    
    ggseg_cols = [element.rstrip(strip_suffix_col) for element in modality_cols]
    
    effect_hc_precl = {key: value for key, value in dict(zip(ggseg_cols, sig_cohen_hc_precl)).items() if value != 0}
    effect_hc_cdr05 = {key: value for key, value in dict(zip(ggseg_cols, sig_cohen_hc_cdr05)).items() if value != 0}
    effect_hc_cdr1 = {key: value for key, value in dict(zip(ggseg_cols, sig_cohen_hc_cdr1)).items() if value != 0}
    
    def plot_maps(effect_list, sig_cohen_list, modality, fig_title, value_range):
        
        ggseg_python.plot_dk(effect_list, cmap='hot',
                          background='w', edgecolor='k', bordercolor='gray', figsize = (6,4), fontsize = 18, vminmax = value_range, 
                          ylabel='Effect size', title=fig_title)
        print('Max = {}, Min = {}'.format(max(sig_cohen_list), min(sig_cohen_list)))


        ggseg_python.plot_aseg(effect_list, cmap='hot',
                          background='w', edgecolor='k', bordercolor='gray', figsize = (6,4), fontsize = 18, vminmax = value_range, 
                          ylabel='Effect size', title=fig_title)
        print('Max = {}, Min = {}'.format(max(sig_cohen_list), min(sig_cohen_list)))

    
    plot_maps(effect_hc_precl, sig_cohen_hc_precl, modality, 'HC vs precl ({} {} deviations)'.format(dataset, modality), value_range)
    plot_maps(effect_hc_cdr05, sig_cohen_hc_cdr05, modality, 'HC vs cdr=0.5 ({} {} deviations)'.format(dataset, modality), value_range)
    plot_maps(effect_hc_cdr1, sig_cohen_hc_cdr1, modality, 'HC vs cdr>=1 ({} {} deviations)'.format(dataset, modality), value_range)
    
    #return sig_cohen_hc_precl, sig_cohen_hc_cdr05, sig_cohen_hc_cdr1, effect_hc_precl, effect_hc_cdr05, effect_hc_cdr1 

    
    
#--------------------------------------
#--------------------------------------

def cohen_d(group1, group2):
    
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    std1 = np.std(group1, ddof=1)
    std2 = np.std(group2, ddof=1)
    n1 = len(group1)
    n2 = len(group2)

    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std
    
    return cohens_d


def calculate_effect_size(temp_dev_mmvae, MRI_vol_cols):
    
#     dev_hc = temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'cdr = 0 amyloid negative']
#     dev_precl = temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'cdr = 0 amyloid positive']
#     dev_cdr05 = temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'cdr = 0.5']
#     dev_cdr1 = temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'cdr >= 1']
    
    dev_hc = (temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'HC'][MRI_vol_cols])
    dev_precl = (temp_dev_mmvae.loc[temp_dev_mmvae.stage == 'preclinical'][MRI_vol_cols])
    dev_cdr05 = (temp_dev_mmvae.loc[(temp_dev_mmvae.stage == 'cdr = 0.5')][MRI_vol_cols])
    dev_cdr1 = (temp_dev_mmvae.loc[(temp_dev_mmvae.stage == 'cdr >= 1')][MRI_vol_cols])

    p_list_hc_precl = []
    p_list_hc_cdr05 = []
    p_list_hc_cdr1 = []

    cohen_list_hc_precl = []
    cohen_list_hc_cdr05 = []
    cohen_list_hc_cdr1 = []

    for mri_col in MRI_vol_cols:
        F, p = f_oneway(dev_hc[mri_col], dev_precl[mri_col])
        p_list_hc_precl.append(p)
        cohen_list_hc_precl.append(cohen_d(dev_hc[mri_col], dev_precl[mri_col]))

    for mri_col in MRI_vol_cols:
        F, p = f_oneway(dev_hc[mri_col], dev_cdr05[mri_col])
        p_list_hc_cdr05.append(p)
        cohen_list_hc_cdr05.append(cohen_d(dev_hc[mri_col], dev_cdr05[mri_col]))

    for mri_col in MRI_vol_cols:
        F, p = f_oneway(dev_hc[mri_col], dev_cdr1[mri_col])
        p_list_hc_cdr1.append(p)
        cohen_list_hc_cdr1.append(cohen_d(dev_hc[mri_col], dev_cdr1[mri_col]))

    fdr_p_list_hc_precl = fdrcorrection(p_list_hc_precl, method='indep')[1]
    fdr_p_list_hc_cdr05 = fdrcorrection(p_list_hc_cdr05, method='indep')[1]
    fdr_p_list_hc_cdr1 = fdrcorrection(p_list_hc_cdr1, method='indep')[1]

    sig_cohen_hc_precl = cohen_list_hc_precl.copy()
    for i in range(len(cohen_list_hc_precl)):
        if fdr_p_list_hc_precl[i] < 0.1:
            sig_cohen_hc_precl[i] = cohen_list_hc_precl[i]
        else:
            sig_cohen_hc_precl[i] = 0


    sig_cohen_hc_cdr05 = cohen_list_hc_cdr05.copy()
    for i in range(len(cohen_list_hc_cdr05)):
        if fdr_p_list_hc_cdr05[i] < 0.1:
            sig_cohen_hc_cdr05[i] = cohen_list_hc_cdr05[i]
        else:
            sig_cohen_hc_cdr05[i] = 0


    sig_cohen_hc_cdr1 = cohen_list_hc_cdr1.copy()
    for i in range(len(cohen_list_hc_cdr1)):
        if fdr_p_list_hc_cdr1[i] < 0.1:
            sig_cohen_hc_cdr1[i] = cohen_list_hc_cdr1[i]
        else:
            sig_cohen_hc_cdr1[i] = 0
            
#     if MRI == True:
#         #ggseg_mri_cols = [element.rstrip('_Nvol') for element in MRI_vol_cols]
#         effect_hc_precl = {key: value for key, value in dict(zip(ggseg_mri_cols, sig_cohen_hc_precl)).items() if value != 0}
#         effect_hc_cdr05 = {key: value for key, value in dict(zip(ggseg_mri_cols, sig_cohen_hc_cdr05)).items() if value != 0}
#         effect_hc_cdr1 = {key: value for key, value in dict(zip(ggseg_mri_cols, sig_cohen_hc_cdr1)).items() if value != 0}

#     if SUVR == True:
#         #ggseg_mri_cols = [element.rstrip('_Nvol') for element in MRI_vol_cols]
#         effect_hc_precl = {key: abs(value) for key, value in dict(zip(ggseg_mri_cols, sig_cohen_hc_precl)).items() if value != 0}
#         effect_hc_cdr05 = {key: abs(value) for key, value in dict(zip(ggseg_mri_cols, sig_cohen_hc_cdr05)).items() if value != 0}
#         effect_hc_cdr1 = {key: abs(value) for key, value in dict(zip(ggseg_mri_cols, sig_cohen_hc_cdr1)).items() if value != 0}

    return sig_cohen_hc_precl, sig_cohen_hc_cdr05, sig_cohen_hc_cdr1
    

#---------------------------------
#---------------------------------
    
def effect_size_maps(temp_dev_mmvae, MRI_vol_cols, ggseg_mri_cols):
    
    sig_cohen_hc_precl_mri, sig_cohen_hc_cdr05_mri, sig_cohen_hc_cdr1_mri = calculate_effect_size(temp_dev_mmvae, MRI_vol_cols)
    
    sig_cohen_hc_precl_mri = [abs(x) for x in sig_cohen_hc_precl_mri]
    sig_cohen_hc_cdr05_mri = [abs(x) for x in sig_cohen_hc_cdr05_mri]
    sig_cohen_hc_cdr1_mri = [abs(x) for x in sig_cohen_hc_cdr1_mri]

    #ggseg_mri_cols = [element.rstrip('_Nvol') for element in MRI_vol_cols]
    effect_hc_precl_mri = {key: value for key, value in dict(zip(ggseg_mri_cols, sig_cohen_hc_precl_mri)).items() if value != 0}
    effect_hc_cdr05_mri = {key: value for key, value in dict(zip(ggseg_mri_cols, sig_cohen_hc_cdr05_mri)).items() if value != 0}
    effect_hc_cdr1_mri = {key: value for key, value in dict(zip(ggseg_mri_cols, sig_cohen_hc_cdr1_mri)).items() if value != 0}

    return effect_hc_precl_mri, effect_hc_cdr05_mri, effect_hc_cdr1_mri

#-------------------------------------------------------
############### Main function -------------------------
#-------------------------------------------------------

from scipy.stats import f_oneway

reject_corr = pd.read_csv('./saved_dataframes/sig_deviations.csv')
temp_dev_mmvae = pd.read_csv('./saved_dataframes/deviations_mmvae.csv')

with open("./saved_dataframes/MRI_vol_cols", "rb") as fp: 
    MRI_vol_cols = pickle.load(fp)
    
with open("./saved_dataframes/amyloid_SUVR_cols", "rb") as fp: 
    amyloid_SUVR_cols = pickle.load(fp)
    
with open("./saved_dataframes/tau_SUVR_cols", "rb") as fp: 
    tau_SUVR_cols = pickle.load(fp)


#--------------------ADNI ------------------------
plot_effect_size(temp_dev_mmvae, MRI_vol_cols, 'MRI', '_Nvol', 'ADNI', [0, 3.5])
plot_effect_size(temp_dev_mmvae, amyloid_SUVR_cols, 'Amyloid', '_Asuvr', 'ADNI', [0, 3.5])
plot_effect_size(temp_dev_mmvae, tau_SUVR_cols, 'Tau', '_Tsuvr', 'ADNI', [0, 3.5])


#--------------------KARI ------------------------
plot_effect_size(temp_dev_mmvae, MRI_vol_cols, 'MRI', '_Nvol', 'KARI', [0, 3.5])
plot_effect_size(temp_dev_mmvae, amyloid_SUVR_cols, 'Amyloid', '_Asuvr', 'KARI', [0, 3.5])
plot_effect_size(temp_dev_mmvae, tau_SUVR_cols, 'Tau', '_Tsuvr', 'KARI', [0, 3.5])

