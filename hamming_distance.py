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

def plot_dissimilarity(reject_corr, modality_cols, modality, dataset, only_density = True):
    
    save_path = './Plots/Hamming'
    
    reject_corr_temp = reject_corr.loc[reject_corr.dataset == dataset].reset_index(drop = True)
    reject_corr_temp[modality_cols] = reject_corr_temp[modality_cols].fillna(0)
    
    reject_corr_hc = reject_corr_temp.loc[reject_corr_temp.stage == 'HC'].reset_index(drop = True)
    reject_corr_precl = reject_corr_temp.loc[reject_corr_temp.stage == 'preclinical'].reset_index(drop = True)
    reject_corr_cdr05 = reject_corr_temp.loc[reject_corr_temp.stage == 'cdr = 0.5'].reset_index(drop = True)
    reject_corr_cdr1 = reject_corr_temp.loc[reject_corr_temp.stage == 'cdr >= 1'].reset_index(drop = True)

    def hamming_distance(df):
        distances = np.zeros((df.shape[0], df.shape[0]))
        for i in range(df.shape[0]):
            for j in range(df.shape[0]):
                distances[i, j] = sum(df.iloc[i] != df.iloc[j])
        return distances

    dist_hc = hamming_distance(reject_corr_hc[modality_cols])
    dist_precl = hamming_distance(reject_corr_precl[modality_cols])
    dist_cdr05 = hamming_distance(reject_corr_cdr05[modality_cols])
    dist_cdr1 = hamming_distance(reject_corr_cdr1[modality_cols])
    
    #max_val = max(dist_hc.flatten() + dist_precl.flatten() + dist_cdr05.flatten() + dist_cdr1.flatten())

    #--------------------------------
    
    if only_density == False:
        
        plt.subplots(figsize = (18, 12))

        max_val = 150

        plt.subplot(2,2,1)
        sns.heatmap(dist_hc, xticklabels=False, yticklabels=False, annot=False, vmin = 0, vmax = max_val)
        plt.title('{} {} cogntively unimpaired (n = {})'.format(dataset, modality, len(dist_hc)), fontsize = 20)

        plt.subplot(2,2,2)
        sns.heatmap(dist_precl, xticklabels=False, yticklabels=False, annot=False, vmin = 0, vmax = max_val)
        plt.title('{} {} preclinical (n = {})'.format(dataset, modality, len(dist_precl)), fontsize = 20)

        plt.subplot(2,2,3)
        sns.heatmap(dist_cdr05, xticklabels=False, yticklabels=False, annot=False, vmin = 0, vmax = max_val)
        plt.title('{} {} CDR = 0.5 (n = {})'.format(dataset, modality, len(dist_cdr05)), fontsize = 20)

        plt.subplot(2,2,4)
        sns.heatmap(dist_cdr1, xticklabels=False, yticklabels=False, annot=False, vmin = 0, vmax = max_val)
        plt.title('{} {} CDR >= 1 (n = {})'.format(dataset, modality, len(dist_cdr1)), fontsize = 20)

        plt.tight_layout()
        #plt.suptitle('Hamming distance heatmaps')
        plt.subplots_adjust(top = 0.9, hspace= 0.2, wspace = 0.05)
        
        plt.savefig(os.path.join(save_path, 'hamm_'+ modality + '_' + dataset + '.pdf'), bbox_inches = 'tight', dpi = 600)


    #----------------------------------

    plt.figure(figsize = (8,5))
    sns.kdeplot(dist_hc.flatten(), color = 'red', fill = True, label = 'CU')
    sns.kdeplot(dist_precl.flatten(), color = 'blue', fill = True, label = 'preclinical')
    sns.kdeplot(dist_cdr05.flatten(), color = 'green', fill = True, label = 'cdr = 0.5')
    sns.kdeplot(dist_cdr1.flatten(), color = 'yellow', fill = True, label = 'cdr >= 1')
    
    plt.xlabel('Dissimilarity', fontsize = 18)
    plt.ylabel('Density', fontsize = 18)
    plt.title('Density of dissimilarity ({} {})'.format(dataset, modality), fontsize = 20)
    plt.legend(fontsize = 20)
    
    plt.savefig(os.path.join(save_path, 'hamm_kde_'+ modality + '_' + dataset + '.pdf'), bbox_inches = 'tight', dpi = 600)

    
    #return dist_hc, dist_precl, dist_cdr05, dist_cdr1

#-------------------------------------------------------
############### Main function -------------------------
#-------------------------------------------------------
    
reject_corr = pd.read_csv('./saved_dataframes/sig_deviations.csv')
temp_dev_mmvae = pd.read_csv('./saved_dataframes/deviations_mmvae.csv')

with open("./saved_dataframes/MRI_vol_cols", "rb") as fp: 
    MRI_vol_cols = pickle.load(fp)
    
with open("./saved_dataframes/amyloid_SUVR_cols", "rb") as fp: 
    amyloid_SUVR_cols = pickle.load(fp)
    
with open("./saved_dataframes/tau_SUVR_cols", "rb") as fp: 
    tau_SUVR_cols = pickle.load(fp)

fs_cols = MRI_vol_cols + amyloid_SUVR_cols + tau_SUVR_cols
    
#-------------ADNI ------------------
plot_dissimilarity(reject_corr, MRI_vol_cols, 'MRI', 'ADNI', only_density = False)
plot_dissimilarity(reject_corr, amyloid_SUVR_cols, 'Amyloid', 'ADNI', only_density = False)
plot_dissimilarity(reject_corr, tau_SUVR_cols, 'Tau', 'ADNI', only_density = False)
plot_dissimilarity(reject_corr, fs_cols, 'all_modalities', 'ADNI', only_density = False)

#-------------KARI ------------------
plot_dissimilarity(reject_corr, MRI_vol_cols, 'MRI', 'KARI', only_density = False)
plot_dissimilarity(reject_corr, amyloid_SUVR_cols, 'Amyloid', 'KARI', only_density = False)
plot_dissimilarity(reject_corr, tau_SUVR_cols, 'Tau', 'KARI', only_density = False)
plot_dissimilarity(reject_corr, fs_cols, 'all_modalities', 'KARI', only_density = False)
