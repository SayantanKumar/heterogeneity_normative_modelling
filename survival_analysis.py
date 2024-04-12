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
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

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

def calculate_progression_event(cdr_both_test, dataset):
    
    prog_cdr1 = cdr_both_test.copy()

    prog_cdr1['followup_interval'] = 0
    for i in range(len(prog_cdr1) - 1):
        if prog_cdr1.loc[i, 'ID'] == prog_cdr1.loc[i+1, 'ID']:
            prog_cdr1.loc[i+1, 'followup_interval'] = (pd.to_datetime(prog_cdr1.loc[i+1, 'TESTDATE']) - pd.to_datetime(prog_cdr1.loc[i, 'TESTDATE'])).days

    prog_cdr1['visit_type'] = 'followup'
    prog_cdr1.loc[0, 'visit_type'] = 'baseline'
    for i in range(len(prog_cdr1) - 1):
        if prog_cdr1.loc[i, 'ID'] != prog_cdr1.loc[i+1, 'ID']:
            prog_cdr1.loc[i+1, 'visit_type'] = 'baseline'

    excl_baseline_event = prog_cdr1.loc[(prog_cdr1.visit_type == 'baseline') & (prog_cdr1.cdr >= 1)].ID.values

    prog_cdr1 = prog_cdr1.loc[~prog_cdr1.ID.isin(excl_baseline_event)].reset_index(drop = True)

    prog_cdr1['event'] = 0
    for i in range(len(prog_cdr1) - 1):
        if (prog_cdr1.loc[i, 'visit_type'] == 'followup') & (prog_cdr1.loc[i, 'cdr'] >= 1):
            prog_cdr1.loc[i, 'event'] = 1

    prog_cdr1['time_to_event'] = prog_cdr1.groupby('ID')['followup_interval'].cumsum()
    
    prog_cdr1_cov = prog_cdr1.merge(temp_dev_mmvae[['ID','Age', 'Sex', 'stage', 'all_toc', 'mean_dev_all_sig_fdr']], on = 'ID', how = 'left')
    prog_cdr1_cov = prog_cdr1_cov.loc[prog_cdr1_cov.stage != 'HC'].reset_index(drop = True)

    prog_cdr1_cov['quantile_sev'] = pd.qcut(prog_cdr1_cov['mean_dev_all_sig_fdr'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates = 'drop')
    prog_cdr1_cov['quantile_toc'] = pd.qcut(prog_cdr1_cov['all_toc'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates = 'drop')

    prog_cdr1_cov['time_to_event'] = round(prog_cdr1_cov['time_to_event']/30)
    
    return prog_cdr1_cov


#------------------------------------------------
#------------------------------------------------


def plot_survival_curve(prog_cdr1_cov, quantile_col, dataset):
    
    save_path = './Plots'
    
    #quantile_col = 'quantile_sev'
    prog_cdr1_cov_q1 = prog_cdr1_cov.loc[prog_cdr1_cov[quantile_col] == 'Q1']
    prog_cdr1_cov_q2 = prog_cdr1_cov.loc[prog_cdr1_cov[quantile_col] == 'Q2']
    prog_cdr1_cov_q3 = prog_cdr1_cov.loc[prog_cdr1_cov[quantile_col] == 'Q3']
    prog_cdr1_cov_q4 = prog_cdr1_cov.loc[prog_cdr1_cov[quantile_col] == 'Q4']

    #-------------------------------
    cph1 = CoxPHFitter()
    cph1.fit(prog_cdr1_cov_q1, duration_col='time_to_event', event_col='event', formula='Age + Sex')
    #print(cph1.summary)

    cph2 = CoxPHFitter()
    cph2.fit(prog_cdr1_cov_q2, duration_col='time_to_event', event_col='event', formula='Age + Sex')
    #print(cph2.summary)

    cph3 = CoxPHFitter()
    cph3.fit(prog_cdr1_cov_q3, duration_col='time_to_event', event_col='event', formula='Age + Sex')
    #print(cph3.summary)

    cph4 = CoxPHFitter()
    cph4.fit(prog_cdr1_cov_q4, duration_col='time_to_event', event_col='event', formula='Age + Sex')
    #print(cph4.summary)

    #--------------------------------

    kmf1 = KaplanMeierFitter()
    kmf1.fit(prog_cdr1_cov_q1['time_to_event'], event_observed=prog_cdr1_cov_q1['event'])

    kmf2 = KaplanMeierFitter()
    kmf2.fit(prog_cdr1_cov_q2['time_to_event'], event_observed=prog_cdr1_cov_q2['event'])

    kmf3 = KaplanMeierFitter()
    kmf3.fit(prog_cdr1_cov_q3['time_to_event'], event_observed=prog_cdr1_cov_q3['event'])

    kmf4 = KaplanMeierFitter()
    kmf4.fit(prog_cdr1_cov_q4['time_to_event'], event_observed=prog_cdr1_cov_q4['event'])

    #---------------------------------

    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10, 5))

    # kmf1.survival_function_.plot(ax=ax, color='blue', linewidth = 2, label='quantile_1')
    # kmf2.survival_function_.plot(ax=ax, color='red', linewidth = 2, label='quantile_2')
    # kmf3.survival_function_.plot(ax=ax, color='green', linewidth = 2, label='quantile_3')
    # kmf4.survival_function_.plot(ax=ax, color='yellow', linewidth = 2, label='quantile_4')

    kmf1.plot(ax=ax, color='blue', linewidth = 2, label='q1')
    kmf2.plot(ax=ax, color='red', linewidth = 2, label='q2')
    kmf3.plot(ax=ax, color='green', linewidth = 2, label='q3')
    kmf4.plot(ax=ax, color='orange', linewidth = 2, label='q4')


    marker1, = ax.plot([], [], color='blue', label='q1')
    marker2, = ax.plot([], [], color='red', label='q2')
    marker3, = ax.plot([], [], color='green', label='q3')
    marker4, = ax.plot([], [], color='orange', label='q4')
    ax.legend(handles=[marker1, marker2, marker3, marker4], fontsize = 16)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_xlabel('Follow-up (months)', fontsize = 16)
    ax.set_ylabel('Progression probability', fontsize = 16)
    ax.set_title("{} Progression to CDR >= 1 (disease severity)".format(dataset), fontsize = 18)

    plt.savefig(os.path.join(save_path, 'progression_' + dataset + '.pdf'), bbox_inches = 'tight', dpi = 600)



#-------------------------------------------------------
############### Main function -------------------------
#-------------------------------------------------------

reject_corr = pd.read_csv('./saved_dataframes/sig_deviations.csv')
temp_dev_mmvae = pd.read_csv('./saved_dataframes/deviations_mmvae.csv')
all_atn_adni = pd.read_csv('./saved_dataframes/all_atn_adni.csv')
all_atn_kari = pd.read_csv('./saved_dataframes/all_atn_kari.csv')

with open("./saved_dataframes/MRI_vol_cols", "rb") as fp: 
    MRI_vol_cols = pickle.load(fp)
    
with open("./saved_dataframes/amyloid_SUVR_cols", "rb") as fp: 
    amyloid_SUVR_cols = pickle.load(fp)
    
with open("./saved_dataframes/tau_SUVR_cols", "rb") as fp: 
    tau_SUVR_cols = pickle.load(fp)

    
#------------------------------------------------
#------------------------------------------------

cdr_adni = pd.read_csv('/Users/sayantankumar/Desktop/Aris_Work/Data/ADNI/CDR.csv')[['RID', 'USERDATE', 'CDGLOBAL']]
cdr_adni = cdr_adni.loc[cdr_adni.RID.isin(temp_dev_mmvae.ID.values)].reset_index(drop = True).rename(columns = {'RID':'ID', 'USERDATE':'TESTDATE', 'CDGLOBAL':'cdr'})

cdr_adni_test = pd.merge(cdr_adni, all_atn_adni[['RID', 'mri_date', 'amyloid_date', 'tau_date']], left_on = 'ID', right_on = 'RID', how = 'right').sort_values(by = ['ID', 'TESTDATE']).reset_index(drop = True)
cdr_adni_test['date_diff'] = (pd.to_datetime(cdr_adni_test['TESTDATE']) - pd.to_datetime(cdr_adni_test['mri_date'])).dt.days

cdr_adni_test = cdr_adni_test.sort_values(by = ['ID', 'date_diff']).drop_duplicates(subset = ['ID', 'TESTDATE'], keep = 'first')

cdr_adni_test = cdr_adni_test.loc[cdr_adni_test.date_diff > -60].reset_index(drop = True) 
cdr_adni_test['dataset'] = 'ADNI'
#-----------------------------------------

cdr_kari = pd.read_excel(os.path.join('/Users/sayantankumar/Desktop/Aris_Work/Data/module_freeze_june22/clinical_core/mod_b4_cdr.xlsx'))[['ID', 'TESTDATE', 'cdr']]
cdr_kari = cdr_kari.loc[cdr_kari.ID.isin(temp_dev_mmvae.ID.values)].reset_index(drop = True).rename(columns = {'RID':'ID', 'USERDATE':'TESTDATE', 'CDGLOBAL':'cdr'})

cdr_kari_test = pd.merge(cdr_kari, all_atn_kari[['ID', 'mri_date', 'amyloid_date', 'tau_date']], left_on = 'ID', right_on = 'ID', how = 'right').sort_values(by = ['ID', 'TESTDATE']).reset_index(drop = True)
cdr_kari_test['date_diff'] = (pd.to_datetime(cdr_kari_test['TESTDATE']) - pd.to_datetime(cdr_kari_test['mri_date'])).dt.days

cdr_kari_test = cdr_kari_test.sort_values(by = ['ID', 'date_diff']).drop_duplicates(subset = ['ID', 'TESTDATE'], keep = 'first')

cdr_kari_test = cdr_kari_test.loc[cdr_kari_test.date_diff > -60].reset_index(drop = True) 
cdr_kari_test['dataset'] = 'KARI'

cdr_both_test = pd.concat([cdr_adni_test, cdr_kari_test])[['ID', 'TESTDATE', 'cdr']].reset_index(drop = True)

#------------------------------------------------
#------------------------------------------------

prog_cdr1_cov_ADNI = calculate_progression_event(cdr_adni_test, 'ADNI')
prog_cdr1_cov_KARI = calculate_progression_event(cdr_kari_test, 'KARI')

plot_survival_curve(prog_cdr1_cov_ADNI, 'quantile_sev', 'ADNI')
plot_survival_curve(prog_cdr1_cov_KARI, 'quantile_toc', 'KARI')