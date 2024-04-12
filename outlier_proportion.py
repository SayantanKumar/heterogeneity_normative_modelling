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

def deviation_atlas_maps(dev_bvae, MRI_vol_cols, amyloid_SUVR_cols, tau_SUVR_cols):
    
    temp_dev_bvae = dev_bvae.copy()

    temp_dev_mri = temp_dev_bvae[common_cols + MRI_vol_cols]
    temp_dev_amyloid = temp_dev_bvae[common_cols + amyloid_SUVR_cols]
    temp_dev_tau = temp_dev_bvae[common_cols + tau_SUVR_cols]

    temp_dev_mri.columns = temp_dev_mri.columns.str.replace(r'_Nvol', '')
    temp_mri_cols = [col for col in temp_dev_mri.columns if col not in common_cols]

    temp_dev_amyloid.columns = temp_dev_amyloid.columns.str.replace(r'_Asuvr', '')
    temp_amyloid_cols = [col for col in temp_dev_amyloid.columns if col not in common_cols]

    temp_dev_tau.columns = temp_dev_tau.columns.str.replace(r'_Tsuvr', '')
    temp_tau_cols = [col for col in temp_dev_tau.columns if col not in common_cols]

    temp_dev_mri_precl = temp_dev_mri.loc[temp_dev_mri.stage == 'cdr = 0 amyloid positive']
    temp_dev_mri_cdr05 = temp_dev_mri.loc[temp_dev_mri.stage == 'cdr = 0.5']
    temp_dev_mri_cdr1 = temp_dev_mri.loc[temp_dev_mri.stage == 'cdr >= 1']

    temp_dev_amyloid_precl = temp_dev_amyloid.loc[temp_dev_amyloid.stage == 'cdr = 0 amyloid positive']
    temp_dev_amyloid_cdr05 = temp_dev_amyloid.loc[temp_dev_amyloid.stage == 'cdr = 0.5']
    temp_dev_amyloid_cdr1 = temp_dev_amyloid.loc[temp_dev_amyloid.stage == 'cdr >= 1']

    temp_dev_tau_precl = temp_dev_tau.loc[temp_dev_tau.stage == 'cdr = 0 amyloid positive']
    temp_dev_tau_cdr05 = temp_dev_tau.loc[temp_dev_tau.stage == 'cdr = 0.5']
    temp_dev_tau_cdr1 = temp_dev_tau.loc[temp_dev_tau.stage == 'cdr >= 1']


    #********************************************************************************************
    ##------------------------ Plotting MRI deviation maps across disease stages ---------
    #********************************************************************************************

    ggseg_python.plot_dk(abs(temp_dev_mri_precl[temp_mri_cols]).mean().to_dict(), cmap='hot',
                    background='w', edgecolor='k',  bordercolor='gray', vminmax = [0.5, 2.1],
                    ylabel='MRI cortical deviation', figsize = (6,4), fontsize = 24, title='MRI cortical deviation maps for preclinical (N = {})'.format(len(temp_dev_mri_precl)))
    print('temp_dev_mri_precl_cort : Max = {}, Min = {}'.format(abs(temp_dev_mri_precl[temp_mri_cols]).mean().max(), abs(temp_dev_mri_precl[temp_mri_cols]).mean().min()))


    ggseg_python.plot_dk(abs(temp_dev_mri_cdr05[temp_mri_cols]).mean().to_dict(), cmap='hot',
                    background='w', edgecolor='k',  bordercolor='gray', vminmax = [0.5, 2.1],
                    ylabel='MRI cortical deviation', figsize = (6,4), fontsize = 24, title='MRI cortical deviation maps for CDR = 0.5 (N = {})'.format(len(temp_dev_mri_cdr05)))
    print('temp_dev_mri_cdr05_cort : Max = {}, Min = {}'.format(abs(temp_dev_mri_cdr05[temp_mri_cols]).mean().max(), abs(temp_dev_mri_cdr05[temp_mri_cols]).mean().min()))


    ggseg_python.plot_dk(abs(temp_dev_mri_cdr1[temp_mri_cols]).mean().to_dict(), cmap='hot',
                    background='w', edgecolor='k',  bordercolor='gray', vminmax = [0.5, 2.1],
                    ylabel='MRI cortical deviation', figsize = (6,4), fontsize = 24, title='MRI cortical deviation maps for CDR >= 1 (N = {})'.format(len(temp_dev_mri_cdr1)))
    print('temp_dev_mri_precl_cort : Max = {}, Min = {}'.format(abs(temp_dev_mri_cdr1[temp_mri_cols]).mean().max(), abs(temp_dev_mri_cdr1[temp_mri_cols]).mean().min()))


    #********************************************************************************************
    ##------------------------ Plotting amyloid deviation maps across disease stages ---------
    #********************************************************************************************

    ggseg_python.plot_dk(abs(temp_dev_amyloid_precl[temp_amyloid_cols]).mean().to_dict(), cmap='hot',
                    background='w', edgecolor='k',  bordercolor='gray', vminmax = [0.5, 4.5],
                    ylabel='Amyloid cortical deviation', figsize = (6,4), fontsize = 24, title='Amyloid cortical deviation maps for preclinical (N = {})'.format(len(temp_dev_amyloid_precl)))
    print('temp_dev_amyloid_precl_cort : Max = {}, Min = {}'.format(abs(temp_dev_amyloid_precl[temp_amyloid_cols]).mean().max(), abs(temp_dev_amyloid_precl[temp_amyloid_cols]).mean().min()))


    ggseg_python.plot_dk(abs(temp_dev_amyloid_cdr05[temp_amyloid_cols]).mean().to_dict(), cmap='hot',
                    background='w', edgecolor='k',  bordercolor='gray', vminmax = [0.5, 4.5],
                    ylabel='Amyloid cortical deviation', figsize = (6,4), fontsize = 24, title='Amyloid cortical deviation maps for CDR = 0.5 (N = {})'.format(len(temp_dev_amyloid_cdr05)))
    print('temp_dev_amyloid_cdr05_cort : Max = {}, Min = {}'.format(abs(temp_dev_amyloid_cdr05[temp_amyloid_cols]).mean().max(), abs(temp_dev_amyloid_cdr05[temp_amyloid_cols]).mean().min()))


    ggseg_python.plot_dk(abs(temp_dev_amyloid_cdr1[temp_amyloid_cols]).mean().to_dict(), cmap='hot',
                    background='w', edgecolor='k',  bordercolor='gray', vminmax = [0.5, 4.5],
                    ylabel='Amyloid cortical deviation', figsize = (6,4), fontsize = 24, title='Amyloid cortical deviation maps for CDR >=1 (N = {})'.format(len(temp_dev_amyloid_cdr1)))
    print('temp_dev_amyloid_cdr1_cort : Max = {}, Min = {}'.format(abs(temp_dev_amyloid_cdr1[temp_amyloid_cols]).mean().max(), abs(temp_dev_amyloid_cdr1[temp_amyloid_cols]).mean().min()))


    #********************************************************************************************
    ##------------------------ Plotting tau deviation maps across disease stages ---------
    #********************************************************************************************

    ggseg_python.plot_dk(abs(temp_dev_tau_precl[temp_tau_cols]).mean().to_dict(), cmap='hot',
                    background='w', edgecolor='k',  bordercolor='gray', vminmax = [0.5, 5],
                    ylabel='Tau cortical deviation', figsize = (6,4), fontsize = 24, title='Tau cortical deviation maps for preclinical (N = {})'.format(len(temp_dev_tau_precl)))
    print('temp_dev_tau_precl_cort : Max = {}, Min = {}'.format(abs(temp_dev_tau_precl[temp_tau_cols]).mean().max(), abs(temp_dev_tau_precl[temp_tau_cols]).mean().min()))


    ggseg_python.plot_dk(abs(temp_dev_tau_cdr05[temp_tau_cols]).mean().to_dict(), cmap='hot',
                    background='w', edgecolor='k',  bordercolor='gray', vminmax = [0.5, 5],
                    ylabel='Tau cortical deviation', figsize = (6,4), fontsize = 24, title='Tau cortical deviation maps for CDR = 0.5 (N = {})'.format(len(temp_dev_tau_cdr05)))
    print('temp_dev_tau_cdr05_cort : Max = {}, Min = {}'.format(abs(temp_dev_tau_cdr05[temp_tau_cols]).mean().max(), abs(temp_dev_tau_cdr05[temp_tau_cols]).mean().min()))


    ggseg_python.plot_dk(abs(temp_dev_tau_cdr1[temp_tau_cols]).mean().to_dict(), cmap='hot',
                    background='w', edgecolor='k',  bordercolor='gray', vminmax = [0.5, 5],
                    ylabel='Tau cortical deviation', figsize = (6,4), fontsize = 24, title='Tau cortical deviation maps for CDR >=1 (N = {})'.format(len(temp_dev_tau_cdr1)))
    print('temp_dev_tau_cdr1_cort : Max = {}, Min = {}'.format(abs(temp_dev_tau_cdr1[temp_tau_cols]).mean().max(), abs(temp_dev_tau_cdr1[temp_tau_cols]).mean().min()))

    
#------------------------------------------------------------------------
#------------------------------------------------------------------------

def freq_of_sig_disease_cat(reject_corr, MRI_vol_cols, amyloid_SUVR_cols, tau_SUVR_cols):
    
    mri_roi_sig_hc = pd.DataFrame(reject_corr.loc[reject_corr.stage == 'HC'][MRI_vol_cols].apply(lambda x: (x == 1).sum(), axis=0), index = MRI_vol_cols).rename(columns = {0:'sig_count_mri'})
    mri_roi_sig_precl = pd.DataFrame(reject_corr.loc[reject_corr.stage == 'preclinical'][MRI_vol_cols].apply(lambda x: (x == 1).sum(), axis=0), index = MRI_vol_cols).rename(columns = {0:'sig_count_mri'})
    mri_roi_sig_cdr05 = pd.DataFrame(reject_corr.loc[(reject_corr.stage == 'cdr = 0.5') & (reject_corr.amyloid_positive == 1)][MRI_vol_cols].apply(lambda x: (x == 1).sum(), axis=0), index = MRI_vol_cols).rename(columns = {0:'sig_count_mri'})
    mri_roi_sig_cdr1 = pd.DataFrame(reject_corr.loc[(reject_corr.stage == 'cdr >= 1') & (reject_corr.amyloid_positive == 1)][MRI_vol_cols].apply(lambda x: (x == 1).sum(), axis=0), index = MRI_vol_cols).rename(columns = {0:'sig_count_mri'})

    amyloid_roi_sig_hc = pd.DataFrame(reject_corr.loc[reject_corr.stage == 'HC'][amyloid_SUVR_cols].apply(lambda x: (x == 1).sum(), axis=0), index = amyloid_SUVR_cols).rename(columns = {0:'sig_count_amyloid'})
    amyloid_roi_sig_precl = pd.DataFrame(reject_corr.loc[reject_corr.stage == 'preclinical'][amyloid_SUVR_cols].apply(lambda x: (x == 1).sum(), axis=0), index = amyloid_SUVR_cols).rename(columns = {0:'sig_count_amyloid'})
    amyloid_roi_sig_cdr05 = pd.DataFrame(reject_corr.loc[(reject_corr.stage == 'cdr = 0.5') & (reject_corr.amyloid_positive == 1)][amyloid_SUVR_cols].apply(lambda x: (x == 1).sum(), axis=0), index = amyloid_SUVR_cols).rename(columns = {0:'sig_count_amyloid'})
    amyloid_roi_sig_cdr1 = pd.DataFrame(reject_corr.loc[(reject_corr.stage == 'cdr >= 1') & (reject_corr.amyloid_positive == 1)][amyloid_SUVR_cols].apply(lambda x: (x == 1).sum(), axis=0), index = amyloid_SUVR_cols).rename(columns = {0:'sig_count_amyloid'})

    tau_roi_sig_hc = pd.DataFrame(reject_corr.loc[reject_corr.stage == 'HC'][tau_SUVR_cols].apply(lambda x: (x == 1).sum(), axis=0), index = tau_SUVR_cols).rename(columns = {0:'sig_count_tau'})
    tau_roi_sig_precl = pd.DataFrame(reject_corr.loc[reject_corr.stage == 'preclinical'][tau_SUVR_cols].apply(lambda x: (x == 1).sum(), axis=0), index = tau_SUVR_cols).rename(columns = {0:'sig_count_tau'})
    tau_roi_sig_cdr05 = pd.DataFrame(reject_corr.loc[(reject_corr.stage == 'cdr = 0.5') & (reject_corr.amyloid_positive == 1)][tau_SUVR_cols].apply(lambda x: (x == 1).sum(), axis=0), index = tau_SUVR_cols).rename(columns = {0:'sig_count_tau'})
    tau_roi_sig_cdr1 = pd.DataFrame(reject_corr.loc[(reject_corr.stage == 'cdr >= 1') & (reject_corr.amyloid_positive == 1)][tau_SUVR_cols].apply(lambda x: (x == 1).sum(), axis=0), index = tau_SUVR_cols).rename(columns = {0:'sig_count_tau'})

    mri_roi_sig_hc.index = mri_roi_sig_hc.index.str.rstrip('_Nvol')
    mri_roi_sig_precl.index = mri_roi_sig_precl.index.str.rstrip('_Nvol')
    mri_roi_sig_cdr05.index = mri_roi_sig_cdr05.index.str.rstrip('_Nvol')
    mri_roi_sig_cdr1.index = mri_roi_sig_cdr1.index.str.rstrip('_Nvol')

    amyloid_roi_sig_hc.index = amyloid_roi_sig_hc.index.str.rstrip('_Asuvr')
    amyloid_roi_sig_precl.index = amyloid_roi_sig_precl.index.str.rstrip('_Asuvr')
    amyloid_roi_sig_cdr05.index = amyloid_roi_sig_cdr05.index.str.rstrip('_Asuvr')
    amyloid_roi_sig_cdr1.index = amyloid_roi_sig_cdr1.index.str.rstrip('_Asuvr')

    tau_roi_sig_hc.index = tau_roi_sig_hc.index.str.rstrip('_Tsuvr')
    tau_roi_sig_precl.index = tau_roi_sig_precl.index.str.rstrip('_Tsuvr')
    tau_roi_sig_cdr05.index = tau_roi_sig_cdr05.index.str.rstrip('_Tsuvr')
    tau_roi_sig_cdr1.index = tau_roi_sig_cdr1.index.str.rstrip('_Tsuvr')


    ############################################################
    #----------------- MRI ------------------------------------
    ############################################################
    
    mri_roi_sig_hc = (mri_roi_sig_hc/len(reject_corr.loc[reject_corr.stage == 'HC']))*100
    mri_roi_sig_precl = (mri_roi_sig_precl/len(reject_corr.loc[reject_corr.stage == 'preclinical']))*100
    mri_roi_sig_cdr05 = (mri_roi_sig_cdr05/len(reject_corr.loc[reject_corr.stage == 'cdr = 0.5']))*100
    mri_roi_sig_cdr1 = (mri_roi_sig_cdr1/len(reject_corr.loc[reject_corr.stage == 'cdr >= 1']))*100
    
    
    #----- healthy controls -------
    ggseg_python.plot_dk(mri_roi_sig_hc.loc[mri_roi_sig_hc['sig_count_mri'] > 0]['sig_count_mri'].to_dict(), cmap= 'hot',
                        background='w', edgecolor='k',  bordercolor='gray', vminmax = [0, 100],
                        ylabel='Outlier proportion (%)', figsize = (6,4), fontsize = 18, title='MRI Cortical for controls (n = 18)')

    ggseg_python.plot_aseg(mri_roi_sig_hc.loc[mri_roi_sig_hc['sig_count_mri'] > 0]['sig_count_mri'].to_dict(), cmap= 'hot',
                        background='w', edgecolor='k',  bordercolor='gray', vminmax = [0, 100],
                        ylabel='Outlier proportion (%)', figsize = (6,4), fontsize = 18, title='MRI Cortical for controls (n = 18)')

    #----- preclinical -------
    ggseg_python.plot_dk(mri_roi_sig_precl.loc[mri_roi_sig_precl['sig_count_mri'] > 0]['sig_count_mri'].to_dict(), cmap= 'hot',
                        background='w', edgecolor='k',  bordercolor='gray', vminmax = [0, 100],
                        ylabel='Frequency (%) of Significance', figsize = (6,4), fontsize = 18, title='MRI Cortical for preclinical (n = 166)')

    ggseg_python.plot_aseg(mri_roi_sig_precl.loc[mri_roi_sig_precl['sig_count_mri'] > 0]['sig_count_mri'].to_dict(), cmap='hot',
                        background='w', edgecolor='k',  bordercolor='gray', vminmax = [0, 100],
                        ylabel='Frequency (%) of Significance', figsize = (6,4), fontsize = 18, title='MRI subortical for preclinical (n = 166)')

    #----- CDR = 0.5 -------
    ggseg_python.plot_dk(mri_roi_sig_cdr05.loc[mri_roi_sig_cdr05['sig_count_mri'] > 0]['sig_count_mri'].to_dict(), cmap= 'hot',
                        background='w', edgecolor='k',  bordercolor='gray', vminmax = [0, 100],
                        ylabel='Frequency (%) of Significance', figsize = (6,4), fontsize = 18, title='MRI Cortical for CDR = 0.5 (n = 172)')

    ggseg_python.plot_aseg(mri_roi_sig_cdr05.loc[mri_roi_sig_cdr05['sig_count_mri'] > 0]['sig_count_mri'].to_dict(), cmap='hot',
                        background='w', edgecolor='k',  bordercolor='gray', vminmax = [0, 100],
                        ylabel='Frequency (%) of Significance', figsize = (6,4), fontsize = 18, title='MRI subortical for CDR = 0.5 (n = 172)')

    #----- CDR >= 1 -------
    ggseg_python.plot_dk(mri_roi_sig_cdr1.loc[mri_roi_sig_cdr1['sig_count_mri'] > 0]['sig_count_mri'].to_dict(), cmap= 'hot',
                        background='w', edgecolor='k',  bordercolor='gray', vminmax = [0, 100],
                        ylabel='Frequency (%) of Significance', figsize = (6,4), fontsize = 18, title='MRI Cortical for CDR >=1 (n = 43)')

    ggseg_python.plot_aseg(mri_roi_sig_cdr1.loc[mri_roi_sig_cdr1['sig_count_mri'] > 0]['sig_count_mri'].to_dict(), cmap='hot',
                        background='w', edgecolor='k',  bordercolor='gray', vminmax = [0, 100],
                        ylabel='Frequency (%) of Significance', figsize = (6,4), fontsize = 18, title='MRI subortical for CDR >=1 (n = 43)')


    ############################################################
    #----------------- Amyloid ------------------------------------
    ############################################################

    amyloid_roi_sig_hc = (amyloid_roi_sig_hc/len(reject_corr.loc[reject_corr.stage == 'HC']))*100
    amyloid_roi_sig_precl = (amyloid_roi_sig_precl/len(reject_corr.loc[reject_corr.stage == 'preclinical']))*100
    amyloid_roi_sig_cdr05 = (amyloid_roi_sig_cdr05/len(reject_corr.loc[reject_corr.stage == 'cdr = 0.5']))*100
    amyloid_roi_sig_cdr1 = (amyloid_roi_sig_cdr1/len(reject_corr.loc[reject_corr.stage == 'cdr >= 1']))*100
    
    #----- healthy controls -------
    ggseg_python.plot_dk(amyloid_roi_sig_hc.loc[amyloid_roi_sig_hc['sig_count_amyloid'] > 0]['sig_count_amyloid'].to_dict(), cmap= 'hot',
                        background='w', edgecolor='k',  bordercolor='gray', vminmax = [0, 100],
                        ylabel='Frequency (%) of Significance', figsize = (6,4), fontsize = 18, title='Amyloid Cortical for controls (n = 18)')

    ggseg_python.plot_aseg(amyloid_roi_sig_hc.loc[amyloid_roi_sig_hc['sig_count_amyloid'] > 0]['sig_count_amyloid'].to_dict(), cmap='hot',
                        background='w', edgecolor='k',  bordercolor='gray', vminmax = [0, 100],
                        ylabel='Frequency (%) of Significance', figsize = (6,4), fontsize = 18, title='Amyloid subortical for controls (n = 18)')

    #----- preclinical -------
    ggseg_python.plot_dk(amyloid_roi_sig_precl.loc[amyloid_roi_sig_precl['sig_count_amyloid'] > 0]['sig_count_amyloid'].to_dict(), cmap= 'hot',
                        background='w', edgecolor='k',  bordercolor='gray', vminmax = [0, 100],
                        ylabel='Frequency (%) of Significance', figsize = (6,4), fontsize = 18, title='Amyloid Cortical for preclinical (n = 166)')

    ggseg_python.plot_aseg(amyloid_roi_sig_precl.loc[amyloid_roi_sig_precl['sig_count_amyloid'] > 0]['sig_count_amyloid'].to_dict(), cmap='hot',
                        background='w', edgecolor='k',  bordercolor='gray', vminmax = [0, 100],
                        ylabel='Frequency (%) of Significance', figsize = (6,4), fontsize = 18, title='Amyloid subortical for preclinical (n = 166)')

    #----- CDR = 0.5 -------
    ggseg_python.plot_dk(amyloid_roi_sig_cdr05.loc[amyloid_roi_sig_cdr05['sig_count_amyloid'] > 0]['sig_count_amyloid'].to_dict(), cmap= 'hot',
                        background='w', edgecolor='k',  bordercolor='gray', vminmax = [0, 100],
                        ylabel='Frequency (%) of Significance', figsize = (6,4), fontsize = 18, title='Amyloid Cortical for CDR = 0.5 (n = 172)')

    ggseg_python.plot_aseg(amyloid_roi_sig_cdr05.loc[amyloid_roi_sig_cdr05['sig_count_amyloid'] > 0]['sig_count_amyloid'].to_dict(), cmap='hot',
                        background='w', edgecolor='k',  bordercolor='gray', vminmax = [0, 100],
                        ylabel='Frequency (%) of Significance', figsize = (6,4), fontsize = 18, title='Amyloid subortical for CDR = 0.5 (n = 172)')

    #----- CDR >= 1 -------
    ggseg_python.plot_dk(amyloid_roi_sig_cdr1.loc[amyloid_roi_sig_cdr1['sig_count_amyloid'] > 0]['sig_count_amyloid'].to_dict(), cmap= 'hot',
                        background='w', edgecolor='k',  bordercolor='gray', vminmax = [0, 100],
                        ylabel='Frequency (%) of Significance', figsize = (6,4), fontsize = 18, title='Amyloid Cortical for CDR >=1 (n = 43)')

    ggseg_python.plot_aseg(amyloid_roi_sig_cdr1.loc[amyloid_roi_sig_cdr1['sig_count_amyloid'] > 0]['sig_count_amyloid'].to_dict(), cmap='hot',
                        background='w', edgecolor='k',  bordercolor='gray', vminmax = [0, 100],
                        ylabel='Frequency (%) of Significance', figsize = (6,4), fontsize = 18, title='Amyloid subortical for CDR >=1 (n = 43)')


    ############################################################
    #----------------- Tau ------------------------------------
    ############################################################

    tau_roi_sig_hc = (tau_roi_sig_hc/len(reject_corr.loc[reject_corr.stage == 'HC']))*100
    tau_roi_sig_precl = (tau_roi_sig_precl/len(reject_corr.loc[reject_corr.stage == 'preclinical']))*100
    tau_roi_sig_cdr05 = (tau_roi_sig_cdr05/len(reject_corr.loc[reject_corr.stage == 'cdr = 0.5']))*100
    tau_roi_sig_cdr1 = (tau_roi_sig_cdr1/len(reject_corr.loc[reject_corr.stage == 'cdr >= 1']))*100
    
    #----- healthy controls -------
    ggseg_python.plot_dk(tau_roi_sig_hc.loc[tau_roi_sig_hc['sig_count_tau'] > 0]['sig_count_tau'].to_dict(), cmap= 'hot',
                        background='w', edgecolor='k',  bordercolor='gray', vminmax = [0, 100],
                        ylabel='Frequency (%) of Significance', figsize = (6,4), fontsize = 18, title='Tau Cortical for controls (n = 18)')

    ggseg_python.plot_aseg(tau_roi_sig_hc.loc[tau_roi_sig_hc['sig_count_tau'] > 0]['sig_count_tau'].to_dict(), cmap='hot',
                        background='w', edgecolor='k',  bordercolor='gray', vminmax = [0, 100],
                        ylabel='Frequency (%) of Significance', figsize = (6,4), fontsize = 18, title='Tau subortical for controls (n = 18)')

    #----- preclinical -------
    ggseg_python.plot_dk(tau_roi_sig_precl.loc[tau_roi_sig_precl['sig_count_tau'] > 0]['sig_count_tau'].to_dict(), cmap= 'hot',
                        background='w', edgecolor='k',  bordercolor='gray', vminmax = [0, 100],
                        ylabel='Frequency (%) of Significance', figsize = (6,4), fontsize = 18, title='Tau Cortical for preclinical (n = 166)')

    ggseg_python.plot_aseg(tau_roi_sig_precl.loc[tau_roi_sig_precl['sig_count_tau'] > 0]['sig_count_tau'].to_dict(), cmap='hot',
                        background='w', edgecolor='k',  bordercolor='gray', vminmax = [0, 100],
                        ylabel='Frequency (%) of Significance', figsize = (6,4), fontsize = 18, title='Tau subortical for preclinical (n = 166)')

    #----- CDR = 0.5 -------
    ggseg_python.plot_dk(tau_roi_sig_cdr05.loc[tau_roi_sig_cdr05['sig_count_tau'] > 0]['sig_count_tau'].to_dict(), cmap= 'hot',
                        background='w', edgecolor='k',  bordercolor='gray', vminmax = [0, 100],
                        ylabel='Frequency (%) of Significance', figsize = (6,4), fontsize = 18, title='Tau Cortical for CDR = 0.5 (n = 172)')

    ggseg_python.plot_aseg(tau_roi_sig_cdr05.loc[tau_roi_sig_cdr05['sig_count_tau'] > 0]['sig_count_tau'].to_dict(), cmap='hot',
                        background='w', edgecolor='k',  bordercolor='gray', vminmax = [0, 100],
                        ylabel='Frequency (%) of Significance', figsize = (6,4), fontsize = 18, title='Tau subortical for CDR = 0.5 (n = 172)')

    #----- CDR >= 1 -------
    ggseg_python.plot_dk(tau_roi_sig_cdr1.loc[tau_roi_sig_cdr1['sig_count_tau'] > 0]['sig_count_tau'].to_dict(), cmap= 'hot',
                        background='w', edgecolor='k',  bordercolor='gray', vminmax = [0, 100],
                        ylabel='Frequency (%) of Significance', figsize = (6,4), fontsize = 18, title='Tau Cortical for CDR >=1 (n = 43)')

    ggseg_python.plot_aseg(tau_roi_sig_cdr1.loc[tau_roi_sig_cdr1['sig_count_tau'] > 0]['sig_count_tau'].to_dict(), cmap='hot',
                        background='w', edgecolor='k',  bordercolor='gray', vminmax = [0, 100],
                        ylabel='Frequency (%) of Significance', figsize = (6,4), fontsize = 18, title='Tau subortical for CDR >=1 (n = 43)')

    return mri_roi_sig_hc, mri_roi_sig_precl, mri_roi_sig_cdr05, mri_roi_sig_cdr1, amyloid_roi_sig_hc, amyloid_roi_sig_precl, amyloid_roi_sig_cdr05, amyloid_roi_sig_cdr1, tau_roi_sig_hc, tau_roi_sig_precl, tau_roi_sig_cdr05, tau_roi_sig_cdr1


    
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


reject_corr_ADNI = reject_corr.loc[reject_corr.dataset == 'ADNI'].reset_index(drop = True)

mri_roi_sig_hc_ADNI, mri_roi_sig_precl_ADNI, mri_roi_sig_cdr05_ADNI, mri_roi_sig_cdr1_ADNI, amyloid_roi_sig_hc_ADNI, amyloid_roi_sig_precl_ADNI, amyloid_roi_sig_cdr05_ADNI, amyloid_roi_sig_cdr1_ADNI, tau_roi_sig_hc_ADNI, tau_roi_sig_precl_ADNI, tau_roi_sig_cdr05_ADNI, tau_roi_sig_cdr1_ADNI = freq_of_sig_disease_cat(reject_corr_ADNI, MRI_vol_cols, amyloid_SUVR_cols, tau_SUVR_cols)



reject_corr_KARI = reject_corr.loc[reject_corr.dataset == 'KARI'].reset_index(drop = True)

mri_roi_sig_hc_KARI, mri_roi_sig_precl_KARI, mri_roi_sig_cdr05_KARI, mri_roi_sig_cdr1_KARI, amyloid_roi_sig_hc_KARI, amyloid_roi_sig_precl_KARI, amyloid_roi_sig_cdr05_KARI, amyloid_roi_sig_cdr1_KARI, tau_roi_sig_hc_KARI, tau_roi_sig_precl_KARI, tau_roi_sig_cdr05_KARI, tau_roi_sig_cdr1_KARI = freq_of_sig_disease_cat(reject_corr_KARI, MRI_vol_cols, amyloid_SUVR_cols, tau_SUVR_cols)


#deviation_atlas_maps(temp_dev_mmvae, MRI_vol_cols, amyloid_SUVR_cols, tau_SUVR_cols)

#----------------------------------------
#---------------------------------------

