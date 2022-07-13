# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#importing packages 
import pandas as pd

#importing data set ##
claimant = pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/DataSets-Data Pre Processing/DataSets/claimants.csv')

claimant.info() ##to check null or NA value#
claimant.describe() #to check mean , median , IQR##

claimant.shape ##to check rows and columns##

##lets check for null value##
claimant.isna().sum()
claimant.isnull().sum()

claimant.dtypes
claimant.CLMAGE.unique()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#lets find outlier in the CLMAGE ##
sns.boxplot(claimant.CLMAGE);plt.title('Boxplot');plt.show() ## we have some outlier##

##missing value imputation###

from sklearn.impute import SimpleImputer

mode_imputer=SimpleImputer(missing_values=np.nan,strategy='most_frequent')

## Mode imputer for CLMSEX

claimant['CLMSEX'] = pd.DataFrame(mode_imputer.fit_transform(claimant[['CLMSEX']]))

##Mode imputer for CLMINSUR##

claimant['CLMINSUR']= pd.DataFrame(mode_imputer.fit_transform(claimant[['CLMINSUR']]))

##mode imputer for SEATBELT##

claimant['SEATBELT']= pd.DataFrame(mode_imputer.fit_transform(claimant[['SEATBELT']]))


##median imputer for CLMAGE##

median_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
claimant['CLMAGE'] = pd.DataFrame(median_imputer.fit_transform(claimant[['CLMAGE']]))
claimant['CLMAGE'].isnull().sum()

claimant.isnull().sum()

EDA = pd.DataFrame({'columns_name':[claimant.columns],
                    'mean':[claimant.mean()],
                    'median':[claimant.median()],
                    'mode':[claimant.mode()],
                    'std dev':[claimant.std()],
                    'variance':[claimant.var()],
                    'skewness':[claimant.skew()],
                    'kurtosis':[claimant.kurt()]})








