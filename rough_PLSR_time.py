from functions import conduct_pls, plot_q_t_plot, convert_to_arrays, format_df, apply_sgfilter 
from optimize_mir import get_wavenumber_range

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from scipy.stats import f
from sklearn.model_selection import cross_val_predict, LeaveOneOut, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from kennard_stone import train_test_split

from math import sqrt

import os


from functions import _pls_explained_variance

############# ----INPUTS---- ##############
cal_file_location = "data/dil+infogest_mir_noPr(Infogest)_conc_NoNewSamples.csv"
val_file_location = "data/infogest_validation_mir.csv"
project_name = "ForPaper"

exp_type = "All"
starch = "All"
y_variable = "maltose_concentration"

start_WN = 3998
end_WN = 800

sg_smoothing_point = 0
sg_derivative = 0
sg_polynomial = 2

no_of_components = 12
sample_presentation = "Supernatant"
sample_presentation = "Turbid"


tick_parameter = 4.25
loadings_height = 0.04
prominence = 0
peaks_on = False

txt_string = " cal_file_location: " + cal_file_location + "\n" + " val_file_location: " + val_file_location + "\n" + " exp_type: " + exp_type + "\n" + " starch: " + starch + "\n" + " y_variable: " + y_variable + "\n" + " start_WN: " + str(start_WN) + "\n" + " end_WN: " + str(end_WN) + "\n" + " sg_smoothing_point: " + str(sg_smoothing_point) + "\n" + " sg_derivative: " + str(sg_derivative) + "\n"  + " sg_polynomial: " + str(sg_polynomial) + "\n" + " no_of_components: " + str(no_of_components) + "\n" + " sample_presentation: "  + sample_presentation + "\n" + " loadings_height: " + str(loadings_height) + "\n"

#################


if sg_derivative != 0 or sg_smoothing_point != 0:
    print("Life is good")

quit()

df_cal = pd.read_csv(cal_file_location)
#df_val = pd.read_csv(val_file_location)

df_cal= format_df(df_cal)
#df_val= format_df(df_val)

if starch != "All":
    df_cal = df_cal[df_cal["starch"] == starch]

if exp_type != "All":
    df_cal = df_cal[df_cal["exp_type"] == exp_type]

if sample_presentation != "All":
    df_cal = df_cal[df_cal["supernatant"] == sample_presentation]

#Selecting Wavenumbers and assign x and Y values
wavenumbers = list(df_cal.columns[9:])
wavenumbers = get_wavenumber_range(wavenumbers, start_WN, end_WN)

tick_distance = int(len(wavenumbers)/tick_parameter)


y_time = df_cal['time'].to_numpy()
y_time = y_time.ravel()



#X,y arrays - Calibration
X_cal,y_cal = convert_to_arrays(df_cal, wavenumbers, y_variable = y_variable)
if sg_derivative != 0 and sg_smoothing_point != 0:
    print('Applying SG filter')
    X_cal = apply_sgfilter(X = X_cal, wavenumber_region = wavenumbers, window_length = sg_smoothing_point, poly_order = sg_polynomial, deriv = sg_derivative)

X_cal, X_val, y_cal, y_val, y_time_cal, y_time_val = train_test_split(X_cal, y_cal, y_time, test_size=0.2, random_state=42)


quit()


#Apply PLSR
plsr = PLSRegression(n_components=no_of_components, scale = False)
plsr.fit(X_cal, y_cal)
y_pred_plsr = np.ravel(plsr.predict(X_cal))
score = r2_score(y_cal, y_pred_plsr)


# #leave one out cross validation
# loo = LeaveOneOut()
# y_cv_original = cross_val_predict(plsr, X_cal, y_cal, cv=loo)
# score_cv = r2_score(y_cal, y_cv_original)

#get x-scores
X_scores = plsr.x_scores_
#apply linear regression to X_scores and y_cal
reg = LinearRegression().fit(X_scores, y_time_cal)
y_pred_lr = reg.predict(X_scores)
#score
r2 = r2_score(y_time_cal, y_pred_lr)
print(score, r2)


#PLSR for time and X
plsr_time = PLSRegression(n_components=no_of_components, scale = False)
plsr_time.fit(X_cal, y_time_cal)
y_pred_plsr_time = np.ravel(plsr_time.predict(X_cal))
score_time = r2_score(y_time_cal, y_pred_plsr_time)

#cross validation
loo = LeaveOneOut()
y_cv_time = cross_val_predict(plsr_time, X_cal, y_time_cal, cv=loo)
score_cv_time = r2_score(y_time_cal, y_cv_time)

print(score_time, score_cv_time)

quit()
################################################

#apply multiple linear regression from X-scores and y_cal with cross validation
plsr = PLSRegression(n_components=no_of_components, scale = False)\

# Cross-validation
loo = LeaveOneOut()
loo.get_n_splits(X_cal)
y_cv = []
for train_index, test_index in loo.split(X_cal):
        X_train, X_test = X_cal[train_index], X_cal[test_index]
        y_train, y_test = y_cal[train_index], y_cal[test_index]
        y_train_time, y_test_time = y_time[train_index], y_time[test_index]

        #get PLSR scores
        plsr.fit(X_train, y_train)
        X_scores = plsr.x_scores_

        #apply linear regression to X_scores and y_train
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        #convert y_pred to a single value
        y_pred = np.ravel(y_pred)
        y_pred = y_pred[0]
        y_cv.append(y_pred)


score_cv = r2_score(y_cal, y_cv_original)
score_pred_cv = r2_score(y_cal, y_cv)


print(score_cv, score_pred_cv)


quit()



#get explained variance
x_explained_variance, y_explained_variance, overall_r2 = pls_explained_variance(plsr, X_cal, y_cal)


comp = 1

#PLot first component of X-scores
plt.figure(figsize=(10,5))
plt.plot(plsr.x_scores_[:,comp], y_cal, 'o', color = 'blue', label = 'Calibration')
plt.plot(plsr.x_scores_[:,comp], y_cv, 'o', color = 'red', label = 'Cross-validation')
plt.xlabel(f'X-scores - {x_explained_variance[comp]}')
plt.ylabel(f'{y_variable} - {y_explained_variance[comp]}')
plt.legend()
plt.show()
