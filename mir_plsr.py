from functions import conduct_pls, plot_q_t_plot
from optimize_mir import get_wavenumber_range

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from sklearn.cross_decomposition import PLSRegression
from scipy.stats import f
from sklearn.model_selection import cross_val_predict, LeaveOneOut, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from math import sqrt

############# ----INPUTS---- ##############
cal_file_location = "data/dil+infogest_mir_noPrRice_conc.csv"
val_file_location = "data/infogest_validation_mir.csv"

start_WN = 3998
end_WN = 800

sg_smoothing_point = 31
sg_derivative = 1
sg_polynomial = 2

no_of_components = 5

sample_presentation = "Supernatant"
#sample_presentation = "Turbid"
#################

def format_df(df):

    ### This function formats df for further analysis

    #Drop columns and rename
    drop_columns = ['Technical_rep']
    df = df.drop(drop_columns,  axis= 1)
    df.rename(columns={"Unnamed: 0": "sample_id"}, inplace = True)

    #Change wavenumber to whole numbers
    wavenumbers_old = list(df.columns[7:])
    wavenumbers = list(map(float, wavenumbers_old))
    wavenumbers = list(map(round, wavenumbers))
    wavenumbers = list(map(str, wavenumbers))
    df.rename(columns = dict(zip(wavenumbers_old, wavenumbers)), inplace = True)
    
    return df

def convert_to_arrays(df, sample_presentation, wavenumber_region):

    """Converts dataframe in to arrays which can be used to do PLSR"""

    if sample_presentation not in ["Turbid", "Supernatant"]:
        raise("The Argument Sample presentation should either be 'Turbid' or 'Supernatant'")

    df = df[df['supernatant'] == sample_presentation]


    X = df[wavenumber_region].values
    y = df['maltose_concentration'].values

    return X, y


df_cal = pd.read_csv(cal_file_location)
df_val = pd.read_csv(val_file_location)

df_cal= format_df(df_cal)
df_val= format_df(df_val)


#Selecting Wavenumbers and assign x and Y values
wavenumbers = list(df_cal.columns[7:])
wavenumbers_3998_800 = get_wavenumber_range(wavenumbers, start_WN, end_WN)

#X,y arrays - Calibration
X_cal,y_cal = convert_to_arrays(df_cal, sample_presentation, wavenumbers_3998_800)
X_cal = savgol_filter(X_cal, sg_smoothing_point, polyorder=sg_polynomial, deriv= sg_derivative)

#X.y Arrays - External Validation
X_val,y_val = convert_to_arrays(df_val, sample_presentation, wavenumbers_3998_800)
X_val = savgol_filter(X_val, sg_smoothing_point, polyorder=sg_polynomial, deriv= sg_derivative)

#Apply PLSR
plsr = PLSRegression(n_components=no_of_components)
plsr.fit(X_cal, y_cal)
y_c = np.ravel(plsr.predict(X_cal))

# Cross-validation
loocv = LeaveOneOut()
y_cv = np.ravel(cross_val_predict(plsr, X_cal, y_cal, cv=loocv))

#External Validation
y_ev = np.ravel(plsr.predict(X_val))

# Calculate scores for calibration, cross-validation, and external-validation
score_c = r2_score(y_cal, y_c)
score_cv = r2_score(y_cal, y_cv)
score_ev = r2_score(y_val, y_ev)

# Calculate RMSE for calibration, cross-validation, and external-validation
rmse_c = sqrt(mean_squared_error(y_cal, y_c))
rmse_cv = sqrt(mean_squared_error(y_cal, y_cv))
rmse_ev = sqrt(mean_squared_error(y_val, y_ev))

# Calculate MAE for calibration, cross-validation, and external-validation
err = (y_ev-y_val)*100/y_val
df_err = pd.DataFrame({'A': y_val, 'B': err})
print(df_err)


#Print stats
print('R2 calib: %5.3f'  % score_c)
print('R2 CV: %5.3f'  % score_cv)
print('R2 EV: %5.3f'  % score_ev)
print('RMSE calib: %5.3f' % rmse_c)
print('RMSE CV: %5.3f' % rmse_cv)
print('RMSE EV: %5.3f' % rmse_ev)

#y_c, y_cv, score_c, score_cv, rmse_c, rmse_cv, x_load = conduct_pls(ncomp, X_cal_sn, y_cal_sn)



# #plot y range values 
# rangey = max(y) - min(y)
# rangex_c = max(y_c) - min(y_c)
# rangex_cv = max(y_cv) - min(y_cv)

# # Fit a line to the CV vs response
# z_c = np.polyfit(y, y_c, 1)
# z_cv = np.polyfit(y, y_cv, 1)


# with plt.style.context(('ggplot')):
#     fig, ax = plt.subplots(figsize=(9, 5))
#     ax.scatter(y_c, y, c='red', edgecolors='k')
#     #Plot the best fit line (calibration)
#     ax.plot(np.polyval(z_c,y), y, c='blue', linewidth=1)
#     #Plot the best fit line (cross validation)
#     ax.plot(np.polyval(z_cv,y), y, c='yellow', linewidth=1)
#     #Plot the ideal 1:1 line
#     ax.plot(y, y, color='green', linewidth=1)
#     plt.title('$R^{2}$ (CV): '+str(round(score_cv,2)))
#     plt.xlabel('Predicted Maltose Concentration ()')
#     plt.ylabel('Measured Maltose Concentration ()')

#     plt.show()
    

# print('R2 calib: %5.3f'  % score_c)
# print('R2 CV: %5.3f'  % score_cv)
# print('RMSE calib: %5.3f' % rmse_c)
# print('RMSE CV: %5.3f' % rmse_cv)