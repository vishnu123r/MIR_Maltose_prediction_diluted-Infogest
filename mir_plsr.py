from functions import conduct_pls, plot_q_t_plot, convert_to_arrays, format_df, apply_sgfilter 
from optimize_mir import get_wavenumber_range

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

from sklearn.cross_decomposition import PLSRegression
from scipy.stats import f
from sklearn.model_selection import cross_val_predict, LeaveOneOut, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from math import sqrt

import os

############# ----INPUTS---- ##############
cal_file_location = "data/dil+infogest_mir_noPr_conc.csv"
val_file_location = "data/infogest_validation_mir.csv"

exp_type = "dil"
starch = "Pregelled Maize Starch"
y_variable = "maltose_concentration"

start_WN = 1499
end_WN = 800

sg_smoothing_point = 21
sg_derivative = 2
sg_polynomial = 2

no_of_components = 4
#sample_presentation = "Supernatant"
sample_presentation = "Turbid"


tick_distance = 80
loadings_height = 0.1

txt_string = " cal_file_location: " + cal_file_location + "\n" + " val_file_location: " + val_file_location + "\n" + " exp_type: " + exp_type + "\n" + " starch: " + starch + "\n" + " y_variable: " + y_variable + "\n" + " start_WN: " + str(start_WN) + "\n" + " end_WN: " + str(end_WN) + "\n" + " sg_smoothing_point: " + str(sg_smoothing_point) + "\n" + " sg_derivative: " + str(sg_derivative) + "\n"  + " sg_polynomial: " + str(sg_polynomial) + "\n" + " no_of_components: " + str(no_of_components) + "\n" + " sample_presentation: "  + sample_presentation + "\n" + " tick_distance: " + str(tick_distance) + "\n" + " loadings_height: " + str(loadings_height) + "\n"

#################
df_cal = pd.read_csv(cal_file_location)
df_val = pd.read_csv(val_file_location)

df_cal= format_df(df_cal)
df_val= format_df(df_val)

# if starch != "All":
#     df_cal = df_cal[df_cal["starch"] == starch]

# if exp_type != "All":
#     df_cal = df_cal[df_cal["exp_type"] == exp_type]

# #Selecting Wavenumbers and assign x and Y values
# wavenumbers = list(df_cal.columns[7:])
# wavenumbers = get_wavenumber_range(wavenumbers, start_WN, end_WN)

# #X,y arrays - Calibration
# X_cal,y_cal = convert_to_arrays(df_cal, sample_presentation, wavenumbers, y_variable = y_variable)
# X_cal = apply_sgfilter(X_cal, wavenumbers, sg_smoothing_point, sg_polynomial, sg_derivative)
# #X_cal = savgol_filter(X_cal, sg_smoothing_point, polyorder=sg_polynomial, deriv= sg_derivative)

# #X.y Arrays - External Validation
# X_val,y_val = convert_to_arrays(df_val, sample_presentation, wavenumbers, y_variable = y_variable)
# X_val = apply_sgfilter(X_val, wavenumbers, sg_smoothing_point, sg_polynomial, sg_derivative)
# #X_val = savgol_filter(X_val, sg_smoothing_point, polyorder=sg_polynomial, deriv= sg_derivative)


# #Apply PLSR
# plsr = PLSRegression(n_components=no_of_components, scale = False)
# plsr.fit(X_cal, y_cal)
# y_c = np.ravel(plsr.predict(X_cal))

# # Cross-validation
# loocv = LeaveOneOut()
# y_cv = np.ravel(cross_val_predict(plsr, X_cal, y_cal, cv=loocv))

# #External Validation
# y_ev = np.ravel(plsr.predict(X_val))

# # Calculate scores for calibration, cross-validation, and external-validation
# score_c = r2_score(y_cal, y_c)
# score_cv = r2_score(y_cal, y_cv)
# score_ev = r2_score(y_val, y_ev)

# # Calculate RMSE for calibration, cross-validation, and external-validation
# rmse_c = sqrt(mean_squared_error(y_cal, y_c))
# rmse_cv = sqrt(mean_squared_error(y_cal, y_cv))
# rmse_ev = sqrt(mean_squared_error(y_val, y_ev))

# # Calculate MAE for calibration, cross-validation, and external-validation
# mae_c = mean_absolute_error(y_cal, y_c)
# mae_cv = mean_absolute_error(y_cal, y_cv)
# mae_ev = mean_absolute_error(y_val, y_ev)
# # err = (y_ev-y_val)*100/y_val
# # df_err = pd.DataFrame({'Actual_external_val': y_val, 'MAEev': err})

# #calculate standard error of the estimate
# se_c = np.std(y_cal)
# se_cv = np.std(y_cal)
# se_ev = np.std(y_val)

# #RPD values
# rpd_c = se_c/rmse_c
# rpd_cv = se_cv/rmse_cv
# rpd_ev = se_ev/rmse_ev


# #Print stats
# print('R2 calib: %5.3f'  % score_c)
# print('R2 CV: %5.3f'  % score_cv)
# print('R2 EV: %5.3f'  % score_ev)
# print("\n")

# print('RMSE calib: %5.3f' % rmse_c)
# print('RMSE CV: %5.3f' % rmse_cv)
# print('RMSE EV: %5.3f' % rmse_ev)
# print("\n")

# print('MAE calib: %5.3f' % mae_c)
# print('MAE CV: %5.3f' % mae_cv)
# print('MAE EV: %5.3f' % mae_ev)
# print("\n")

# print('RPD calib: %5.3f' % rpd_c)
# print('RPD CV: %5.3f' % rpd_cv)
# print('RPD EV: %5.3f' % rpd_ev)
# print("\n")


fig, ax = plt.subplots()

for starch in ["Rice", "Gelose 50", "Gelose 80", "Potato", "Pregelled Maize Starch"]:

    if starch != "All":
        df_cal_new = df_cal[df_cal["starch"] == starch]

    else:
        df_cal_new = df_cal

    if exp_type != "All":
        df_cal_new = df_cal_new[df_cal_new["exp_type"] == exp_type]

    #Selecting Wavenumbers and assign x and Y values
    wavenumbers = list(df_cal_new.columns[7:])
    wavenumbers = get_wavenumber_range(wavenumbers, start_WN, end_WN)
    
    #X,y arrays - Calibration
    X_cal,y_cal = convert_to_arrays(df_cal_new, sample_presentation, wavenumbers, y_variable = y_variable)
    X_cal = apply_sgfilter(X_cal, wavenumbers, sg_smoothing_point, sg_polynomial, sg_derivative)
    #X_cal = savgol_filter(X_cal, sg_smoothing_point, polyorder=sg_polynomial, deriv= sg_derivative)

    rpd_c, rpd_cv, score_c, score_cv, rmse_c, rmse_cv, x_load = conduct_pls(components = no_of_components, X = X_cal, y = y_cal)
    wavenumbers = list(map(int,wavenumbers))
    #plot loadings
    factor_load = x_load[:,0]
    ax.plot(wavenumbers, factor_load)

ax.legend(["Rice", "Gelose 50", "Gelose 80", "Potato", "Pregelled Maize Starch"])
ax.set_xlabel('Wavenumber')
ax.set_ylabel('Loadings')
ax.set_title('Loadings for factor 1')
ax.grid(True)
    
    
plt.show()

quit()

def get_peaks(loadings, height):
    positive_peaks,_ = find_peaks(loadings, height = height)
    negative_peaks,_ = find_peaks(-loadings, height = height)
    peaks = np.concatenate((positive_peaks, negative_peaks))

    return peaks



path = "output/{0}/images/{1}/loadings/{2}/{3}/{4}-{5}/{6}sg{7}".format(y_variable, exp_type, starch, sample_presentation, wavenumbers[0], wavenumbers[-1], sg_derivative, sg_smoothing_point)
if not os.path.exists(path):
    os.makedirs(path)
else: 
    filelist = [f for f in os.listdir(path)]
    for f in filelist:
        os.remove(os.path.join(path, f))


def create_loadings_plot(starch, y_variable, wavenumbers, x_load, txt_string, tick_distance, sg_smoothing_point, sg_derivative, peaks_on = True, height = loadings_height):
    
    if y_variable not in ["maltose_concentration", "time"]:
        raise("The Argument Sample presentation should either be 'maltose_concentration' or 'time'")


    for comp in range(x_load.shape[1]):

        factor_load = x_load[:,comp]

        fig, ax = plt.subplots()
        wavenumbers = list(map(int,wavenumbers))
        ax.plot(wavenumbers, factor_load)

        #assigning the peaks
        if peaks_on:
            peaks = get_peaks(factor_load, height = height)
            for peak in peaks:
                ax.plot(wavenumbers[peak], factor_load[peak], "o", color = "red")
                ax.annotate(wavenumbers[peak], xy = (wavenumbers[peak], factor_load[peak]), xytext = (wavenumbers[peak] + 30, factor_load[peak]+0.01), size =5)

        ax.set_xlabel('Wavenumber (cm-1)')
        ax.set_ylabel('Loadings (Factor {})'.format(comp+1))

        ax.set_title("{}-{}".format(starch, y_variable))

        ax.tick_params(axis='both', which='major', labelsize=5)
        ax.set_xticks(wavenumbers[::tick_distance])
        ax.set_xticklabels(wavenumbers[::tick_distance])
        ax.invert_xaxis()
        ax.set_xlim(wavenumbers[0], wavenumbers[-1])

        plt.axhline(0, color='black', linewidth = 0.5)
        file_name = "/Load_{0}_{1}_{2}_{3}_{4}sg{5}.png".format(comp+1, starch, wavenumbers[0], wavenumbers[-1], sg_derivative, sg_smoothing_point)
        
        plt.savefig(path+file_name, dpi = 1000)

    with open(path + '/parameters.txt', 'w') as f:
        f.write(txt_string)


create_loadings_plot(starch = starch, y_variable = y_variable, 
wavenumbers = wavenumbers, x_load = x_load, txt_string=txt_string, tick_distance = tick_distance, 
sg_derivative=sg_derivative, sg_smoothing_point=sg_smoothing_point)






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