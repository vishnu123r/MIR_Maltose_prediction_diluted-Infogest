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

from kennard_stone import train_test_split

from math import sqrt

import os


def pls_explained_variance(pls, X, Y_true, do_plot=False):
    r2 = np.zeros(pls.n_components)
    x_transformed = pls.transform(X) # Project X into low dimensional basis
    for i in range(0, pls.n_components):
        Y_pred = (np.dot(x_transformed[:, i][:, np.newaxis],
                        pls.y_loadings_[:, i][:, np.newaxis].T) * pls._y_std   
                + pls._y_mean)
        r2[i] = np.round(r2_score(Y_true, Y_pred)*100, 2)
        overall_r2 = np.round(r2_score(Y_true, pls.predict(X))*100,2)  # Use all components together

    #x explained variance
    tot_variance_x = sum(np.var(X, axis=0))
    variance_x_score = np.var(x_transformed, axis=0)
    x_explained_variance = np.round(variance_x_score*100/tot_variance_x, 2)

    if do_plot:
        component = np.arange(pls.n_components) + 1
        plt.plot(component, r2, '.-')
        plt.xticks(component)
        plt.xlabel('PLS Component #'), plt.ylabel('r2')
        plt.title(f'Summed individual r2: {np.sum(r2):.3f}, '
                f'Overall r2: {overall_r2:.3f}')
        plt.show()

    return x_explained_variance, r2, overall_r2

if __name__ == "__main__":

    ############# ----INPUTS---- ##############
    cal_file_location = "data/dil+infogest_mir_noPr(Infogest)_conc_NoNewSamples.csv"
    val_file_location = "data/infogest_validation_mir.csv"
    project_name = "ForPaper"

    exp_type = "All"
    starch = "All"
    y_variable = "time"

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
    #X,y arrays - Calibration
    X_cal,y_cal = convert_to_arrays(df_cal, wavenumbers, y_variable = y_variable)
    if sg_derivative != 0 and sg_smoothing_point != 0:
        print('Applying SG filter')
        X_cal = apply_sgfilter(X = X_cal, wavenumber_region = wavenumbers, window_length = sg_smoothing_point, poly_order = sg_polynomial, deriv = sg_derivative)
    #X_cal = savgol_filter(X_cal, sg_smoothing_point, polyorder=sg_polynomial, deriv= sg_derivative)



    def get_df_sg(df_cal,X_cal, project_name, wavenumbers):

        df_other = df_cal.iloc[:,0:8].reset_index()
        df_other.drop(['index'], axis=1, inplace=True)
        df_sg = pd.DataFrame(X_cal, columns = wavenumbers)
        df_sg = pd.concat([df_other, df_sg], axis=1)

        df_sg.to_excel("data/sg_" + project_name + ".xlsx", index=False)

        return df_sg



    X_cal, X_val, y_cal, y_val = train_test_split(X_cal, y_cal, test_size=0.2, random_state=42)

    # #X.y Arrays - External Validation
    # X_val,y_val = convert_to_arrays(df_val, wavenumbers, y_variable = y_variable)
    # X_val = apply_sgfilter(X_val, wavenumbers, sg_smoothing_point, sg_polynomial, sg_derivative)
    # #X_val = savgol_filter(X_val, sg_smoothing_point, polyorder=sg_polynomial, deriv= sg_derivative)


    #Apply PLSR
    plsr = PLSRegression(n_components=no_of_components, scale = False)
    plsr.fit(X_cal, y_cal)
    y_c = np.ravel(plsr.predict(X_cal))

    # Cross-validation
    loocv = LeaveOneOut()
    y_cv = np.ravel(cross_val_predict(plsr, X_cal, y_cal, cv=loocv))

    #External Validation
    #y_ev = np.ravel(plsr.predict(X_val))

    # Calculate scores for calibration, cross-validation, and external-validation
    score_c = r2_score(y_cal, y_c)
    score_cv = r2_score(y_cal, y_cv)
    #score_ev = r2_score(y_val, y_ev)

    # Calculate RMSE for calibration, cross-validation, and external-validation
    rmse_c = sqrt(mean_squared_error(y_cal, y_c))
    rmse_cv = sqrt(mean_squared_error(y_cal, y_cv))
    #rmse_ev = sqrt(mean_squared_error(y_val, y_ev))

    # Calculate MAE for calibration, cross-validation, and external-validation
    mae_c = mean_absolute_error(y_cal, y_c)
    mae_cv = mean_absolute_error(y_cal, y_cv)
    ##mae_ev = mean_absolute_error(y_val, y_ev)
    # err = (y_ev-y_val)*100/y_val
    # df_err = pd.DataFrame({'Actual_external_val': y_val, 'MAEev': err})

    #calculate standard error of the estimate
    se_c = np.std(y_cal)
    se_cv = np.std(y_cal)
    #se_ev = np.std(y_val)

    #RPD values
    rpd_c = se_c/rmse_c
    rpd_cv = se_cv/rmse_cv
    #rpd_ev = se_ev/rmse_ev


    #Print stats
    print("Starch type: ", starch)
    print("sample_size: ", X_cal.shape[0])
    print("\n")
    print('R2 calib: %5.3f'  % score_c)
    print('R2 CV: %5.3f'  % score_cv)
    #print('R2 EV: %5.3f'  % score_ev)
    print("\n")
    print('RMSE calib: %5.3f' % rmse_c)
    print('RMSE CV: %5.3f' % rmse_cv)
    ##print('RMSE EV: %5.3f' % rmse_ev)
    print("\n")
    print('MAE calib: %5.3f' % mae_c)
    print('MAE CV: %5.3f' % mae_cv)
    #print('MAE EV: %5.3f' % mae_ev)
    print("\n")
    print('RPD calib: %5.3f' % rpd_c)
    print('RPD CV: %5.3f' % rpd_cv)
    #print('RPD EV: %5.3f' % rpd_ev)
    print("\n")


    def get_peaks(loadings, height, prominence):
        positive_peaks,_ = find_peaks(loadings, height = height, prominence = prominence)
        negative_peaks,_ = find_peaks(-loadings, height = height, prominence = prominence)
        peaks = np.concatenate((positive_peaks, negative_peaks))

        return peaks





    path = f"output/{project_name}/{y_variable}/images/{exp_type}/loadings/{starch}/{sample_presentation}/{wavenumbers[0]}-{wavenumbers[-1]}/{sg_derivative}sg{sg_smoothing_point}"
    if not os.path.exists(path):
        os.makedirs(path)
    else: 
        filelist = [f for f in os.listdir(path)]
        for f in filelist:
            os.remove(os.path.join(path, f))

    stats_string = f"Starch type: {starch} \n sample_size: {X_cal.shape[0]} \n \n R2 calib: {score_c} \n R2 CV: {score_cv} \n \n RMSE calib: {rmse_c} \n RMSE CV: {rmse_cv} \n \n MAE calib: {mae_c} \n MAE CV: {mae_cv} \n \n RPD calib: {rpd_c} \n RPD CV: {rpd_cv}"
    with open(path + f'/stats_{starch}.txt', 'w') as f:
        f.write(stats_string)

    def create_loadings_plot(starch, y_variable, wavenumbers, pls, X, Y_true, txt_string, tick_distance, sg_smoothing_point, sg_derivative, height, prominence, peaks_on = True):
        
        if y_variable not in ["maltose_concentration", "time", "starch_digestibility"]:
            raise("The Argument Sample presentation should either be 'maltose_concentration', 'starch_digestibility' or 'time'")

        x_load = pls.x_loadings_

        x_explained_variance, y_explained_variance, overall_r2 = pls_explained_variance(plsr, X, Y_true)

        for comp in range(x_load.shape[1]):

            factor_load = x_load[:,comp]
            x_exp_var = x_explained_variance[comp]
            y_exp_var = y_explained_variance[comp]

            fig, ax = plt.subplots()
            wavenumbers = list(map(int,wavenumbers))
            ax.plot(wavenumbers, factor_load)

            #assigning the peaks
            if peaks_on:
                peaks = get_peaks(factor_load, height = height, prominence = prominence)
                for peak in peaks:
                    ax.plot(wavenumbers[peak], factor_load[peak], "o", color = "red")
                    ax.annotate(wavenumbers[peak], xy = (wavenumbers[peak], factor_load[peak]), xytext = (wavenumbers[peak] + 30, factor_load[peak]+0.0025), size =5)

            ax.set_xlabel('Wavenumber (cm-1)')
            ax.set_ylabel(f"Factor {comp+1} [{x_exp_var:.2f}%, {y_exp_var:.2f}%]")

            ax.set_title(f"{sample_presentation}")

            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.set_xticks(wavenumbers[::tick_distance])
            ax.set_xticklabels(wavenumbers[::tick_distance])
            ax.invert_xaxis()
            ax.set_xlim(wavenumbers[0], wavenumbers[-1])

            plt.axhline(0, color='black', linewidth = 0.5)
            file_name = "/Load_{0}_{1}_{2}_{3}_{4}sg{5}.png".format(comp+1, starch, wavenumbers[0], wavenumbers[-1], sg_derivative, sg_smoothing_point)
            

            plt.savefig(path+file_name, dpi = 1000)

        with open(path + '/parameters.txt', 'w') as f:
            f.write(txt_string)

        loadings_string = f"Starch_type: {starch} \nX explained variance: {x_explained_variance} \nY explained variance: {y_explained_variance} \nOverall r2: {overall_r2}"
        with open(path + '/expl_variance.txt', 'w') as f:
            f.write(loadings_string)


    create_loadings_plot(starch = starch, y_variable = y_variable, 
    wavenumbers = wavenumbers, pls = plsr, X = X_cal, Y_true =y_cal, txt_string=txt_string, tick_distance = tick_distance, 
    sg_derivative=sg_derivative, sg_smoothing_point=sg_smoothing_point, height = loadings_height, prominence = prominence, peaks_on=peaks_on)






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