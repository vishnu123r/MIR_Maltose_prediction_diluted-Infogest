from functions import create_loadings_plot, conduct_pls, plot_q_t_plot, convert_to_arrays, format_df, apply_sgfilter, get_stats, get_peaks
from optimize_mir import get_wavenumber_range

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from sklearn.cross_decomposition import PLSRegression
from scipy.stats import f
from sklearn.model_selection import cross_val_predict, LeaveOneOut, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from kennard_stone import train_test_split


from math import sqrt

import os

if __name__ == "__main__":

    ############# ----INPUTS---- ##############
    cal_file_location = "data/dil+infogest_mir_noPr(Infogest)_conc_NoNewSamples.csv"
    val_file_location = "data/infogest_validation_mir.csv"
    project_name = "NoNewSamples"

    exp_type = "All"
    starch = "All"
    y_variable = "maltose_concentration"

    start_WN = 3998
    end_WN = 800

    sg_smoothing_point = 0
    sg_derivative = 0
    sg_polynomial = 2

    no_of_components = 10
    sample_presentation = "Supernatant"
    sample_presentation = "Turbid"


    tick_parameter = 4.25
    loadings_height = 0.04
    prominence = 0
    peaks_on = False

    txt_string = " cal_file_location: " + cal_file_location + "\n" + " val_file_location: " + val_file_location + "\n" + " exp_type: " + exp_type + "\n" + " starch: " + starch + "\n" + " y_variable: " + y_variable + "\n" + " start_WN: " + str(start_WN) + "\n" + " end_WN: " + str(end_WN) + "\n" + " sg_smoothing_point: " + str(sg_smoothing_point) + "\n" + " sg_derivative: " + str(sg_derivative) + "\n"  + " sg_polynomial: " + str(sg_polynomial) + "\n" + " no_of_components: " + str(no_of_components) + "\n" + " sample_presentation: "  + sample_presentation + "\n" + " loadings_height: " + str(loadings_height) + "\n"

    #################
    df_cal = pd.read_csv(cal_file_location)
    df_cal= format_df(df_cal)

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
    if sg_derivative != 0 or sg_smoothing_point != 0:
        print('Applying SG filter')
        X_cal = apply_sgfilter(X = X_cal, wavenumber_region = wavenumbers, window_length = sg_smoothing_point, poly_order = sg_polynomial, deriv = sg_derivative)


    X_cal, X_val, y_cal, y_val = train_test_split(X_cal, y_cal, test_size=0.2, random_state= 42)

    descriptive_y_cal = pd.DataFrame(y_cal).describe().T
    descriptive_y_val = pd.DataFrame(y_val).describe().T    

    #concat the two dataframes
    descriptive_y = pd.concat([descriptive_y_cal, descriptive_y_val], axis=0)
    print(descriptive_y)



    #Apply PLSR
    y_c, y_cv, y_ev, plsr = conduct_pls(components = no_of_components, X_cal=X_cal, X_val=X_val, y_cal=y_cal, y_val=y_val, val= True)
    score_c, rmse_c, rpd_c = get_stats(y_cal, y_c)
    score_cv, rmse_cv, rpd_cv = get_stats(y_cal, y_cv)
    score_ev, rmse_ev, rpd_ev = get_stats(y_val, y_ev)
    

    #slope, bias
    slope, bias = np.polyfit(y_cal, y_cv, 1)


    #Print stats
    print("Starch type: ", starch)
    print("sample_size: ", X_cal.shape[0])
    print("\n")
    print('R2 calib: %5.3f'  % score_c)
    print('R2 CV: %5.3f'  % score_cv)
    print('R2 EV: %5.3f'  % score_ev)
    print("\n")
    print('RMSE calib: %5.3f' % rmse_c)
    print('RMSE CV: %5.3f' % rmse_cv)
    print('RMSE EV: %5.3f' % rmse_ev)
    print("\n")
    print('RPD calib: %5.3f' % rpd_c)
    print('RPD CV: %5.3f' % rpd_cv)
    print('RPD EV: %5.3f' % rpd_ev)
    print("\n")
    print('Slope: %5.3f' % slope)
    print('Bias: %5.3f' % bias)
    
    path = f"output/{project_name}/{y_variable}/images/{exp_type}/{starch}/{sample_presentation}/{wavenumbers[0]}-{wavenumbers[-1]}/{sg_derivative}sg{sg_smoothing_point}"
    

    

    


    x_load = plsr.x_loadings_
    factor_load = x_load[:,no_of_components - 1]

    fig, ax = plt.subplots(figsize = (8,5))
    wavenumbers = list(map(int,wavenumbers))
    ax.plot(wavenumbers, factor_load)

    #assigning the peaks
    if peaks_on:
        peaks = get_peaks(factor_load, height = loadings_height, prominence = prominence)
        for peak in peaks:
            ax.plot(wavenumbers[peak], factor_load[peak], "o", color = "red")
            ax.annotate(wavenumbers[peak], xy = (wavenumbers[peak], factor_load[peak]), xytext = (wavenumbers[peak] + 30, factor_load[peak]+0.0025), size =5)


    ax.set_xlabel('Wavenumber (cm-1)')
    ax.set_ylabel(f"Factor {no_of_components}")

    ax.set_title(f"{sample_presentation}")

    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xticks(wavenumbers[::tick_distance])
    ax.set_xticklabels(wavenumbers[::tick_distance])
    ax.invert_xaxis()
    ax.set_xlim(wavenumbers[0], wavenumbers[-1])

    plt.axhline(0, color='black', linewidth = 0.5)

    plt.show()


    quit()


    create_loadings_plot(starch = starch, y_variable = y_variable, sample_presentation = sample_presentation,
    wavenumbers = wavenumbers, pls = plsr, X = X_cal, Y_true =y_cal, txt_string=txt_string, tick_distance = tick_distance, 
    sg_derivative=sg_derivative, sg_smoothing_point=sg_smoothing_point, height = loadings_height, prominence = prominence, path = path, peaks_on=peaks_on)

    stats_string = f"Starch type: {starch} \n sample_size: {X_cal.shape[0]} \n \n R2 calib: {score_c} \n R2 CV: {score_cv} \n \n RMSE calib: {rmse_c} \n RMSE CV: {rmse_cv} \n \n RPD calib: {rpd_c} \n RPD CV: {rpd_cv}"
    with open(path + f'/stats_{starch}.txt', 'w') as f:
        f.write(stats_string)


    #plot y predicted vs y true for plsr model
    #plot y range values 
    # rangey = max(y_c) - min(y_c)
    # rangex_c = max(y_c) - min(y_c)
    # rangex_cv = max(y_cv) - min(y_cv)

    # Fit a line to the CV vs response
    z_c = np.polyfit(y_cal, y_c, 1)
    z_cv = np.polyfit(y_cal, y_cv, 1)


    with plt.style.context(('ggplot')):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(y_c, y_cal, c='red', edgecolors='k')
        #Plot the best fit line (calibration)
        ax.plot(np.polyval(z_c,y_cal), y_cal, c='blue', linewidth=1)
        #Plot the best fit line (cross validation)
        ax.plot(np.polyval(z_cv,y_cal), y_cal, c='yellow', linewidth=1)
        #Plot the ideal 1:1 line
        ax.plot(y_cal, y_cal, color='green', linewidth=1)
        plt.title('$R^{2}$ (CV): '+str(round(score_cv,2)))
        plt.xlabel(f'Predicted {y_variable} ()')
        plt.ylabel(f'Measured {y_variable} ()')

        plt.show()
        
