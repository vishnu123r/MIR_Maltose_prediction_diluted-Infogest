from sys import stdout

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from scipy.signal import savgol_filter, find_peaks

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, LeaveOneOut, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
from math import sqrt

import os

from functions import format_df

class PlsrMir(object):
    
    def __init__(self, df, project_name, spectra_begin = 8):
        self.df = df
        self.project_name = project_name
        self.wavenumbers = list(df.columns[spectra_begin:])
        self.wavenumbers = list(map(int, self.wavenumbers))
        self.other_vars = list(df.columns[0:spectra_begin])
    
    def _ready_img_folder(self, y_variable, exp_type, img_type, starch, sample_presentation, wavenumber_list, sg_derivative, sg_smoothing_points, sg_polynomial):
        
        """This function creates a folder or removes all the files if folder exists"""
        
        path = f"output/{self.project_name}/{y_variable}/images/{exp_type}/{img_type}/{starch}/{sample_presentation}/{wavenumber_list[0]}-{wavenumber_list[-1]}/der{sg_derivative}smooth{sg_smoothing_points}poly{sg_polynomial}"
        
        if not os.path.exists(path):
            os.makedirs(path)
        else: 
            filelist = [file for file in os.listdir(path)]
            for file in filelist:
                os.remove(os.path.join(path, file))

        return path
    
    def _get_wavenumber_range(self, wavenumber_start = 3998, wavenumber_end = 800):
        
        """Gets the wavenumbers for analysis"""
        
        if wavenumber_start < wavenumber_end:
            raise("Starting wavenumber is higher than ending wavenumber")    
        
        wavenumber_for_analysis = []
        for wavenumber in self.wavenumbers:
            if wavenumber <= int(wavenumber_start) and wavenumber >= int(wavenumber_end):
                wavenumber_for_analysis.append(str(wavenumber))

        return wavenumber_for_analysis
    
    def _convert_to_arrays(self, sample_presentation, starch, exp_type, wavenumber_region = (3998, 800), y_variable = 'maltose_concentration'):

        """Converts dataframe in to arrays which can be used to do PLSR"""

        if sample_presentation not in ["Turbid", "Supernatant"]:
            raise("The Argument Sample presentation should either be 'Turbid' or 'Supernatant'")
        
        starch_list = list(df_cal['starch'].unique()) + ["All"]
        exp_list = list(df_cal['exp_type'].unique()) + ["All"]
        
        if starch not in starch_list or exp_type not in exp_list:
            raise("The Argument for starch or exy_type is not valid")
        
        if starch != "All":
            df_subset = self.df[self.df["starch"] == starch]

        if exp_type != "All":
            df_subset = df_subset[df_subset["exp_type"] == exp_type]
        
        df_subset = df_subset[df_subset['supernatant'] == sample_presentation]
        wavenumber_list = self._get_wavenumber_range(wavenumber_region[0],wavenumber_region[1])
        
        X = df_subset[wavenumber_list].values
        y = df_subset[y_variable].values

        wavenumber_list = list(map(int, wavenumber_list))
        
        return X, y, wavenumber_list
    
    def _get_peaks(self, loadings, height):
        
        """ Find peaks of the loadings plot a specific height"""
        
        positive_peaks,_ = find_peaks(loadings, height = height, prominence = 0.1)
        negative_peaks,_ = find_peaks(-loadings, height = height, prominence = 0.1)
        peaks = np.concatenate((positive_peaks, negative_peaks))

        return peaks
    
    def _raise_value_error_y(self, y_variable):
        if y_variable not in self.other_vars:
            raise ValueError(f"The Argument '{y_variable}' is not mentioned in the columns. Available column names are {str(self.other_vars)}")
    
    def create_loadings_plot(self, components, starch, y_variable, wavenumber_region, sample_presentation, exp_type, sg_smoothing_points, sg_derivative, sg_polynomial, peaks_on = True, height = 0.1):
        
        self._raise_value_error_y(y_variable)
        
        X, y, wavenumber_list = self._convert_to_arrays(sample_presentation = sample_presentation, starch = starch, exp_type = exp_type, wavenumber_region = wavenumber_region, y_variable = y_variable)
        path = self._ready_img_folder(y_variable = y_variable, exp_type = exp_type, img_type = "Loadings", starch = starch, sample_presentation = sample_presentation, wavenumber_list = wavenumber_list, sg_derivative = sg_derivative, sg_smoothing_points = sg_smoothing_points, sg_polynomial=sg_polynomial)
        y_c, y_cv, score_c, score_cv, rmse_c, rmse_cv, x_load = self._conduct_pls(components = components, X = X, y = y) 
        
        for comp in range(x_load.shape[1]):

            factor_load = x_load[:,comp]
            file_name = f"/Load_{comp+1}_{starch}_{wavenumber_list[0]}_{wavenumber_list[-1]}_{sg_derivative}sg{sg_smoothing_points}.png"
            
            fig = self._plot_loadings(factor_load = factor_load, comp = comp, wavenumber_list = wavenumber_list, starch = starch, y_variable = y_variable, peaks_on = True, height = 0.1)
            fig.savefig(path+file_name, dpi = 1000)

    
        txt_string = " Project name: " + self.project_name + "\n" + " exp_type: " + exp_type + "\n" + " starch: " + starch + "\n" + " y_variable: " + y_variable + "\n" + " start_WN: " + str(wavenumber_region[0]) + "\n" + " end_WN: " + str(wavenumber_region[1]) + "\n" + " sg_smoothing_point: " + str(sg_smoothing_points) + "\n" + " sg_derivative: " + str(sg_derivative) + "\n"  + " sg_polynomial: " + str(sg_polynomial) + "\n" + " no_of_components: " + str(components) + "\n" + " sample_presentation: "  + sample_presentation + "\n"
        with open(path + '/parameters.txt', 'w') as f:
            f.write(txt_string)

    def _plot_loadings(self, factor_load, comp, wavenumber_list, starch, y_variable, peaks_on = True, height = 0.1):
        
        """This function plots the loadings with their peaks and saves them"""
        
        fig, ax = plt.subplots()
        ax.plot(wavenumber_list, factor_load)

        #assigning the peaks
        if peaks_on:
            peaks = self._get_peaks(factor_load, height = height)
            for peak in peaks:
                ax.plot(wavenumber_list[peak], factor_load[peak], "o", color = "red")
                ax.annotate(wavenumber_list[peak], xy = (wavenumber_list[peak], factor_load[peak]), xytext = (wavenumber_list[peak] + 30, factor_load[peak]+0.005), size =5)

        ax.set_xlabel('Wavenumber (cm-1)')
        ax.set_ylabel(f'Loadings (Factor {comp+1})')

        ax.set_title(f"{starch}-{y_variable}")
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        tick_distance = int(len(wavenumber_list)/4.25)
        ax.set_xticks(wavenumber_list[::tick_distance])
        ax.set_xticklabels(wavenumber_list[::tick_distance])
        
        ax.invert_xaxis()
        ax.set_xlim(wavenumber_list[0], wavenumber_list[-1])

        plt.axhline(0, color='black', linewidth = 0.5)
        
        return fig
    
    def _conduct_pls(self, components, X, y):
        
        """Conducts PLS and returns values"""
        
        # Define PLS object with optimal number of components
        pls_opt = PLSRegression(n_components= components, scale = False)
    
        # For to the entire dataset
        pls_opt.fit(X, y)
        y_c = pls_opt.predict(X)
        
        #Loadings
        x_load = pls_opt.x_loadings_
    
        # Cross-validation
        loocv = LeaveOneOut()
        y_cv = cross_val_predict(pls_opt, X, y, cv=loocv)
    
        # Calculate scores for calibration and cross-validation
        score_c = r2_score(y, y_c)
        score_cv = r2_score(y, y_cv)
    
        # Calculate mean squared error for calibration and cross validation
        rmse_c = sqrt(mean_squared_error(y, y_c))
        rmse_cv = sqrt(mean_squared_error(y, y_cv))
        
        # print('No of components: {}'.format(components))
        # print('R2 calib: %5.3f'  % score_c)
        # print('R2 CV: %5.3f'  % score_cv)
        # print('RMSE calib: %5.3f' % rmse_c)
        # print('RMSE CV: %5.3f' % rmse_cv)
        
        return (y_c, y_cv, score_c, score_cv, rmse_c, rmse_cv, x_load)
    
    
if __name__ == '__main__':
    df_cal = pd.read_csv("data/dil+infogest_mir_noPr_conc.csv")
    df_cal= format_df(df_cal)
    
    plsr = PlsrMir(df_cal, "Trial")
    plsr.create_loadings_plot(components = 4, starch = "Potato", y_variable = "tim.e", wavenumber_region = (1499,800), sample_presentation = "Supernatant"
                              , exp_type = "dil", sg_smoothing_points = 21, sg_derivative = 2, sg_polynomial = 3)