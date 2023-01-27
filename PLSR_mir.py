from sys import stdout

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from scipy.signal import savgol_filter

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, LeaveOneOut, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
from math import sqrt

from functions import format_df

class PlsrMir(object):
    def __init__(self, df, spectra_region = (8,) ):
        self.df = df
        self.wavenumbers = list(df.columns[spectra_region[0]:spectra_region[1]])
    
    def _conduct_pls(components, X, y):
        
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
    
    def _optimise_plsr_factors(self, x, Y, n_comp, plot_components = False):
        '''Run PLS including a variable number of components, up to n_comp, and calculate RMSE
       
        Returns optimum number of components based on minimum RMSE'''
        
        rmse = []
        components = np.arange(1, n_comp+1)

        for component in components:
            
            y_c, y_cv, score_c, score_cv, rmse_c, rmse_cv, x_load = self._conduct_pls(component, x, Y)
            
            rmse.append(rmse_cv)
    
            comp = 100*(component)/n_comp

            # Trick to update status on the same line
            stdout.write("\r%d%% completed" % comp)
            stdout.flush()
        stdout.write("\n")
    
        # Calculate and print the position of minimum in MSE
        rmsemin = np.argmin(rmse)
        optimum_components = rmsemin + 1 
        print("Suggested number of components: ", optimum_components)
        stdout.write("\n")
    
        if plot_components:
            with plt.style.context(('ggplot')):
                plt.plot(component, np.array(rmse), '-v', color = 'blue', mfc='blue')
                plt.plot(component[rmsemin], np.array(rmse)[rmsemin], 'P', ms=10, mfc='red')
                plt.xlabel('Number of PLS components')
                plt.ylabel('RMSE')
                plt.xticks(np.arange(min(component), max(component) +1, 2.0))
                plt.title('Optimal number of components for PLS')
                plt.xlim(left=-1)
    
                plt.show()
        
        return optimum_components
    
    def optimise_plsr(self):
        pass