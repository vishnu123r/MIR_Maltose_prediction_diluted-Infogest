from sys import stdout

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from scipy.signal import savgol_filter, find_peaks
 
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, LeaveOneOut, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import os 

from math import sqrt

def format_df(df):

    ### This function formats df for further analysis

    #Drop columns and rename
    df.rename(columns={"Unnamed: 0": "sample_id"}, inplace = True)

    #Change wavenumber to whole numbers
    wavenumbers_old = list(df.columns[9:])
    wavenumbers = list(map(float, wavenumbers_old))
    wavenumbers = list(map(round, wavenumbers))
    wavenumbers = list(map(str, wavenumbers))
    df.rename(columns = dict(zip(wavenumbers_old, wavenumbers)), inplace = True)
    
    return df

def convert_to_arrays(df, wavenumber_region, y_variable = 'maltose_concentration'):

    """Converts dataframe in to arrays which can be used to do PLSR"""


    X = df[wavenumber_region].values
    y = df[y_variable].values

    return X, y

def get_wavenumber_range(wavenumbers_list, wavenumber_start = 3998, wavenumber_end = 800):
    """Gets the wavenumbers for analysis"""
    wavenumbers_int = list(map(int, wavenumbers_list))
    wavenumber_for_analysis = []
    for wavenumber in wavenumbers_int:
        if wavenumber <= wavenumber_start and wavenumber >= wavenumber_end:
            wavenumber_for_analysis.append(str(wavenumber))

    return wavenumber_for_analysis

def apply_sgfilter(X, wavenumber_region, window_length, poly_order, deriv, is_plot = False):
    """
    Apply SG filter to determine X data
    """
    X_sg = savgol_filter(X, window_length = window_length, polyorder = poly_order, deriv = deriv)
    wavenumber_region_int = list(map(int,wavenumber_region))

    if is_plot:
        fig, ax = plt.subplots()
        ax.plot(wavenumber_region_int, X_sg.T)
        ax.set_xlim(wavenumber_region_int[0] , wavenumber_region_int[-1])
        ax.set_xlabel('Wavenumber (cm-1)')
        ax.set_ylabel('D2 Absorbance')
        ax.ticklabel_format(style='sci', axis = 'y')
        plt.show()

    return X_sg

def optimise_pls_cv(X, y, n_comp, plot_components=True):
 
    '''Run PLS including a variable number of components, up to n_comp,
       and calculate MSE
       
        Returns optimum number of components based on minimum MSE'''
 
    mse = []
    component = np.arange(1, n_comp+1)

    for i in component:

        pls = PLSRegression(n_components=i, scale = False)
 
        # Cross-validation
        loocv = LeaveOneOut()
        y_cv = cross_val_predict(pls, X, y, cv=loocv)
        mse.append(mean_squared_error(y, y_cv))
 
        comp = 100*(i)/n_comp

        # Trick to update status on the same line
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")
 
    # Calculate and print the position of minimum in MSE
    msemin = np.argmin(mse)
    optimum_components = msemin + 1 
    print("Suggested number of components: ", optimum_components)
    stdout.write("\n")
 
    if plot_components is True:
        with plt.style.context(('ggplot')):
            plt.plot(component, np.array(mse), '-v', color = 'blue', mfc='blue')
            plt.plot(component[msemin], np.array(mse)[msemin], 'P', ms=10, mfc='red')
            plt.xlabel('Number of PLS components')
            plt.ylabel('MSE')
            plt.xticks(np.arange(min(component), max(component) +1, 2.0))
            plt.title('Optimal number of components for PLS')
            plt.xlim(left=-1)
 
            plt.show()
    
    return optimum_components

def conduct_pls(components, X_cal, X_val, y_cal, y_val, val = False):
    """Conducts PLS and returns values"""
    # Define PLS object with optimal number of components
    pls_opt = PLSRegression(n_components= components, scale = False)
 
    # For to the entire dataset
    pls_opt.fit(X_cal, y_cal)
    y_c = pls_opt.predict(X_cal)

    # Cross-validation
    loocv = LeaveOneOut()
    y_cv = cross_val_predict(pls_opt, X_cal, y_cal, cv=loocv)
    
    if val:
        # For external validation
        y_v = pls_opt.predict(X_val)

        return (y_c, y_cv, y_v)
    
    else:
        y_v = None

        return (y_c, y_cv, y_v)


def loadings_plot():
    pass

def plot_q_t_plot(X, y, ncomp):

    #### To get Q^2 vs T^2 plot
    # Define PLS object
    pls = PLSRegression(n_components=ncomp, scale = False)
    # Fit data
    pls.fit(X, y)
 
    # Get X scores
    T = pls.x_scores_
    # Get X loadings
    P = pls.x_loadings_
 
    # Calculate error array
    Err = X - np.dot(T,P.T)
 
    # Calculate Q-residuals (sum over the rows of the error array)
    Q = np.sum(Err**2, axis=1)
 
    # Calculate Hotelling's T-squared (note that data are normalised by default)
    Tsq = np.sum((pls.x_scores_/np.std(pls.x_scores_, axis=0))**2, axis=1)

    # set the confidence level
    conf = 0.95
 

    # Calculate confidence level for T-squared from the ppf of the F distribution
    Tsq_conf =  f.ppf(q=conf, dfn=ncomp, \
            dfd=X.shape[0])*ncomp*(X.shape[0]-1)/(X.shape[0]-ncomp)
 
    # Estimate the confidence level for the Q-residuals
    i = np.max(Q)+1
    while 1-np.sum(Q>i)/np.sum(Q>0) > conf:
        i -= 1
    Q_conf = i

    ax = plt.figure(figsize=(8,4.5))
    with plt.style.context(('ggplot')):
        plt.plot(Tsq, Q, 'o')
 
        plt.plot([Tsq_conf,Tsq_conf],[plt.axis()[2],plt.axis()[3]],  '--')
        plt.plot([plt.axis()[0],plt.axis()[1]],[Q_conf,Q_conf],  '--')
        plt.xlabel("Hotelling's T-squared")
        plt.ylabel('Q residuals')
 
    plt.show()

def _create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else: 
        filelist = [f for f in os.listdir(path)]
        for f in filelist:
            os.remove(os.path.join(path, f))

def _pls_explained_variance(pls, X, Y_true, do_plot=False):
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

def get_peaks(loadings, height, prominence):
    positive_peaks,_ = find_peaks(loadings, height = height, prominence = prominence)
    negative_peaks,_ = find_peaks(-loadings, height = height, prominence = prominence)
    peaks = np.concatenate((positive_peaks, negative_peaks))

    return peaks

def get_stats(y_true, y_pred):
    
    score = r2_score(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    std_y_true= np.std(y_true)
    rpd = std_y_true/rmse

    return score, rmse, rpd


def get_df_sg(df_cal,X_cal, project_name, wavenumbers):

    df_other = df_cal.iloc[:,0:8].reset_index()
    df_other.drop(['index'], axis=1, inplace=True)
    df_sg = pd.DataFrame(X_cal, columns = wavenumbers)
    df_sg = pd.concat([df_other, df_sg], axis=1)

    df_sg.to_excel("data/sg_" + project_name + ".xlsx", index=False)

    return df_sg

def create_loadings_plot(starch, y_variable, sample_presentation, wavenumbers, pls, X, Y_true, txt_string, tick_distance, sg_smoothing_point, sg_derivative, height, prominence, path, peaks_on = True):
    
    if y_variable not in ["maltose_concentration", "time", "starch_digestibility"]:
        raise("The Argument Sample presentation should either be 'maltose_concentration', 'starch_digestibility' or 'time'")

    x_load = pls.x_loadings_

    x_explained_variance, y_explained_variance, overall_r2 = _pls_explained_variance(pls, X, Y_true)
    
    path = path + "/loadings_plots"
    _create_folder(path)
    
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

