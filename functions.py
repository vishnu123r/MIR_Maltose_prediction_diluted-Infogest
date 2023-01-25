from sys import stdout

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from scipy.signal import savgol_filter
 
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, LeaveOneOut, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

from math import sqrt

def format_df(df):

    ### This function formats df for further analysis

    #Drop columns and rename
    drop_columns = ['Technical_rep']
    df = df.drop(drop_columns,  axis= 1)
    df.rename(columns={"Unnamed: 0": "sample_id"}, inplace = True)

    #Change wavenumber to whole numbers
    wavenumbers_old = list(df.columns[8:])
    wavenumbers = list(map(float, wavenumbers_old))
    wavenumbers = list(map(round, wavenumbers))
    wavenumbers = list(map(str, wavenumbers))
    df.rename(columns = dict(zip(wavenumbers_old, wavenumbers)), inplace = True)
    
    return df

def convert_to_arrays(df, sample_presentation, wavenumber_region, y_variable = 'maltose_concentration'):

    """Converts dataframe in to arrays which can be used to do PLSR"""

    if sample_presentation not in ["Turbid", "Supernatant"]:
        raise("The Argument Sample presentation should either be 'Turbid' or 'Supernatant'")

    df = df[df['supernatant'] == sample_presentation]


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
    
    #Stats for calibration
    score_c = r2_score(y_cal, y_c)
    rmse_c = sqrt(mean_squared_error(y_cal, y_c))
    rpd_c = np.std(y_cal)/rmse_c

    # Cross-validation
    loocv = LeaveOneOut()
    y_cv = cross_val_predict(pls_opt, X_cal, y_cal, cv=loocv)
    
    # Stats for cross validation
    score_cv = r2_score(y_cal, y_cv)
    rmse_cv = sqrt(mean_squared_error(y_cal, y_cv))
    rpd_cv = np.std(y_cal)/rmse_cv

    if val:
        # For external validation
        y_v = pls_opt.predict(X_val)
        score_v = r2_score(y_val, y_v)
        rmse_v = sqrt(mean_squared_error(y_val, y_v))
        rpd_v = np.std(y_val)/rmse_v

        return (rpd_c, rpd_cv, rpd_v, score_c, score_cv, score_v, rmse_c, rmse_cv, rmse_v)
    
    else:
        score_v = None
        rmse_v = None
        rpd_v = None

        return (rpd_c, rpd_cv, rpd_v, score_c, score_cv, score_v, rmse_c, rmse_cv, rmse_v)


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



if __name__ == '__main__':
    drop_columns = ['Technical_rep']

    #Read CSV and Drop columns
    df_old = pd.read_csv("data/dil+infogest_mir_all_conc.csv")
    df = df_old.drop(drop_columns,  axis= 1)
    df.rename(columns={"Unnamed: 0": "sample_id"}, inplace = True)

    #Change wavenumber to whole numbers
    wavenumbers_old = list(df.columns[8:])
    wavenumbers = list(map(float, wavenumbers_old))
    wavenumbers = list(map(round, wavenumbers))
    wavenumbers = list(map(str, wavenumbers))
    df.rename(columns = dict(zip(wavenumbers_old, wavenumbers)), inplace = True)

    #Segment into supernatant and turbid
    df_turbid = df[df['supernatant'] == "Turbid"]
    df_SN = df[df['supernatant'] == "Supernatant"]

    #Selecting Wavenumbers and assign x and Y values
    wavenumbers_3998_800 = get_wavenumber_range(wavenumbers, 3998, 800)
    X = df_SN[wavenumbers_3998_800].values
    #X = savgol_filter(X, 51, 2, 2)
    y = df_SN['maltose_concentration'].values

    ### LOdings Plot
    # rpd_c, rpd_cv, score_c, score_cv, rmse_c, rmse_cv, x_load = conduct_pls(4, X, y)
    # factor1_load = x_load[:,0]

    pls = PLSRegression(n_components=5, scale=False)
    pls.fit(X, y)
    x_load = pls.x_loadings_
    factor1_load = x_load[:,0]

    fig, ax = plt.subplots()
    wavenumbers_3998_800 = list(map(int,wavenumbers_3998_800))
    ax.plot(wavenumbers_3998_800, factor1_load)
    

    ax.set_xlabel('Wavenumber (cm-1)')
    ax.set_ylabel('D2 Absorbance')

    ax.set_title('Loadings Plot')
    

    ax.ticklabel_format(style='sci', axis = 'y')
    ax.set_xticks(wavenumbers_3998_800[::100])
    ax.set_xticklabels(wavenumbers_3998_800[::100])
    ax.invert_xaxis()
    ax.set_xlim(3998, 800)

    plt.axhline(0, color='black', linewidth = 0.5)

    plt.show()