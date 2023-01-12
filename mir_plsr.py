from functions import conduct_pls, plot_q_t_plot
from optimize_mir import get_wavenumber_range

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from sklearn.cross_decomposition import PLSRegression
from scipy.stats import f

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


df_cal = pd.read_csv("data/dil+infogest_mir_noPr_conc.csv")
df_val = pd.read_csv("data/dil+infogest_validation_mir.csv")

df_cal= format_df(df_cal)
df_val= format_df(df_val)
    
#Selecting Wavenumbers and assign x and Y values
wavenumbers = list(df_cal.columns[7:])
wavenumbers_3998_800 = get_wavenumber_range(wavenumbers, 3998, 800)

#supernatant
X_cal_sn,y_cal_sn = convert_to_arrays(df_cal, "Supernatant", wavenumbers_3998_800)
X_cal_sn = savgol_filter(X_cal_sn, 11, 2, 2)

#Apply PLSR
ncomp = 5
y_c, y_cv, score_c, score_cv, rmse_c, rmse_cv, x_load = conduct_pls(ncomp, X_cal_sn, y_cal_sn)




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