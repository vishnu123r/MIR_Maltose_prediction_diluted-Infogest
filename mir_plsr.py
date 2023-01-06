from functions import conduct_pls
from optimize_mir import get_wavenumber_range

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

drop_columns = ['Technical_rep']

#Read CSV and Drop columns
df_old = pd.read_csv("data/dil+infogest_mir_noPr_conc.csv")
df = df_old.drop(drop_columns,  axis= 1)
df.rename(columns={"Unnamed: 0": "sample_id"}, inplace = True)

#Change wavenumber to whole numbers
wavenumbers_old = list(df.columns[7:])
wavenumbers = list(map(float, wavenumbers_old))
wavenumbers = list(map(round, wavenumbers))
wavenumbers = list(map(str, wavenumbers))
df.rename(columns = dict(zip(wavenumbers_old, wavenumbers)), inplace = True)



#Segment into supernatant and turbid
df_turbid = df[df['supernatant'] == "Turbid"]
df_SN = df[df['supernatant'] == "Supernatant"]

#Selecting Wavenumbers and assign x and Y values
wavenumbers_3998_800 = get_wavenumber_range(wavenumbers, 3998, 800)




# #Turbid
# X = df_turbid[wavenumbers_3998_800].values
# y = df_turbid['maltose_concentration'].values

#Supernatant
X = df_SN[wavenumbers_3998_800].values
y = df_SN['maltose_concentration'].values

X = savgol_filter(X, 11, 2, 2)

#Apply PLSR
y_c, y_cv, score_c, score_cv, rmse_c, rmse_cv, x_load = conduct_pls(4, X, y)

#plot y range values 
rangey = max(y) - min(y)
rangex_c = max(y_c) - min(y_c)
rangex_cv = max(y_cv) - min(y_cv)

# Fit a line to the CV vs response
z_c = np.polyfit(y, y_c, 1)
z_cv = np.polyfit(y, y_cv, 1)

with plt.style.context(('ggplot')):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(y_c, y, c='red', edgecolors='k')
    #Plot the best fit line (calibration)
    ax.plot(np.polyval(z_c,y), y, c='blue', linewidth=1)
    #Plot the best fit line (cross validation)
    ax.plot(np.polyval(z_cv,y), y, c='yellow', linewidth=1)
    #Plot the ideal 1:1 line
    ax.plot(y, y, color='green', linewidth=1)
    plt.title('$R^{2}$ (CV): '+str(round(score_cv,2)))
    plt.xlabel('Predicted Maltose Concentration ()')
    plt.ylabel('Measured Maltose Concentration ()')

    plt.show()
    

print('R2 calib: %5.3f'  % score_c)
print('R2 CV: %5.3f'  % score_cv)
print('RMSE calib: %5.3f' % rmse_c)
print('RMSE CV: %5.3f' % rmse_cv)