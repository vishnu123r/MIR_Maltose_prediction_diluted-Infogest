from sklearn.decomposition import PCA
from functions import conduct_pls, plot_q_t_plot, convert_to_arrays, format_df, apply_sgfilter 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from optimize_mir import get_wavenumber_range

from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2


############# ----INPUTS---- ##############
cal_file_location = "data/dil+infogest_mir_noPr(Infogest)_conc_NoNewSamples.csv"
val_file_location = "data/infogest_validation_mir.csv"

exp_type = "dil"
starch = "Rice"
y_variable = "time"

start_WN = 1500
end_WN = 800

sg_smoothing_point = 21
sg_derivative = 2
sg_polynomial = 2

no_of_components = 5
sample_presentation = "Supernatant"
sample_presentation = "Turbid"


tick_distance = 80
loadings_height = 0.1

txt_string = " cal_file_location: " + cal_file_location + "\n" + " val_file_location: " + val_file_location + "\n" + " exp_type: " + exp_type + "\n" + " starch: " + starch + "\n" + " y_variable: " + y_variable + "\n" + " start_WN: " + str(start_WN) + "\n" + " end_WN: " + str(end_WN) + "\n" + " sg_smoothing_point: " + str(sg_smoothing_point) + "\n" + " sg_derivative: " + str(sg_derivative) + "\n"  + " sg_polynomial: " + str(sg_polynomial) + "\n" + " no_of_components: " + str(no_of_components) + "\n" + " sample_presentation: "  + sample_presentation + "\n" + " tick_distance: " + str(tick_distance) + "\n" + " loadings_height: " + str(loadings_height) + "\n"

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

#X,y arrays - Calibration
X_cal,y_cal = convert_to_arrays(df_cal, wavenumbers, y_variable = y_variable)
X_cal = apply_sgfilter(X_cal, wavenumbers, sg_smoothing_point, sg_polynomial, sg_derivative)
scaler = StandardScaler()
X_cal = scaler.fit_transform(X_cal)

#mutiple all values in X_cal by 10^4 (This is done to circumvent the issue of getting a covariance matrix equal to zero)
#X_cal = X_cal * 10**4

#apply PCA
pca = PCA(n_components=no_of_components)
pca.fit(X_cal)
T = pca.transform(X_cal)

robust_cov = MinCovDet().fit(T)
m = robust_cov.mahalanobis(T)

print(T.shape[1])

#create a dataframe with sample_id and mahalanobis distance
df_mahalanobis = pd.DataFrame({'sample_id': df_cal['sample_id'], 'mahalanobis': m})
cutoff = chi2.ppf(0.9999, T.shape[1]-1)
print(cutoff)
print(df_mahalanobis[df_mahalanobis['mahalanobis'] > cutoff])

quit()

colours = [plt.cm.jet(float(i)/max(m)) for i in m]
with plt.style.context(('ggplot')):
    plt.scatter(T[:,0], T[:,1], color=colours)
    for i, txt in enumerate(df_cal['time']):
        plt.annotate(txt, (T[:,0][i], T[:,1][i]))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'PCA - {starch}')
    plt.show()


quit()
fig, ax = plt.subplots()

for starch in ["Rice", "Gelose 50", "Gelose 80", "Potato", "Pregelled Maize Starch"]:

    if starch != "All":
        df_cal_new = df_cal[df_cal["starch"] == starch]

    else:
        df_cal_new = df_cal

    if exp_type != "All":
        df_cal_new = df_cal_new[df_cal_new["exp_type"] == exp_type]

    #Selecting Wavenumbers and assign x and Y values
    wavenumbers = list(df_cal_new.columns[9:])
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