import pandas as pd
from functions import format_df, convert_to_arrays
from functions import apply_sgfilter, optimise_pls_cv, conduct_pls

cal_file_location = "data/dil+infogest_mir_noPrRice_conc.csv"
sample_presentation = "Supernatant"
y_variable = "maltose_concentration"

window = 13
deriv = 1

df_cal = pd.read_csv(cal_file_location)
df_cal= format_df(df_cal)

df_cal = df_cal[df_cal['supernatant'] == sample_presentation]
y = df_cal[y_variable].values

model_stats_list  = []

#Get the unique values of starch and exp_type and assign them to a list
starch_list = df_cal['starch'].unique().tolist()
exp_type_list = df_cal['exp_type'].unique().tolist()
wavenumber_regions = ['wavenumbers_3998_800', 'wavenumbers_1500_800', 'wavenumbers_1250_909']

#create a list with all the permutations of the three lists
permutations_1 = [(starch, exp_type, wavenumber_region) for starch in starch_list for exp_type in exp_type_list for wavenumber_region in wavenumber_regions]

# for permutation in permutations:
#     starch = permutation[0]
#     exp_type = permutation[1]
#     wavenumber_region = permutation[2]
#     print(f"Starch: {starch}, exp_type: {exp_type}, wavenumber_region: {wavenumber_region}")

#     df_cal_subset = df_cal[(df_cal['starch'] == starch) & (df_cal['exp_type'] == exp_type)]
#     X, y = convert_to_arrays(df_cal_subset, sample_presentation, wavenumber_region)
#     wavenumber_string = wavenumber_region

#     X_sg = apply_sgfilter(X, wavenumber_region, window_length=window, poly_order=2, deriv=deriv)
#     optimum_components = optimise_pls_cv(X_sg, y, n_comp = 15, plot_components=False)
#     y_c, y_cv, score_c, score_cv, rmse_c, rmse_cv, x_load = conduct_pls(optimum_components, X_sg, y)
#     model_stats_list.append((wavenumber_string, sample_presentation, deriv, window, 2, optimum_components, score_c, rmse_c, score_cv, rmse_cv))

starch_exp_list = df_cal[['exp_type', 'starch']].to_numpy().tolist()

#make the list unique
starch_exp_list = [list(x) for x in set(tuple(x) for x in starch_exp_list)]


#a list with the permutation of the two lists starch_exp_list and wavenumber_regions
permutations = [(starch_exp, wavenumber_region) for starch_exp in starch_exp_list for wavenumber_region in wavenumber_regions]

for item in permutations:
    print(item[0][0])

