from functions import apply_sgfilter, optimise_pls_cv, conduct_pls, convert_to_arrays
import numpy as np
import pandas as pd

"""
This script goes through different wavenumber regions and savitsky-golay hypreparameters and finds the optimum values
"""

def get_wavenumber_range(wavenumbers_list, wavenumber_start = 3998, wavenumber_end = 800):
    """Gets the wavenumbers for analysis"""
    wavenumbers_int = list(map(int, wavenumbers_list))
    wavenumber_for_analysis = []
    for wavenumber in wavenumbers_int:
        if wavenumber <= wavenumber_start and wavenumber >= wavenumber_end:
            wavenumber_for_analysis.append(str(wavenumber))

    return wavenumber_for_analysis

def apply_pls(df, wavenumber_regions, sg_parameters, sample_presentation, y_variable = 'maltose_concentration', group = False):
    
    """ Applies PLS for the given range of wavenumbers and different hyper parameters for Savgol filter

    Returns:
        list: Returns a list of calibration values for given wavenumber regions and hyper parameters
    """
    if sample_presentation not in ["Turbid", "Supernatant"]:
        raise("The Argument Sample presentation should either be 'Turbid' or 'Supernatant'")
    
    df = df[df['supernatant'] == sample_presentation]
    y = df[y_variable].values
    model_stats_list  = []

    #Make a list of all the required permutations
    starch_exp_list = df[['exp_type', 'starch']].to_numpy().tolist()
    starch_exp_list = [list(x) for x in set(tuple(x) for x in starch_exp_list)]
    permutations = [(starch_exp, wavenumber_region) for starch_exp in starch_exp_list for wavenumber_region in wavenumber_regions]

    if group:
        for permutation in permutations:
            exp_type = permutation[0][0]
            starch = permutation[0][1]
            wavenumber_region = permutation[1]
            wavenumber_string = "{0}-{1} cm-1".format(wavenumber_region[0], wavenumber_region[-1])
            print(f"Starch: {starch}, exp_type: {exp_type}, wavenumber_region: {wavenumber_string}")

            df_subset = df[(df['starch'] == starch) & (df['exp_type'] == exp_type)]
            no_samples = df_subset.shape[0]
            X,y = convert_to_arrays(df_subset, sample_presentation, wavenumber_region, y_variable)

            for deriv, window in sg_parameters:
                X_sg = apply_sgfilter(X, wavenumber_region, window_length=window, poly_order=2, deriv=deriv)
                optimum_components = optimise_pls_cv(X_sg, y, n_comp = 15, plot_components=False)
                rpd_c, rpd_cv, score_c, score_cv, rmse_c, rmse_cv, x_load = conduct_pls(optimum_components, X_sg, y)
                variable_names =(wavenumber_string, starch, exp_type, no_samples, sample_presentation, deriv, window, 2, optimum_components, rpd_c, rpd_cv, score_c, rmse_c, score_cv, rmse_cv)
                model_stats_list.append(variable_names)

    else:
        for wavenumber_region in wavenumber_regions:
            no_samples = df.shape[0]
            X = df[wavenumber_region].values
            wavenumber_string = "{0}-{1} cm-1".format(wavenumber_region[0], wavenumber_region[-1])
            starch = "All"
            exp_type = "All"
            print("Currently doing wavenumber region - {}".format(wavenumber_string))
            for deriv, window in sg_parameters:
                X_sg = apply_sgfilter(X, wavenumber_region, window_length=window, poly_order=2, deriv=deriv)
                optimum_components = optimise_pls_cv(X_sg, y, n_comp = 15, plot_components=False)
                rpd_c, rpd_cv, score_c, score_cv, rmse_c, rmse_cv, x_load = conduct_pls(optimum_components, X_sg, y)
                variable_names =(wavenumber_string, starch, exp_type, no_samples, sample_presentation, deriv, window, 2, optimum_components, rpd_c, rpd_cv, score_c, rmse_c, score_cv, rmse_cv)
                model_stats_list.append(variable_names)

    return model_stats_list

if __name__ == '__main__':

    ##### INPUTS ##########
    y_variable = 'maltose_concentration'
    data_file = "dil+infogest_mir_all_conc"
    group = True

    ###########################
    drop_columns = ['Technical_rep']

    #Read CSV and Drop columns
    df_old = pd.read_csv("data/{}.csv".format(data_file))
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

    #Selecting Wavenumbers
    wavenumbers_3998_800 = get_wavenumber_range(wavenumbers, 3998, 800)
    wavenumbers_1500_800 = get_wavenumber_range(wavenumbers, 1500, 800)
    wavenumbers_1250_909 = get_wavenumber_range(wavenumbers, 1250, 909)
    wavenumbers_3000_2800 = get_wavenumber_range(wavenumbers, 3000, 2800)
    wavenumbers_1550_1250 = get_wavenumber_range(wavenumbers, 1550, 1250)
    wavenumbers_SN = wavenumbers_3000_2800 + wavenumbers_1550_1250

    #defining Hyper parameters
    wavenumber_regions = [wavenumbers_3998_800, wavenumbers_1500_800, wavenumbers_1250_909, wavenumbers_SN]
    sg_parameters = [(1,9),(1,7), (1,5), (1,3), (2, 9), (2, 7), (2,5), (1, 11), (1, 15),  (1, 21),  (1, 25), (1, 31), (2, 11), (2, 15), (2, 21), (2, 25), (2, 31), (2, 35), (2, 41)]

    #Get descriptive stats of Y
    y = df_turbid[y_variable].values
    descriptive_y = df_turbid[y_variable].describe().to_frame().T
    descriptive_y['Coe_variation'] = descriptive_y['std']/descriptive_y['mean']

    model_stats_turbid = apply_pls(df, wavenumber_regions, sg_parameters, sample_presentation = "Turbid", y_variable = y_variable, group=group)
    model_stats_supernatant = apply_pls(df, wavenumber_regions, sg_parameters, sample_presentation = "Supernatant", y_variable = y_variable, group=group)

    excel_columns = ['Wavenumber_region', 'Starch', 'Exp_type', 'no_samples', 'Sample_presentation', 'Derivative', 'Window_length', "Polynomial_order", "No_of_components", 'rpd_c', 'rpd_cv', 'Score_c', 'RMSEC', 'Score_CV', 'RMSECV']
    df_out_turbid = pd.DataFrame.from_records(model_stats_turbid, columns =excel_columns)
    df_out_sn = pd.DataFrame.from_records(model_stats_supernatant, columns =excel_columns)

    with pd.ExcelWriter('output/{0}/{1}'.format(y_variable, 'out_' + data_file + '_' + y_variable + '_' + str(group)+".xlsx" )) as writer:
    #with pd.ExcelWriter("output/test.xlsx" ) as writer:
        descriptive_y.to_excel(writer, sheet_name='descriptive_stats')
        df_out_turbid.to_excel(writer, sheet_name='calibration_stats_turbid')
        df_out_sn.to_excel(writer, sheet_name='calibration_stats_sn')