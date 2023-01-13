import pandas as pd
from functions import format_df, convert_to_arrays


cal_file_location = "data/dil+infogest_mir_noPrRice_conc.csv"

df_cal = pd.read_csv(cal_file_location)
df_cal= format_df(df_cal)


#Get the unique values of starch and exp_type and assign them to a list
starch_list = df_cal['starch'].unique().tolist()
exp_type_list = df_cal['exp_type'].unique().tolist()
wavenumber_regions = ['wavenumbers_3998_800', 'wavenumbers_1500_800', 'wavenumbers_1250_909']

#create a list with all the permutations of the three lists
permutations = [(starch, exp_type, wavenumber_region) for starch in starch_list for exp_type in exp_type_list for wavenumber_region in wavenumber_regions]

for permutation in permutations:
    starch = permutation[0]
    exp_type = permutation[1]
    wavenumber_region = permutation[2]
    print(f"Starch: {starch}, exp_type: {exp_type}, wavenumber_region: {wavenumber_region}")

    df_cal_subset = df_cal[(df_cal['starch'] == starch) & (df_cal['exp_type'] == exp_type)]
    x, y = convert_to_arrays(df_cal_subset, wavenumber_region)
    print(x)
    print(y)
    
