fig, ax = plt.subplots()

for starch in ["Rice", "Gelose 50", "Gelose 80", "Potato", "Pregelled Maize Starch"]:

    if starch != "All":
        df_cal_new = df_cal[df_cal["starch"] == starch]

    else:
        df_cal_new = df_cal

    if exp_type != "All":
        df_cal_new = df_cal_new[df_cal_new["exp_type"] == exp_type]

    #Selecting Wavenumbers and assign x and Y values
    wavenumbers = list(df_cal_new.columns[8:])
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