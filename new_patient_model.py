import numpy as np
import tellurium as te
import matplotlib.pyplot as plt

import pymc as pm
from pytensor.compile.ops import as_op
import pytensor.tensor as pt

from tellurium_model_fitting import generate_objective_function_multiple_times, generate_objective_function

from tellurium_model_fitting import abs_percent_error

new_model_desc_no_leno = """
//Aza PK model

species $X_aza_depot; species $X_aza_central; species $X_aza_eff;

// diffexps
X_aza_depot' = -KA_aza*X_aza_depot;
X_aza_central' = KA_aza*X_aza_depot - (CL_aza/V2_aza)*(X_aza_central);

// mostly for a unit conversion - to get it in units of ng/mL
X_aza_eff := X_aza_central/V2_aza*1000;

// initial values
Dose_aza = 0; // 75
X_aza_depot = Dose_aza;

KA_aza = 17;
CL_aza = 4589;
V2_aza = 110;

// Venetoclax PK model
species $X_ven_depot; species $X_ven_central; species $X_ven_peripheral;

// parameters - see table 2 in the paper.
CL := 449; // paper presents CL/F
KA := TVKA;
TVKA := 3.83;
Q := 130;

V2 := 99;
V3 := 147;

// TODO: bioavailability F = F1
// I'm not sure how to implement F1 in the ODE?
TVF1 := 1;
TVFA := 1;

// food increases bioavailability by 3-5 times;
// F1 := TVF1*TVFA*(Dose/400)^-0.178;

// TODO: alag

// diffexps
X_ven_depot' = -KA*X_ven_depot;
X_ven_central' = KA*X_ven_depot - (CL + Q)/V2*X_ven_central + Q/V3*X_ven_peripheral;
X_ven_peripheral' = Q/V2*X_ven_central - Q/V3*X_ven_peripheral;

// initial values
Dose_ven = 0; // 500
X_ven_depot = Dose_ven;
X_ven_central = 0;
X_ven_peripheral = 0;

// lenograstim (removing)

// WBCs (proliferating, transit, mature)
species $Xprol; species $Xtr; species $Xwbc; 

// PD model - WBCs
// effect
E_ven_wbc := slope_ven_wbc * ln(1 + X_ven_central); // standard kinetic model...
E_aza_wbc := slope_aza_wbc * ln(1 + X_aza_central);

// using the M3 model structure from Jost 2019
wbc_drug := 1 - E_ven_wbc - E_aza_wbc;
F := piecewise(wbc_drug*ktr, wbc_drug < 0, wbc_drug*ktr*(B/Xwbc)^gam);
G := ktr;

// TODO: eps

Xprol' = Xprol*(- G + F);
Xtr' =  G * (Xprol -  Xtr);
Xwbc' = G * Xtr - kwbc * Xwbc;

// leukemic blasts (bone marrow, blood)
species $Xl1; species $Xl2;
species $Xblasts; species $Xblasts_obs;

// PD model - leukemic blasts
E_ven_blast := slope_ven_blast * ln(1 + X_ven_central);
E_aza_blast := slope_aza_blast * ln(1 + X_aza_central);

klc := 1/(1 + c1*Xwbc + c2*Xl2);

Xl1' = (2*a1*klc - 1) * p1 * Xl1 - p1 * (E_ven_blast + E_aza_blast) * Xl1;
Xl2' = 2*(1 - a1*klc) * p1 * Xl1 - d2 * Xl2;

// cellularity
//CR = 0.5;
// DB is the "approximate maximal tumor burden"
//DB = 10^12;
// Xblasts is the observed blast concentration.
//Xblasts := 100*(Xl1 + 0.005*Xtr)/(CR*DB);
//Xblasts0 = 100*(Xl1 + 0.005*Xtr)/(CR*DB);
// okay... I don't know what to do exactly with Xl1.
Xblasts_obs := 100*Xl1;

// PD model of WBCs and leukemic blasts
// source is https://link.springer.com/article/10.1007/s11095-014-1429-9
kwbc = 2.3765; 
keg = 0.592*24;
kANC = 5.64*24;
beta = 0.234;
// source is https://www.nature.com/articles/s41598-018-21115-4
a1 = 0.875;
d2 = 2.3;
c1 = 0.01;
c2 = 0.01;

// parameters to fit - default values can be found in table 1 of the original paper.
ktr = 0.236; // transition rate - unit: 1/day
gam = 0.651; // feedback of G-CSF on WBC proliferation - MAKE SURE THIS VARIABLE IS NAMED gam
B = 4.67; // unit: G/L
B0 = 4.67;

// CANNOT name this variable "gamma" bc that's a reserved function
slope_ven_blast = 0.5;//7.94; // unit: L/micromol
slope_aza_blast = 1; //;
slope_ven_wbc = 0.15;//7.94; // unit: L/micromol
slope_aza_wbc = 0.5; //;
p1 = 0.1; // leukemic blast proliferation rate

// initial values
Xprol = B*kwbc/ktr;
Xtr = B*kwbc/ktr;
Xwbc = B;
Xl1 = 0.04;
"""

param_names = ['ktr', 'gam', 'slope_ven_blast', 'slope_aza_blast', 'slope_ven_wbc', 'slope_aza_wbc', 'B', 'Xl1']
param_names_bounds = [(0, 1), (0, 1), (0, 2), (0, 2), (0, 2), (0, 2), (0, 10), (0, 10)]
param_names_with_initial = ['ktr', 'gam', 'slope_ven_blast', 'slope_aza_blast', 'slope_ven_wbc', 'slope_aza_wbc', 'B', 'Xl1', 'B0']
param_names_with_initial_bounds = [(0, 1), (0, 1), (0, 2), (0, 2), (0, 2), (0, 2), (0, 10), (0, 10), (0, 10)]

def initialization_fn(model, param_vals):
    # initializing some of the model params to equilibrium values?
    ktr = param_vals[0]
    B = param_vals[6]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B*kwbc/ktr)
    model.setValue('Xtr', B*kwbc/ktr)
    model.setValue('Xwbc', B)
    return model

def initialization_fn_with_initial(model, param_vals):
    "Initial WBC is a free parameter"
    ktr = param_vals[0]
    B = param_vals[6]
    B0 = param_vals[8]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B*kwbc/ktr)
    model.setValue('Xtr', B*kwbc/ktr)
    model.setValue('Xwbc', B0)
    return model


def initialization_fn_with_initial_2(model, param_vals):
    "Initial WBC is a free parameter"
    ktr = param_vals[0]
    #B = param_vals[6]
    B0 = param_vals[8]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B0*kwbc/ktr)
    model.setValue('Xtr', B0*kwbc/ktr)
    model.setValue('Xwbc', B0)
    return model




def generate_dosing_component(venex_cycles, aza_cycles):
    """
    Params
    ------
        venex_cycles - pandas dataframe with columns start, end, dose
        aza_cycles - pandas dataframe with columns start, end, dose
    """
    output_str = ''
    for _, row in venex_cycles.iterrows():
        start = row.start
        end = row.end
        dose = row.dose
        for i in range(start, end+1):
            output_str += f'at time >= {i+0.5} : X_ven_depot = {dose};\n'
    output_str += '\n'
    for _, row in aza_cycles.iterrows():
        start = row.start
        end = row.end
        dose = row.dose
        for i in range(start, end+1):
            output_str += f'at time >= {i+0.5} : X_aza_depot = {dose};\n'
    return output_str


def set_model_params(model_desc, cycle_info, param_vals=None, dosing_component_function=None,
        venetoclax_start="venetoclax_start",
        venetoclax_stop="venetoclax_stop",
        venetoclax_dose="venetoclax_dose_mg",
        aza_start="aza_start",
        aza_stop="aza_stop",
        aza_dose="aza_dose_mg",
        **kwargs):
    """
    Params
    ------
        model_desc: text code for model description
        cycle_info: pandas dataframe containing data on cycles - 
            should have fields "venetoclax_start", venetoclax_stop, venetoclax_dose_mg, aza_start, aza_stop, aza_dose_mg
            
    Returns
    -------
    A tellurium model built from the desc.
    """
    venex_cycles = cycle_info[[venetoclax_start, venetoclax_stop, venetoclax_dose]]
    venex_cycles.loc[:, venetoclax_dose] = venex_cycles[venetoclax_dose].fillna(400)
    venex_cycles = venex_cycles.astype(int)
    venex_cycles.columns = ['start', 'end', 'dose']
    aza_cycles = cycle_info[[aza_start, aza_stop, aza_dose]]
    aza_cycles.loc[:, aza_dose] = aza_cycles[aza_dose].fillna(120)
    aza_cycles = aza_cycles.astype(int)
    aza_cycles.columns = ['start', 'end', 'dose']
    if dosing_component_function is None:
        dosing_component_function = generate_dosing_component
    dosing_component = dosing_component_function(venex_cycles, aza_cycles)
    model_desc_final = model_desc + dosing_component
    model = te.loada(model_desc_final)
    if param_vals is not None:
        for k, v in param_vals.items():
            model.setValue(k, v)
    return model


def generate_residual_function(model, wbc, blasts, n_samples=None, params_to_fit=None, initialization=None):
    """
    Generates a residual function using leukocyte and blast data (assuming that the cycle info is already incorporated into the model)
    """
    if initialization is None:
        initialization = initialization_fn
    if params_to_fit is None:
        params_to_fit = param_names
    # n_samples should be max time * 10?
    if n_samples is None:
        n_samples = wbc.iloc[:,0].max()*20
        print('n_samples:', n_samples)
    resid_function = generate_objective_function_multiple_times(model,
                                    {'Xwbc': wbc.iloc[:,1].to_numpy(), 
                                     'Xblasts_obs': blasts.iloc[:,1].to_numpy()}, # observed values
                                    {'Xwbc': wbc.iloc[:,0].to_numpy(),
                                     'Xblasts_obs': blasts.iloc[:,0].to_numpy()}, # observation times
                                    params_to_fit=params_to_fit,
                                    use_val_ranges=False,
                                    set_initial_vals=False,
                                    n_samples=n_samples,
                                    var_weights=None,
                                    #global_param_vals=model_params,
                                    time_range=None,
                                    initialization_fn=initialization,
                                    return_values=True,
                                    handle_errors=True,
                                    print_errors=True)
    return resid_function


def generate_error_function(model, wbc, blasts, n_samples=None, params_to_fit=None, initialization=None,
        wbc_only=False, blasts_only=False, **params):
    """
    Generates an error function for a given model, using tellurium_model_fitting.generate_objective_function.
    The error is the sum of the RMSEs of the blast and WBC data.

    By default, this uses mean absolute percentage error as the metric instead of RMSE.
    Although we probably don't want percent error because the values are often close to 0...
    """
    if initialization is None:
        initialization = initialization_fn
    if params_to_fit is None:
        params_to_fit = param_names
    # n_samples should be max time * 10?
    if n_samples is None:
        n_samples = wbc.iloc[:,0].max()*20
        print('n_samples:', n_samples)
    observed_data = {'Xwbc': wbc.iloc[:,1].to_numpy(), 
                     'Xblasts_obs': blasts.iloc[:,1].to_numpy()}
    if wbc_only:
        observed_data = {'Xwbc': wbc.iloc[:,1].to_numpy()} 
    time_data = {'Xwbc': wbc.iloc[:,0].to_numpy(),
                 'Xblasts_obs': blasts.iloc[:,0].to_numpy()}
    if wbc_only:
        time_data = {'Xwbc': wbc.iloc[:,0].to_numpy()}
    resid_function = generate_objective_function(model,
                                    observed_data, # observed values
                                    time_data, # observation times
                                    params_to_fit=params_to_fit,
                                    use_val_ranges=False,
                                    set_initial_vals=False,
                                    n_samples=n_samples,
                                    var_weights=None,
                                    #global_param_vals=model_params,
                                    time_range=None,
                                    initialization_fn=initialization,
                                    error_fn=abs_percent_error,
                                    error_combo_fn=np.mean,
                                    return_values=True,
                                    handle_errors=True,
                                    print_errors=True,
                                    **params)
    return resid_function


def generate_forward_function(model, wbc, blasts, n_samples=None, params_to_fit=None, initialization=None):
    ode_out = generate_residual_function(model,  wbc, blasts, n_samples=n_samples, params_to_fit=params_to_fit,
            initialization=initialization)
    @as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
    def pytensor_forward_model(theta):
        output = ode_out(theta)
        wbc = output['Xwbc']
        blasts = output['Xblasts_obs']
        # TODO: if wb.
        if len(wbc) > len(blasts):
            blasts = np.pad(blasts, (0, len(wbc) - len(blasts)))
        elif len(blasts) > len(wbc):
            wbc = np.pad(wbc, (0, len(blasts) - len(wbc)))
        # zero-pad the blasts array?
        return np.vstack([wbc, blasts])
    return pytensor_forward_model


def build_pm_model(model, wbc, blasts, theta=None, n_samples=None, params_to_fit=None,
        initialization=None, use_b0=False, uniform_prior=False, **params):
    """
    Builds a PyMC model
    """
    if params_to_fit is None:
        params_to_fit = param_names
    if theta is None:
        model.resetToOrigin()
        default_params = [model.getValue(x) for x in params_to_fit]
        theta = default_params
    #print('theta:', theta)
    pytensor_forward_function = generate_forward_function(model, wbc, blasts, n_samples=n_samples,
            params_to_fit=params_to_fit, initialization=initialization)
    wbc_data = wbc[1].to_numpy(dtype=np.float64)
    blast_data = blasts[1].to_numpy(dtype=np.float64)
    #print('wbc_data:', wbc_data)
    #print('blast_data:', blast_data)
    with pm.Model() as new_patient_model_pm:
        # priors
        # TODO: uniform priors for slope?
        if not uniform_prior:
            ktr = pm.TruncatedNormal("ktr", mu=theta[0], sigma=theta[0]/2, lower=0, upper=1, initval=theta[0])
            gam = pm.TruncatedNormal("gam", mu=theta[1], sigma=theta[1]/2, lower=0, upper=1, initval=theta[1])
            slope_ven_blast = pm.TruncatedNormal("slope_ven_blast", mu=theta[2], sigma=theta[2]/2, lower=0, initval=theta[2])
            slope_aza_blast = pm.TruncatedNormal("slope_aza_blast", mu=theta[3], sigma=theta[3]/2, lower=0, initval=theta[3])
            slope_ven_wbc = pm.TruncatedNormal("slope_ven_wbc", mu=theta[4], sigma=theta[4]/2, lower=0, initval=theta[4])
            slope_aza_wbc = pm.TruncatedNormal("slope_aza_wbc", mu=theta[5], sigma=theta[5]/2, lower=0, initval=theta[5])
            B = pm.TruncatedNormal("B", mu=theta[6], sigma=theta[6]/2, lower=0, initval=theta[6])
            if use_b0:
                B0 = pm.TruncatedNormal('B0', mu=theta[8], sigma=theta[8]/2, lower=0, initval=theta[7])
        else:
            ktr = pm.Uniform("ktr", lower=0, upper=1, initval=theta[0])
            gam = pm.Uniform("gam", lower=0, upper=1, initval=theta[1])
            slope_ven_blast = pm.Uniform("slope_ven_blast", lower=0, upper=2, initval=theta[2])
            slope_aza_blast = pm.Uniform("slope_aza_blast", lower=0, upper=2, initval=theta[3])
            slope_ven_wbc = pm.Uniform("slope_ven_wbc", lower=0, upper=2, initval=theta[4])
            slope_aza_wbc = pm.Uniform("slope_aza_wbc", lower=0, upper=2, initval=theta[5])
            B = pm.Uniform("B", lower=0, upper=10, initval=theta[6])
            if use_b0:
                B0 = pm.Uniform('B0', lower=0, upper=10, initval=theta[7])
        Xl1 = pm.Uniform('Xl1', lower=0, upper=1, initval=theta[7])
        sigma_wbc = pm.HalfNormal("sigma_wbc", 5)
        sigma_blasts = pm.HalfNormal("sigma_blasts", 5)

        # ODE solution function
        ode_solution = pytensor_forward_function(pm.math.stack([ktr, gam,
                                                               slope_ven_blast,
                                                               slope_aza_blast,
                                                               slope_ven_wbc,
                                                               slope_aza_wbc, B, Xl1]))
        if use_b0:
            ode_solution = pytensor_forward_function(pm.math.stack([ktr, gam,
                                                               slope_ven_blast,
                                                               slope_aza_blast,
                                                               slope_ven_wbc,
                                                               slope_aza_wbc, B, Xl1, B0]))
        
        # split up the blasts and WBCs
        wbc_ode = ode_solution[0,:len(wbc_data)]
        blast_ode = ode_solution[1,:len(blast_data)]

        # likelihood
        pm.Normal("Y_wbc_obs", mu=wbc_ode, sigma=sigma_wbc, observed=wbc_data)
        pm.Normal("Y_blast_obs", mu=blast_ode, sigma=sigma_blasts, observed=blast_data)
    return new_patient_model_pm


def generate_blast_interpolation(blast_table):
    import scipy
    bm_blast_interpolator = scipy.interpolate.PchipInterpolator(blast_table['date_bonemarrow'], blast_table['bm_blasts'])
    c = bm_blast_interpolator.c
    poly = ''
    piecewise = 'Xblasts_obs := piecewise('
    for i in range(c.shape[1]):
        start_time = blast_table.date_bonemarrow.iloc[i]
        poly_i = f'c{i+1} := '
        for j in range(c.shape[0]):
            if j < 3:
                if c[j][i] != 0:
                    if start_time >= 0 :
                        poly_i += f'{c[j][i]}*(time-{start_time})^{3-j} + '
                    else:
                        poly_i += f'{c[j][i]}*(time-({start_time}))^{3-j} + '
            else:
                poly_i += f'{c[j][i]}'
        poly_i += ';\n'
        poly += poly_i
        if i < len(blast_table) - 2:
            end_time = blast_table.date_bonemarrow.iloc[i+1]
            piecewise += f'c{i+1}, time >= {start_time} and time <= {end_time}, '
        else:
            piecewise += f'c{i+1}, time >= {start_time});'
    output = poly + piecewise
    #print(output)
    return output

def build_pm_model_from_dataframes(cycle_info, leuk_table, blast_table, model_desc=new_model_desc_no_leno,
                                 param_vals=None, n_samples=None, params_to_fit=None,
                                 build_model_function=None, use_neut=False, use_blasts=True,
                                 dosing_component_function=None, initialization=None,
                                 lab_date='lab_date', 
                                 date_bonemarrow='date_bonemarrow',
                                 event_date='Event.Date',
                                 b_leuk='b_leuk',
                                 b_neut='b_neut',
                                 bm_blasts='bm_blasts',
                                 fill_dates=False,
                                 use_b0=False,
                                 use_blasts_interpolation=False,
                                 use_initial_guess=False,
                                 blasts_only=False,
                                 theta=None,
                                 **kwargs):
    """
    Constructs a model from dataframe...

    Returns: 1) tellurium model; 2) PyMC model constructed using the tellurium model.

    Params
    ------
        cycle_info - pd.DataFrame
        leuk_table - pd.DataFrame
        blast_table - pd.DataFrame

    Optional params
    ---------------
    build_model_function - function that returns a PyMC model.
    use_neut - whether to use neutrophils (if false, uses total WBCs/leukocytes)
    use_blasts
    dosing_component_function
    initialization
    use_b0 - whether or not to use the initial WBC/neutrophil count as a parameter.
    use_blasts_interpolation - whether or not to build an interpolated function for the blasts.
    use_initial_guess
    """
    # parameterize the column names
    # 1. build the actual model
    if dosing_component_function is None:
        dosing_component_function = generate_dosing_component
    new_model_desc = model_desc
    if use_blasts_interpolation:
        blast_table = blast_table.copy()
        if fill_dates:
            blast_table.loc[date_bonemarrow] = blast_table[date_bonemarrow].fillna(blast_table[event_date])
        if 'bm_blasts_range' in blast_table.columns:
            blast_table.loc[blast_table.bm_blasts_range == 0, bm_blasts] = 2.5
        blast_data = blast_table[[date_bonemarrow, bm_blasts]].copy()
        blast_data.loc[blast_data[bm_blasts] == '<5', bm_blasts] = 2.5
        blast_data.columns = [0,1]
        blast_data = blast_data.reset_index().iloc[:,1:]
        blast_data.iloc[:,1] = blast_data.iloc[:,1].map(float)
        interpolation_component = generate_blast_interpolation(blast_table)
        new_model_desc += '\n'
        new_model_desc += interpolation_component
    model = set_model_params(new_model_desc, cycle_info, param_vals, dosing_component_function=dosing_component_function, **kwargs)

    # 2. convert leuk/blast tables to the format used in previous models (and deal with NAs)
    leuk_table = leuk_table.copy()
    if fill_dates:
        leuk_table.loc[lab_date] = leuk_table[lab_date].fillna(leuk_table[event_date])
    leuk_data = leuk_table[[lab_date, b_leuk]]
    if use_neut:
        leuk_data = leuk_table[[lab_date, b_neut]]
    leuk_data.columns = [0,1]
    leuk_data = leuk_data.reset_index().iloc[:,1:]
    # TODO: get default params
    if params_to_fit is not None and use_initial_guess and use_b0:
        model.resetToOrigin()
        default_params = [model.getValue(x) for x in params_to_fit]
        if theta is None:
            theta = default_params
        b_index = params_to_fit.index('B')
        b0_index = params_to_fit.index('B0')
        theta[b_index] = leuk_data.iloc[:, 1].mean()
        theta[b0_index] = leuk_data.iloc[0, 1] + 1e-4
    if build_model_function is None:
        build_model_function = build_pm_model
    if use_blasts:
        blast_table = blast_table.copy()
        if fill_dates:
            blast_table.loc[date_bonemarrow] = blast_table[date_bonemarrow].fillna(blast_table[event_date])
        if 'bm_blasts_range' in blast_table.columns:
            blast_table.loc[blast_table.bm_blasts_range == 0, bm_blasts] = 2.5
        blast_data = blast_table[[date_bonemarrow, bm_blasts]].copy()
        blast_data.loc[blast_data[bm_blasts] == '<5', bm_blasts] = 2.5
        blast_data.columns = [0,1]
        blast_data = blast_data.reset_index().iloc[:,1:]
        blast_data.iloc[:,1] = blast_data.iloc[:,1].map(float)
        pm_model = build_model_function(model, leuk_data, blast_data, n_samples=n_samples, params_to_fit=params_to_fit,
                initialization=initialization, use_b0=use_b0, theta=theta, **kwargs)
    else:
        pm_model = build_model_function(model, leuk_data, n_samples=n_samples, params_to_fit=params_to_fit,
                initialization=initialization, use_b0=use_b0, theta=theta, **kwargs)
    return model, pm_model


def sample_model(pm_model, draws=5000):
    """
    Samples from a model using DEMetropolisZ, and returns the trace.
    """
    vars_list = [x for x in list(pm_model.values_to_rvs.keys()) if not isinstance(x, pt.TensorConstant)]
    tune = draws
    with pm_model:
        trace_DEMZ = pm.sample(step=[pm.DEMetropolisZ(vars_list)], tune=tune, draws=draws)
    trace = trace_DEMZ
    return trace


def fit_model_mc(pm_model, draws=5000, params_to_fit=None, **sample_kwargs):
    "Runs MCMC to get a model posterior. Returns the posterior means and the table of traces."
    if params_to_fit is None:
        params_to_fit = param_names
    vars_list = [x for x in list(pm_model.values_to_rvs.keys()) if not isinstance(x, pt.TensorConstant)]
    tune = draws
    with pm_model:
        trace_DEMZ = pm.sample(step=[pm.DEMetropolisZ(vars_list)], tune=tune, draws=draws, **sample_kwargs)
    trace = trace_DEMZ
    pymc_posterior_means = [float(trace.posterior[x].mean()) for x in params_to_fit]
    return pymc_posterior_means, trace


def fit_model_map(pm_model, params_to_fit=None, **map_kwargs):
    "Returns the MAP optimal results for the model."
    if params_to_fit is None:
        params_to_fit = param_names
    map_result = pm.find_MAP(model=pm_model, **map_kwargs)
    map_params = [float(map_result[k]) for k in params_to_fit]
    return map_params


def split_train_test(max_days, leuk_table, blast_table,
        lab_date='lab_date', date_bonemarrow='date_bonemarrow'):
    """
    Splits the leuk and blast tables into training and test tables, with a cutoff at a certain number of days...
    """
    leuk_table_train = leuk_table[leuk_table[lab_date] <= max_days]
    leuk_table_test = leuk_table[leuk_table[lab_date] > max_days]
    blast_table_train = blast_table[blast_table[date_bonemarrow] <= max_days]
    blast_table_test = blast_table[blast_table[date_bonemarrow] > max_days]
    return leuk_table_train, leuk_table_test, blast_table_train, blast_table_test


def split_cycles(leuk_table, blast_table, cycle_info, n_cycles_train=3,
        n_cycles_test=1, n_cycles_skip=0,
        lab_date='lab_date', date_bonemarrow='date_bonemarrow'):
    """
    Splits data into cycles...

    Params:
        n_cycles_train: number of cycles to use for training data
        n_cycles_test: number of cycles to use for testing
        n_cycles_skip: number of cycles to skip
    """
    # TODO: I believe there is a bug in this...
    skip_end_day = -10
    train_end_day = 0
    test_end_day = 0
    n_train_end = n_cycles_skip + n_cycles_train
    n_test_end = n_cycles_skip + n_cycles_train + n_cycles_test
    cycle_info = cycle_info.reset_index()
    for i, row in cycle_info.iterrows():
        if i < n_cycles_skip:
            continue
        elif i >= n_cycles_skip and skip_end_day == 0 and n_cycles_skip == 0:
            skip_end_day = row.days_from_event
        elif i >= n_train_end and i < n_test_end and train_end_day == 0:
            train_end_day = row.days_from_event
        elif i >= n_test_end:
            test_end_day = row.days_from_event
            break
    leuk_table_train = leuk_table.query(f'lab_date <= {train_end_day} and lab_date >= {skip_end_day}')
    leuk_table_test = leuk_table.query(f'lab_date > {train_end_day} and lab_date <= {test_end_day}')

    blast_table_train = blast_table.query(f'date_bonemarrow <= {train_end_day} and date_bonemarrow >= {skip_end_day}')
    blast_table_test = blast_table.query(f'date_bonemarrow > {train_end_day} and date_bonemarrow <= {test_end_day}')
    # remainders
    leuk_table_remainder = leuk_table.query(f'lab_date > {test_end_day}')
    blast_table_remainder = blast_table.query(f'date_bonemarrow > {test_end_day}')
    return leuk_table_train, leuk_table_test, blast_table_train, blast_table_test, leuk_table_remainder, blast_table_remainder



def extract_data_from_wbc_table(wbc_data, subject_id, use_neut=False):
    """
    Given the wbc_data table (source: patient_data_venex/ven_bloodcounts_jan2024.txt),
    this returns: cycle_info, leuk_table, blast_table

    Params
    ------
        wbc_data: pd.DataFrame
        subject_id: int - a subject id from wbc_data

    Returns
    -------
        cycle_info
        leuk_table
        blast_table
    """
    wbc_data_patient1 = wbc_data[wbc_data['Subject.ID'] == subject_id]
    cycle_info = wbc_data_patient1[wbc_data_patient1.Form=='cycle_info'][['venetoclax_start', 'venetoclax_stop', 'venetoclax_dose_mg', 'aza_start', 'aza_stop', 'aza_dose_mg']]
    cycle_info = cycle_info[~cycle_info.isna().any(axis=1)]
    if use_neut:
        leuk_table = wbc_data_patient1[~wbc_data_patient1.b_neut.isna()].copy()
    else:
        leuk_table = wbc_data_patient1[~wbc_data_patient1.b_leuk.isna()].copy()
    leuk_table.lab_date = leuk_table.lab_date.fillna(leuk_table['Event.Date'])
    blast_table = wbc_data_patient1[~wbc_data_patient1.bm_blasts.isna()].copy()
    # map blast counts that are e.g. '<5' to 4.
    blast_table.loc[blast_table.bm_blasts == '<5', 'bm_blasts'] = 4
    blast_table.date_bonemarrow = blast_table.date_bonemarrow.fillna(blast_table['Event.Date'])
    return cycle_info, leuk_table, blast_table


def extract_data_from_neut_table(wbc_data, subject_id):
    """
    Given the wbc_data table (source: patient_data_venex/ven_bloodcounts_jan2024.txt),
    this returns: cycle_info, neut_table, blast_table

    Params
    ------
        wbc_data: pd.DataFrame
        subject_id: int - a subject id from wbc_data

    Returns
    -------
        cycle_info
        neut_table - table of neutrophil measurements
        blast_table
    """
    wbc_data_patient1 = wbc_data[wbc_data['Subject.ID'] == subject_id]
    cycle_info = wbc_data_patient1[wbc_data_patient1.Form=='cycle_info'][['venetoclax_start', 'venetoclax_stop', 'venetoclax_dose_mg', 'aza_start', 'aza_stop', 'aza_dose_mg']]
    cycle_info = cycle_info[~cycle_info.isna().any(axis=1)]
    neut_table = wbc_data_patient1[~wbc_data_patient1.b_neut.isna()].copy()
    neut_table.lab_date = neut_table.lab_date.fillna(neut_table['Event.Date'])
    blast_table = wbc_data_patient1[~wbc_data_patient1.bm_blasts.isna()].copy()
    # map blast counts that are e.g. '<5' to 4.
    blast_table.loc[blast_table.bm_blasts == '<5', 'bm_blasts'] = 4
    blast_table.date_bonemarrow = blast_table.date_bonemarrow.fillna(blast_table['Event.Date'])
    return cycle_info, neut_table, blast_table


def extract_data_from_tables_new(blood_counts, bm_blasts, cycle_days, subject_id,
        use_neut=True, max_neut=20):
    """
    This uses new-form data, and returns the usual three tables.

    Returns
    -------
        cycle_info
        leuk_table or neut_table - table of leukocyte or neutrophil measurements
        blast_table
    """
    leuk_table = blood_counts[blood_counts['Pseudonym'] == subject_id]
    cycle_info = cycle_days[cycle_days['Pseudonym']==subject_id]
    blast_table = bm_blasts[bm_blasts['Pseudonym']==subject_id]
    # fill in NaNs in cycle dates
    cycle_info.loc[:, 'days_aza_stop'] = cycle_info['days_aza_stop'].fillna(cycle_info['days_aza_start']+cycle_info['aza_days'])
    cycle_info.loc[:, 'days_ven_stop'] = cycle_info['days_ven_stop'].fillna(cycle_info['days_ven_start']+cycle_info['venetoclax_days'])
    # clean up cycles whose day count is obviously off...
    ven_days_subset = (cycle_info.days_ven_stop - cycle_info.days_ven_start).abs() - cycle_info.venetoclax_days > 5
    aza_days_subset = (cycle_info.days_aza_stop - cycle_info.days_aza_start).abs() - cycle_info.aza_days > 5
    cycle_info.loc[aza_days_subset, 'days_aza_stop'] = cycle_info['days_aza_start'] + cycle_info['aza_days']
    cycle_info.loc[ven_days_subset, 'days_ven_stop'] = cycle_info['days_ven_start'] + cycle_info['venetoclax_days']
    # drop all nas
    cycle_info = cycle_info.dropna()
    # fill neutrophils with percentages
    has_neut_percentage = (leuk_table.b_neut.isna() & (~leuk_table.b_neut_percentage.isna()))
    data_has_neut_percentage = leuk_table[has_neut_percentage]
    leuk_table.loc[has_neut_percentage, 'b_neut'] = data_has_neut_percentage.b_neut_percentage*data_has_neut_percentage.b_leuk/100

    # filter neut by is_na, realistic values
    leuk_table = leuk_table[(~leuk_table.b_neut.isna()) & (~leuk_table.days_lab.isna())].copy()
    leuk_table = leuk_table[leuk_table.b_neut <= max_neut]
    leuk_table = leuk_table.sort_values('days_lab')
    leuk_table = leuk_table.loc[~leuk_table.days_lab.duplicated(keep='first'), :]
    # filter blasts
    blast_table.loc[blast_table.bm_blasts_range == 0, 'bm_blasts'] = 2.5
    blast_table.loc[blast_table.bm_blasts == '<5', 'bm_blasts'] = 2.5
    blast_table = blast_table[(~blast_table.bm_blasts.isna()) & (~blast_table.days_from_bm.isna())].copy()
    # remove duplicate days
    blast_table = blast_table.sort_values('days_from_bm')
    blast_table = blast_table.loc[~blast_table.days_from_bm.duplicated(keep='first'), :]
    # convert day columns
    leuk_table['lab_date'] = leuk_table['days_lab']
    blast_table['date_bonemarrow'] = blast_table['days_from_bm']
    cycle_info['venetoclax_start'] = cycle_info['days_ven_start']
    cycle_info['venetoclax_stop'] = cycle_info['days_ven_stop']
    cycle_info['venetoclax_dose'] = cycle_info['venetoclax_dose_mg']
    cycle_info['aza_start'] = cycle_info['days_aza_start']
    cycle_info['aza_stop'] = cycle_info['days_aza_stop']
    cycle_info['aza_dose'] = cycle_info['aza_dose_mg']
    return cycle_info, leuk_table, blast_table


def run_model(model, params, max_time=200, n_samples=2000, selections=None, params_to_fit=None,
        initialization=None):
    """
    Runs the model with the given params.
    """
    if params_to_fit is None:
        params_to_fit = param_names
    if selections == None:
        selections = ['time', 'Xwbc', 'Xblasts_obs']
    if initialization is None:
        initialization = initialization_fn
    model.resetToOrigin()
    model = initialization(model, params)
    for k, v in zip(params_to_fit, params):
        model.setValue(k, v)
    results = model.simulate(0, int(max_time), int(n_samples), selections=selections)
    return results



def plot_data(results, cycle_info, leuk_table, blast_table, subject_id=1, save_fig=True,
        leuk_table_test=None, blast_table_test=None, use_neut=False):
    """
    Plots the data from a single run of the model.
    
    """
    b_leuk = 'b_leuk'
    label = 'leukocytes'
    if use_neut:
        label = 'neutrophils'
        b_leuk = 'b_neut'
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(f'Patient {subject_id}')
    if leuk_table_test is None:
        axes[0].scatter(leuk_table['lab_date'], leuk_table[b_leuk], label='observed ' + label)
    else:
        axes[0].scatter(leuk_table['lab_date'], leuk_table[b_leuk], label=label+' (train)')
        axes[0].scatter(leuk_table_test['lab_date'], leuk_table_test[b_leuk], label=label+' (test)')
    if blast_table_test is None:
        axes[1].scatter(blast_table['date_bonemarrow'], blast_table.bm_blasts.astype(float), label='observed blasts')
    else:
        axes[1].scatter(blast_table['date_bonemarrow'], blast_table.bm_blasts.astype(float), label='blasts (train)')
        axes[1].scatter(blast_table_test['date_bonemarrow'], blast_table_test.bm_blasts.astype(float), label='blasts (test)')
    axes[1].plot(results['time'], results['Xblasts_obs'], label='model blasts')
    #axes[0].scatter(wbc_data_patient1_neut['lab_date'], wbc_data_patient1_neut.b_neut, label='neutrophils')
    if use_neut:
        axes[0].plot(results['time'], results['Xwbc'], label='model neutrophils')
    else:
        axes[0].plot(results['time'], results['Xwbc'], label='model WBCs')
    axes[0].set_xlabel('Day')
    if use_neut:
        axes[0].set_ylabel('Neutrophil counts (10^9 per liter)')
    else:
        axes[0].set_ylabel('WBC counts (10^9 per liter)')
    axes[1].set_xlabel('Day')
    axes[1].set_ylabel('Blast %')
    axes[1].set_ylim(0, 100)
    #axes[1].set_yscale('log')
    for i, (_, row) in enumerate(cycle_info.iterrows()):
        label = None
        # the reason for the i=0 is so that we only have one thing in the legend, rather than a separate legend for every cycle.
        if i == 0:
            label = 'venetoclax admin'
        axes[0].fill_between([row.venetoclax_start, row.venetoclax_stop], 0, 1, alpha=0.2, color='orange',
                         label=label, transform=axes[0].get_xaxis_transform())
        axes[1].fill_between([row.venetoclax_start, row.venetoclax_stop], 0, 1, alpha=0.2, color='orange',
                         label=label, transform=axes[1].get_xaxis_transform())
        if i == 0:
            label = 'azacitidine admin'
        axes[0].fill_between([row.aza_start, row.aza_stop], 0, 1, alpha=0.2, color='green',
                             label=label, transform=axes[0].get_xaxis_transform())
        axes[1].fill_between([row.aza_start, row.aza_stop], 0, 1, alpha=0.2, color='green',
                             label=label, transform=axes[1].get_xaxis_transform())
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'new_patient_model_subject{subject_id}.png', dpi=300)
    plt.show()


def plot_data_wbc_only(results, cycle_info, leuk_table, subject_id=1, save_fig=True,
        leuk_table_test=None, label='wbc', use_neut=False):
    if use_neut:
        leuk_name='b_neut'
        label = 'Neut'
    else:
        leuk_name='b_leuk'
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    fig.suptitle(f'Patient {subject_id}')
    if leuk_table_test is None:
        ax.scatter(leuk_table['lab_date'], leuk_table[leuk_name], label=label)
    else:
        ax.scatter(leuk_table['lab_date'], leuk_table[leuk_name], label=label + ' (train)')
        ax.scatter(leuk_table_test['lab_date'], leuk_table_test[leuk_name], label=label+' (test)')
    ax.plot(results['time'], results['Xwbc'])
    for i, (_, row) in enumerate(cycle_info.iterrows()):
        label = None
        # the reason for the i=0 is so that we only have one thing in the legend, rather than a separate legend for every cycle.
        if i == 0:
            label = 'venetoclax admin'
        ax.fill_between([row.venetoclax_start, row.venetoclax_stop], 0, 1, alpha=0.2, color='orange',
                         label=label, transform=ax.get_xaxis_transform())
        if i == 0:
            label = 'azacitidine admin'
        ax.fill_between([row.aza_start, row.aza_stop], 0, 1, alpha=0.2, color='green',
                             label=label, transform=ax.get_xaxis_transform())
    ax.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'new_patient_model_subject{subject_id}.png', dpi=300)
    plt.show()


def plot_data_blasts_only(results, cycle_info, blast_table, subject_id=1, save_fig=False,
        blast_table_test=None, label='blasts', blast_name='bm_blasts'):
    """
    Plots the blasts...
    """
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    fig.suptitle(f'Patient {subject_id}')
    if blast_table_test is None:
        ax.scatter(blast_table['date_bonemarrow'], blast_table[blast_name], label=label)
    else:
        ax.scatter(blast_table['date_bonemarrow'], blast_table[blast_name], label=label + ' (train)')
        ax.scatter(blast_table_test['date_bonemarrow'], blast_table_test[blast_name], label=label+' (test)')
    ax.plot(results['time'], results['Xblasts_obs'])
    for i, (_, row) in enumerate(cycle_info.iterrows()):
        label = None
        # the reason for the i=0 is so that we only have one thing in the legend, rather than a separate legend for every cycle.
        if i == 0:
            label = 'venetoclax admin'
        ax.fill_between([row.venetoclax_start, row.venetoclax_stop], 0, 1, alpha=0.2, color='orange',
                         label=label, transform=ax.get_xaxis_transform())
        if i == 0:
            label = 'azacitidine admin'
        ax.fill_between([row.aza_start, row.aza_stop], 0, 1, alpha=0.2, color='green',
                             label=label, transform=ax.get_xaxis_transform())
    ax.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'new_patient_model_subject{subject_id}.png', dpi=300)
    plt.show()


def plot_runs(model, optimal_params, trace_df, cycle_info, leuk_table,
        blast_table, subject_id=1, save_fig=True, params_to_fit=None,
        leuk_table_test=None, blast_table_test=None, initialization=None,
        use_neut=False):
    """
    Plots the optimal parameters as well as the MCMC trace.
    """
    if params_to_fit is None:
        params_to_fit = param_names
    # run the model for the optimal params.
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(f'Patient {subject_id}')
    if leuk_table_test is None:
        if use_neut:
            axes[0].scatter(leuk_table['lab_date'], leuk_table.b_neut, label='neutrophils')
        else:
            axes[0].scatter(leuk_table['lab_date'], leuk_table.b_leuk, label='leukocytes')
    else:
        if use_neut:
            axes[0].scatter(leuk_table['lab_date'], leuk_table.b_neut, label='neut (train)')
            axes[0].scatter(leuk_table_test['lab_date'], leuk_table_test.b_neut, label='neut (test)')
        else:
            axes[0].scatter(leuk_table['lab_date'], leuk_table.b_leuk, label='leuk (train)')
            axes[0].scatter(leuk_table_test['lab_date'], leuk_table_test.b_leuk, label='leuk (test)')
    if blast_table_test is None:
        axes[1].scatter(blast_table['date_bonemarrow'], blast_table.bm_blasts.astype(float), label='blasts')
    else:
        axes[1].scatter(blast_table['date_bonemarrow'], blast_table.bm_blasts.astype(float), label='blasts (train)')
        axes[1].scatter(blast_table_test['date_bonemarrow'], blast_table_test.bm_blasts.astype(float), label='blasts (test)')
    #axes[0].scatter(wbc_data_patient1_neut['lab_date'], wbc_data_patient1_neut.b_neut, label='neutrophils')
    axes[0].set_xlabel('Day')
    axes[0].set_ylabel('WBC counts')
    axes[1].set_xlabel('Day')
    axes[1].set_ylabel('Blast count')
    axes[1].set_ylim(0, 100)
    x_max = max(cycle_info.venetoclax_stop.max(), cycle_info.aza_stop.max(), leuk_table.lab_date.max(), blast_table.date_bonemarrow.max()) + 10
    x_max = int(x_max)
    results = run_model(model, optimal_params, max_time=x_max, n_samples=x_max*10, params_to_fit=params_to_fit, initialization=initialization)
    axes[1].plot(results['time'], results['Xblasts_obs'], label='model blasts')
    if use_neut:
        axes[0].plot(results['time'], results['Xwbc'], label='model neutrophils')
    else:
        axes[0].plot(results['time'], results['Xwbc'], label='model WBCs')
    # plot all results
    for _, r in trace_df.iterrows():
        vals = r[params_to_fit].to_numpy()
        try:
            results = run_model(model, vals, max_time=x_max, n_samples=x_max*10, params_to_fit=params_to_fit,
                    initialization=initialization)
            axes[1].plot(results['time'], results['Xblasts_obs'], linewidth=0.1, color='gray')
            axes[0].plot(results['time'], results['Xwbc'], linewidth=0.1, color='gray')
        except RuntimeError:
            pass
    #axes[1].set_yscale('log')
    for i, (_, row) in enumerate(cycle_info.iterrows()):
        label = None
        # the reason for the i=0 is so that we only have one thing in the legend, rather than a separate legend for every cycle.
        if i == 0:
            label = 'venetoclax admin'
        axes[0].fill_between([row.venetoclax_start, row.venetoclax_stop], 0, 1, alpha=0.2, color='orange',
                         label=label, transform=axes[0].get_xaxis_transform())
        axes[1].fill_between([row.venetoclax_start, row.venetoclax_stop], 0, 1, alpha=0.2, color='orange',
                         label=label, transform=axes[1].get_xaxis_transform())
        if i == 0:
            label = 'azacitidine admin'
        axes[0].fill_between([row.aza_start, row.aza_stop], 0, 1, alpha=0.2, color='green',
                             label=label, transform=axes[0].get_xaxis_transform())
        axes[1].fill_between([row.aza_start, row.aza_stop], 0, 1, alpha=0.2, color='green',
                             label=label, transform=axes[1].get_xaxis_transform())
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'new_patient_model_subject{subject_id}.png', dpi=300)
    plt.show()


def plot_runs_wbc_only(model, optimal_params, trace_df, cycle_info, leuk_table,
        subject_id=1, save_fig=True, params_to_fit=None,
        leuk_table_test=None, initialization=None, use_neut=False, label='neut',
        selections=None):
    """
    Plots the optimal parameters as well as the MCMC trace.
    """
    # TODO: plot runs
    import matplotlib.pyplot as plt
    if params_to_fit is None:
        params_to_fit = param_names
    if use_neut:
        leuk_name='b_neut'
    else:
        leuk_name='b_leuk'
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.set_xlabel('Day')
    ax.set_ylabel(leuk_name)
    fig.suptitle(f'Patient {subject_id}')
    if leuk_table_test is None:
        ax.scatter(leuk_table['lab_date'], leuk_table[leuk_name], label=label)
    else:
        ax.scatter(leuk_table['lab_date'], leuk_table[leuk_name], label=label + ' (train)')
        ax.scatter(leuk_table_test['lab_date'], leuk_table_test[leuk_name], label=label+' (test)')
    # run the model for the optimal params.
        #axes[0].scatter(wbc_data_patient1_neut['lab_date'], wbc_data_patient1_neut.b_neut, label='neutrophils')
    x_max = max(cycle_info.venetoclax_stop.max(), cycle_info.aza_stop.max(), leuk_table.lab_date.max()) + 10
    x_max = int(x_max)
    results = run_model(model, optimal_params, max_time=x_max, n_samples=x_max*10, params_to_fit=params_to_fit, initialization=initialization,
            selections=selections)
    ax.plot(results['time'], results['Xwbc'], label='model results')
    # plot all results
    for _, r in trace_df.iterrows():
        vals = r[params_to_fit].to_numpy()
        try:
            results = run_model(model, vals, max_time=x_max, n_samples=x_max*10, params_to_fit=params_to_fit, initialization=initialization,
                    selections=selections)
            ax.plot(results['time'], results['Xwbc'], linewidth=0.1, color='gray')
        except RuntimeError:
            pass
    for i, (_, row) in enumerate(cycle_info.iterrows()):
        label = None
        # the reason for the i=0 is so that we only have one thing in the legend, rather than a separate legend for every cycle.
        if i == 0:
            label = 'venetoclax admin'
        ax.fill_between([row.venetoclax_start, row.venetoclax_stop], 0, 1, alpha=0.2, color='orange',
                         label=label, transform=ax.get_xaxis_transform())
        if i == 0:
            label = 'azacitidine admin'
        ax.fill_between([row.aza_start, row.aza_stop], 0, 1, alpha=0.2, color='green',
                             label=label, transform=ax.get_xaxis_transform())
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'new_patient_model_subject{subject_id}.png', dpi=300)
    plt.show()


def classification_results(model_output, model_times, data_vals, data_times, threshold=0.5):
    """
    Returns the classification results for the model trajectory.

    """
    start_time = model_times[0]
    end_time = model_times[-1]
    time_scale = (end_time - start_time)/(len(model_times))
    model_vals = []
    for j, t in enumerate(data_times):
        index0 = int((t - start_time)/time_scale)
        index1 = index0 + 1
        if index0 <= 0:
            index0 = 0
            index1 = 0
            model_val = model_output[0]
            model_vals.append(model_val)
        else:
            current_time = model_times[index1]
            prev_time = model_times[index0]
            factor = (t - prev_time)/(current_time - prev_time)
            model_val = factor*model_output[index1] + (1-factor)*model_output[index0]
            model_vals.append(model_val)
    # calculate classes
    data_classes = (data_vals < threshold)
    model_vals = np.array(model_vals)
    model_classes = (model_vals < threshold)
    return model_classes, data_classes


def calculate_errors(model, params, cycle_info, leuk_table, blast_table,
        ode_samples_per_day=20, params_to_fit=None, wbc_only=False,
        use_neut=False, initialization=None, selections=None,
        error_fn='rmse', blasts_only=False):
    # TODO: corrcoef???
    from tellurium_model_fitting import rmse
    if params_to_fit is None:
        params_to_fit = param_names
    x_max = max(cycle_info.venetoclax_stop.max(), cycle_info.aza_stop.max(), leuk_table.lab_date.max(), blast_table.date_bonemarrow.max()) + 10
    x_max = int(x_max)
    results = run_model(model, params, max_time=x_max, n_samples=x_max*ode_samples_per_day, params_to_fit=params_to_fit,
            initialization=initialization, selections=selections)
    leuk_table = leuk_table.reset_index()
    blast_table = blast_table.reset_index()
    if blasts_only:
        rmse_val = 0
    else:
        if use_neut:
            rmse_val = rmse(results['Xwbc'], results['time'], leuk_table['b_neut'], leuk_table['lab_date'], error_fn=error_fn)
        else:
            rmse_val = rmse(results['Xwbc'], results['time'], leuk_table['b_leuk'], leuk_table['lab_date'], error_fn=error_fn)
    if not wbc_only:
        rmse_blasts = rmse(results['Xblasts_obs'], results['time'], blast_table['bm_blasts'].astype(float), blast_table['date_bonemarrow'], error_fn=error_fn)
    else:
        rmse_blasts = 0
    return rmse_val, rmse_blasts, results


def calculate_classification_errors(model, params, cycle_info, leuk_table,
        ode_samples_per_day=20, params_to_fit=None,
        use_neut=False, initialization=None, selections=None):
    """
    Calculates the classification results...
    """
    if params_to_fit is None:
        params_to_fit = param_names
    x_max = max(cycle_info.venetoclax_stop.max(), cycle_info.aza_stop.max(), leuk_table.lab_date.max()) + 10
    x_max = int(x_max)
    results = run_model(model, params, max_time=x_max, n_samples=x_max*ode_samples_per_day, params_to_fit=params_to_fit,
            initialization=initialization, selections=selections)
    leuk_table = leuk_table.reset_index()
    if use_neut:
        rmse_val = rmse(results['Xwbc'], results['time'], leuk_table['b_neut'], leuk_table['lab_date'])
    else:
        rmse_val = rmse(results['Xwbc'], results['time'], leuk_table['b_leuk'], leuk_table['lab_date'])
    return rmse_val, results





def plot_runs_area(model, optimal_params, trace_df, cycle_info, leuk_table,
        blast_table, subject_id=1, save_fig=True, params_to_fit=None,
        leuk_table_test=None, blast_table_test=None, initialization=None,
        use_neut=False, percentiles=50):
    """
    Plots the optimal parameters as well as the MCMC trace.

    percentiles - 50 = middle 50% (25-75%), 90 = 5-95%, etc.
    """
    wbc_label = 'WBC'
    if use_neut:
        wbc_label = 'neut'
    if params_to_fit is None:
        params_to_fit = param_names
    # run the model for the optimal params.
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(f'Patient {subject_id}')
    if leuk_table_test is None:
        if use_neut:
            axes[0].scatter(leuk_table['lab_date'], leuk_table.b_neut, label='neutrophils')
        else:
            axes[0].scatter(leuk_table['lab_date'], leuk_table.b_leuk, label='leukocytes')
    else:
        if use_neut:
            axes[0].scatter(leuk_table['lab_date'], leuk_table.b_neut, label='neut (train)')
            axes[0].scatter(leuk_table_test['lab_date'], leuk_table_test.b_neut, label='neut (test)')
        else:
            axes[0].scatter(leuk_table['lab_date'], leuk_table.b_leuk, label='leuk (train)')
            axes[0].scatter(leuk_table_test['lab_date'], leuk_table_test.b_leuk, label='leuk (test)')
    if blast_table_test is None:
        axes[1].scatter(blast_table['date_bonemarrow'], blast_table.bm_blasts.astype(float), label='blasts')
    else:
        axes[1].scatter(blast_table['date_bonemarrow'], blast_table.bm_blasts.astype(float), label='blasts (train)')
        axes[1].scatter(blast_table_test['date_bonemarrow'], blast_table_test.bm_blasts.astype(float), label='blasts (test)')
    axes[0].set_xlabel('Day')
    axes[0].set_ylabel(f'{wbc_label} counts')
    axes[1].set_xlabel('Day')
    axes[1].set_ylabel('Blast count')
    axes[1].set_ylim(0, 100)
    x_max = max(cycle_info.venetoclax_stop.max(), cycle_info.aza_stop.max(), leuk_table.lab_date.max(), blast_table.date_bonemarrow.max()) + 10
    x_max = int(x_max)
    all_results = []
    for _, r in trace_df.iterrows():
        vals = r[params_to_fit].to_numpy()
        try:
            results = run_model(model, vals, max_time=x_max, n_samples=x_max*10,
                                params_to_fit=params_to_fit, initialization=initialization)
            all_results.append(results)
        except RuntimeError:
            pass
    times = all_results[0]['time']
    all_results_xwbc = np.vstack([x['Xwbc'] for x in all_results])
    all_results_xblasts = np.vstack([x['Xblasts_obs'] for x in all_results])
    top_wbc = np.percentile(all_results_xwbc, 50+percentiles/2, 0)
    bottom_wbc = np.percentile(all_results_xwbc, 50-percentiles/2, 0)
    median_wbc = np.percentile(all_results_xwbc, 50, 0)
    top_blasts = np.percentile(all_results_xblasts, 50+percentiles/2, 0)
    bottom_blasts = np.percentile(all_results_xblasts, 50-percentiles/2, 0)
    median_blasts = np.percentile(all_results_xblasts, 50, 0)
    axes[0].plot(times, median_wbc, label=f'model {wbc_label}')
    axes[0].fill_between(times, top_wbc, bottom_wbc, alpha=0.1, color='blue')
    axes[1].plot(times, median_blasts, label='model blasts')
    axes[1].fill_between(times, top_blasts, bottom_blasts, alpha=0.1, color='blue')
    for i, (_, row) in enumerate(cycle_info.iterrows()):
        label = None
        # the reason for the i=0 is so that we only have one thing in the legend, rather than a separate legend for every cycle.
        if i == 0:
            label = 'venetoclax admin'
        axes[0].fill_between([row.venetoclax_start, row.venetoclax_stop], 0, 1, alpha=0.2, color='orange',
                         label=label, transform=axes[0].get_xaxis_transform())
        axes[1].fill_between([row.venetoclax_start, row.venetoclax_stop], 0, 1, alpha=0.2, color='orange',
                         label=label, transform=axes[1].get_xaxis_transform())
        if i == 0:
            label = 'azacitidine admin'
        axes[0].fill_between([row.aza_start, row.aza_stop], 0, 1, alpha=0.2, color='green',
                             label=label, transform=axes[0].get_xaxis_transform())
        axes[1].fill_between([row.aza_start, row.aza_stop], 0, 1, alpha=0.2, color='green',
                             label=label, transform=axes[1].get_xaxis_transform())
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'new_patient_model_subject{subject_id}.png', dpi=300)
    plt.show()
