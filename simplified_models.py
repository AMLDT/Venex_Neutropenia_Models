#!/usr/bin/env python
# coding: utf-8

# # Simplified models
# 
# What are the simplest models that we could get away with using?
# 
# We tried M2, which had a simple on-off switch for the PK component, but that didn't really work very well. Maybe we should have a somewhat different model?
# 
# Are there additional simplifications we could do?
# 
# For venetoclax: E_ven_wbc is around 1-2 during the cycle, and doesn't oscillate *that* much (depending on parameter?). 
# 
# For aza: E_aza_wbc peaks at around 3? It's definitely not a continuous thing, it should be a pulse of maybe around 3 hours?


import numpy as np

import pymc as pm

from new_patient_model import generate_forward_function

from new_patient_additional_model_descs import generate_forward_function_wbc_only


# ## M2 model descriptions


model_desc_m2 = """
//Aza PK model

in_aza_treatment = 0;
in_ven_treatment = 0;

// WBCs (proliferating, transit, mature)
species $Xprol; species $Xtr; species $Xwbc; 

// simplified drug effect model
// effect
E_ven_wbc := in_ven_treatment*slope_ven_wbc; 
E_aza_wbc := in_aza_treatment*slope_aza_wbc;

// using the M3 model structure from Jost 2019
eps = 1e-8;
wbc_drug := 1 - E_ven_wbc - E_aza_wbc;
F := piecewise(wbc_drug*ktr, wbc_drug < 0, wbc_drug*ktr*((B+eps)/(Xwbc+eps))^gam);
G := ktr;

Xprol' = - G * Xprol + F * Xprol;
Xtr' =  G * Xprol - G * Xtr;
Xwbc' = G * Xtr - kwbc * Xwbc;

// leukemic blasts (bone marrow, blood)
species $Xl1; species $Xl2;
species $Xblasts; species $Xblasts_obs;

// PD model - leukemic blasts
E_ven_blast := slope_ven_blast * in_ven_treatment;
E_aza_blast := slope_aza_blast * in_aza_treatment;

klc := 1/(1 + c1*Xwbc + c2*Xl2);

Xl1' = (2*a1*klc - 1) * p1 * Xl1 - p1 * (E_ven_blast + E_aza_blast) * Xl1;
Xl2' = 2*(1 - a1*klc) * p1 * Xl1 - d2 * Xl2;

// cellularity
CR = 0.5;
// DB is the "approximate maximal tumor burden"
DB = 10^12;
// Xblasts is the observed blast concentration.
Xblasts := 100*(Xl1 + 0.005*Xtr)/(CR*DB);
Xblasts0 = 100*(Xl1 + 0.005*Xtr)/(CR*DB);
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

slope_ven_blast = 1.1;//7.94; // unit: L/micromol
slope_aza_blast = 1.1; //;
slope_ven_wbc = 1.1;//7.94; // unit: L/micromol
slope_aza_wbc = 1.1; //;
p1 = 0.1; // leukemic blast proliferation rate

// initial values
Xprol = B*kwbc/ktr;
Xtr = B*kwbc/ktr;
Xwbc = B;
Xl1 = 0.04;
"""

param_names_m2 = ['ktr', 'gam', 'slope_ven_blast', 'slope_aza_blast', 'slope_ven_wbc', 'slope_aza_wbc', 'B', 'Xl1']
param_names_m2_with_b0 = ['ktr', 'gam', 'slope_ven_blast', 'slope_aza_blast', 'slope_ven_wbc', 'slope_aza_wbc', 'B', 'Xl1', 'B0']
# fitting the leukemic blast proliferation rate as a parameter
param_names_m2_with_prolif = ['ktr', 'gam', 'slope_ven_blast', 'slope_aza_blast', 'slope_ven_wbc', 'slope_aza_wbc', 'p1', 'B', 'Xl1', 'B0']
param_bounds_m2 = [(0, 1), (0, 1), (0, 5), (0, 5), (0, 5), (0, 5), (0, 10), (0, 10), (0, 10)]


def initialization_fn_m2(model, param_vals):
    # initializing some of the model params to equilibrium values?
    ktr = param_vals[0]
    B = param_vals[6]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B*kwbc/ktr)
    model.setValue('Xtr', B*kwbc/ktr)
    model.setValue('Xwbc', B)
    return model

def initialization_fn_m2_with_b0(model, param_vals):
    # initializing some of the model params to equilibrium values?
    ktr = param_vals[0]
    B = param_vals[6]
    B0 = param_vals[8]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B*kwbc/ktr)
    model.setValue('Xtr', B*kwbc/ktr)
    model.setValue('Xwbc', B0)
    return model

def initialization_fn_m2_2(model, param_vals):
    # initializing some of the model params to equilibrium values?
    ktr = param_vals[0]
    B0 = param_vals[8]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B0*kwbc/ktr)
    model.setValue('Xtr', B0*kwbc/ktr)
    model.setValue('Xwbc', B0)
    return model

def generate_dosing_component_m2(venex_cycles, aza_cycles):
    """
    Params
    ------
        venex_cycles - pandas dataframe with columns start, end, dose
        aza_cycles - pandas dataframe with columns start, end, dose
    """
    output_str = ''
    # TODO: dosages should be in effect...
    # default dosages are: 400 for ven, 120 for aza. dosages should be proportions of those.
    for _, row in venex_cycles.iterrows():
        start = row.start
        end = row.end
        dose = row.dose/400
        output_str += f'at time >= {start} : in_ven_treatment = {dose};\n'
        output_str += f'at time >= {end} : in_ven_treatment = 0;\n'
    output_str += '\n'
    for _, row in aza_cycles.iterrows():
        start = row.start
        end = row.end
        dose = row.dose/120
        # this represents the pulse of injections - 
        for day in range(start, end):
            output_str += f'at time >= {day} : in_aza_treatment = {dose};\n'
            output_str += f'at time >= {day} + 0.2 : in_aza_treatment = 0;\n'
    return output_str


def build_pm_model_m2(model, wbc, blasts, theta=None, n_samples=None, params_to_fit=None,
        initialization=None, use_b0=False, uniform_prior=False, **params):
    """
    Builds a PyMC model
    """
    if params_to_fit is None:
        params_to_fit = param_names_m2
    if theta is None:
        model.resetToOrigin()
        default_params = [model.getValue(x) for x in params_to_fit]
        theta = default_params
    print('theta:', theta)
    pytensor_forward_function = generate_forward_function(model, wbc, blasts, n_samples=n_samples,
            params_to_fit=params_to_fit, initialization=initialization)
    wbc_data = wbc[1].to_numpy(dtype=np.float64)
    blast_data = blasts[1].to_numpy(dtype=np.float64)
    print('wbc_data:', wbc_data)
    print('blast_data:', blast_data)
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
            slope_ven_blast = pm.Uniform("slope_ven_blast", lower=0, upper=5, initval=theta[2])
            slope_aza_blast = pm.Uniform("slope_aza_blast", lower=0, upper=5, initval=theta[3])
            slope_ven_wbc = pm.Uniform("slope_ven_wbc", lower=0, upper=5, initval=theta[4])
            slope_aza_wbc = pm.Uniform("slope_aza_wbc", lower=0, upper=5, initval=theta[5])
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


# ### WBC-only model


##########
# model m2_wbc_only
# model with only a wbc component and no blast component...
model_desc_m2_wbc_only = """
// WBCs (proliferating, transit, mature)
in_ven_treatment = 0;
in_aza_treatment = 0;

species $Xprol; species $Xtr; species $Xwbc; 

// PD model - WBCs
// effect
E_ven_wbc := slope_ven_wbc * in_ven_treatment;
E_aza_wbc := slope_aza_wbc * in_aza_treatment;

// using the M3 model structure from Jost 2019
wbc_drug := 1 - E_ven_wbc - E_aza_wbc;
F := piecewise(wbc_drug*ktr, wbc_drug < 0, wbc_drug*ktr*(B/Xwbc)^gam);
G := ktr;

// lmao a hack to deal with numerical instabilities
at Xwbc < 1e-5: Xwbc = 1e-5;

Xprol' = - G * Xprol + F * Xprol;
Xtr' =  G * Xprol - G * Xtr;
Xwbc' = G * Xtr - kwbc * Xwbc;

// leukemic blasts (bone marrow, blood)
// PD model of WBCs and leukemic blasts
// source is https://link.springer.com/article/10.1007/s11095-014-1429-9
kwbc = 2.3765; 
// source is https://www.nature.com/articles/s41598-018-21115-4

// parameters to fit - default values can be found in table 1 of the original paper.
ktr = 0.236; // transition rate - unit: 1/day
gam = 0.651; // feedback of G-CSF on WBC proliferation - MAKE SURE THIS VARIABLE IS NAMED gam
B = 2;//4.67; // unit: G/L
B0 = 2;//4.67;

// CANNOT name this variable "gamma" bc that's a reserved function
slope_ven_wbc = 1;//7.94; // unit: L/micromol
slope_aza_wbc = 1; //;

// initial values
Xprol = B*kwbc/ktr;
Xtr = B*kwbc/ktr;
Xwbc = B;
"""

param_names_m2_wbc_only = ['ktr', 'gam', 'slope_ven_wbc', 'slope_aza_wbc', 'B', 'B0']

def initialization_fn_m2_wbc_only(model, param_vals):
    # initializing some of the model params to equilibrium values?
    ktr = param_vals[0]
    B = param_vals[4]
    B0 = param_vals[5]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B*kwbc/ktr)
    model.setValue('Xtr', B*kwbc/ktr)
    model.setValue('Xwbc', B0)
    return model


def build_pm_model_m2_wbc_only(model, wbc, theta=None, n_samples=None, params_to_fit=None, **params):
    """
    Builds a PyMC model
    """
    if params_to_fit is None:
        params_to_fit = param_names_m2_wbc_only
    if theta is None:
        model.resetToOrigin()
        default_params = [model.getValue(x) for x in params_to_fit]
        theta = default_params
    pytensor_forward_function = generate_forward_function_wbc_only(model, wbc, n_samples=n_samples, params_to_fit=params_to_fit)
    wbc_data = wbc[1].to_numpy(dtype=np.float64)
    print('theta:', theta)
    print('wbc_data:', wbc_data)
    with pm.Model() as new_patient_model_pm:
        # priors
        # TODO: uniform priors for slope?
        ktr = pm.TruncatedNormal("ktr", mu=theta[0], sigma=theta[0]/2, lower=0, initval=theta[0])
        gam = pm.TruncatedNormal("gam", mu=theta[1], sigma=theta[1]/3, lower=0, initval=theta[1])
        slope_ven_wbc = pm.TruncatedNormal("slope_ven_wbc", mu=theta[2], sigma=theta[2]/2, lower=0, initval=theta[2])
        slope_aza_wbc = pm.TruncatedNormal("slope_aza_wbc", mu=theta[3], sigma=theta[3]/2, lower=0, initval=theta[3])
        B = pm.TruncatedNormal("B", mu=theta[4], sigma=theta[4]/2, lower=0, initval=theta[4])
        B0 = pm.TruncatedNormal("B0", mu=theta[5], sigma=theta[5]/2, lower=0, initval=theta[5])
        sigma_wbc = pm.HalfNormal("sigma_wbc", 5)

        # ODE solution function
        ode_solution = pytensor_forward_function(pm.math.stack([ktr, gam,
                                                               slope_ven_wbc,
                                                               slope_aza_wbc, B, B0]))

        # likelihood
        pm.Normal("Y_neut_obs", mu=ode_solution, sigma=sigma_wbc, observed=wbc_data)
    return new_patient_model_pm

def build_pm_model_m2_wbc_only_uniform_prior(model, wbc, theta=None, n_samples=None, params_to_fit=None, **params):
    """
    Builds a PyMC model
    """
    if params_to_fit is None:
        params_to_fit = param_names_m2_wbc_only
    if theta is None:
        model.resetToOrigin()
        default_params = [model.getValue(x) for x in params_to_fit]
        theta = default_params
    pytensor_forward_function = generate_forward_function_wbc_only(model, wbc, n_samples=n_samples, params_to_fit=params_to_fit)
    wbc_data = wbc[1].to_numpy(dtype=np.float64)
    print('wbc_data:', wbc_data)
    with pm.Model() as new_patient_model_pm:
        # priors
        # TODO: uniform priors for slope?
        ktr = pm.Uniform("ktr", lower=0, upper=1, initval=theta[0])
        gam = pm.Uniform("gam", lower=0, upper=1, initval=theta[1])
        slope_ven_wbc = pm.Uniform("slope_ven_wbc", lower=0, upper=10, initval=theta[2])
        slope_aza_wbc = pm.Uniform("slope_aza_wbc", lower=0, upper=10, initval=theta[3])
        B = pm.Uniform("B", lower=0, upper=15, initval=theta[4])
        B0 = pm.Uniform("B0", lower=0, upper=15, initval=theta[5])
        # Prior for initial blasts should go between 0.0 and 1
        sigma_wbc = pm.HalfNormal("sigma_wbc", 5)

        # ODE solution function
        ode_solution = pytensor_forward_function(pm.math.stack([ktr, gam,
                                                               slope_ven_wbc,
                                                               slope_aza_wbc, B, B0]))

        # likelihood
        pm.Normal("Y_neut_obs", mu=ode_solution, sigma=sigma_wbc, observed=wbc_data)
    return new_patient_model_pm


# ### M2b - simplified model with M4b-style blast dynamics


model_desc_m2b = """
//Aza PK model

in_aza_treatment = 0;
in_ven_treatment = 0;

// WBCs (proliferating, transit, mature)
species $Xprol; species $Xtr; species $Xwbc; 

// simplified drug effect model
// effect
E_ven_wbc := in_ven_treatment*slope_ven_wbc; 
E_aza_wbc := in_aza_treatment*slope_aza_wbc;

// using the M3 model structure from Jost 2019
eps = 1e-8;
wbc_drug := 1 - E_ven_wbc - E_aza_wbc;
F := piecewise(wbc_drug*ktr, wbc_drug < 0, wbc_drug*ktr*((B+eps)/(Xwbc+eps))^gam);
G := ktr/(1+bi*Xblasts_obs);
D := (Xprol + d*Xl1);
BM_eq = 4*B*kwbc/ktr;
dc_piece := D - BM_eq;
dc := piecewise(0, dc_piece <= 0, d*dc_piece);

Xprol' = Xprol*(F - G - dc);
Xtr' =  G * Xprol - G * Xtr;
Xwbc' = G * Xtr - kwbc * Xwbc;

// leukemic blasts (bone marrow, blood)
species $Xl1; species $Xl2;
species $Xblasts; species $Xblasts_obs;

// PD model - leukemic blasts
E_ven_blast := slope_ven_blast * in_ven_treatment;
E_aza_blast := slope_aza_blast * in_aza_treatment;

klc := 1/(1 + c1*Xwbc + c2*Xl2);

Xl1' = Xl1*(2*p1*a1 - p1 - p1*(E_ven_blast+E_aza_blast) - dc);
Xl2' = 2*(1 - a1*klc) * p1 * Xl1 - d2 * Xl2;

// cellularity
CR = 0.5;
// DB is the "approximate maximal tumor burden"
DB = 10^12;
// Xblasts is the observed blast concentration.
//Xblasts := 100*(Xl1 + 0.005*Xtr)/(CR*DB);
//Xblasts0 = 100*(Xl1 + 0.005*Xtr)/(CR*DB);
// okay... I don't know what to do exactly with Xl1.
Xblasts_obs := 100*(Xl1/(Xl1 + Xprol + 1e-2));

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

slope_ven_blast = 1.1;//7.94; // unit: L/micromol
slope_aza_blast = 1.1; //;
slope_ven_wbc = 1.1;//7.94; // unit: L/micromol
slope_aza_wbc = 1.1; //;
p1 = 0.1; // leukemic blast proliferation rate
d = 1;
bi = 0.1;

// initial values
Xprol = B*kwbc/ktr;
Xtr = B*kwbc/ktr;
Xwbc = B;
Xl1 = 0.04;
"""

param_names_m2b = ['ktr', 'gam', 'slope_ven_blast', 'slope_aza_blast', 'slope_ven_wbc', 'slope_aza_wbc', 'B', 'Xl1', 'B0', 'p1', 'bi']
param_bounds_m2b = [(0, 1), (0, 1), (0, 5), (0, 5), (0, 5), (0, 5), (0, 10), (0, 10), (0, 10), (0, 1), (0, 1)]

# TODO: blast only sub-model with the same desc but different parameters

param_names_m2b_blast_only = ['slope_ven_blast', 'slope_aza_blast', 'Xl1', 'p1']
param_bounds_m2b_blast_only = [(0, 5), (0, 5), (0, 10), (0, 1)]

# no initialization!
def initialization_fn_m2b_blast_only(model, param_vals):
    return model

def initialization_fn_m2b(model, param_vals):
    # initializing some of the model params to equilibrium values?
    ktr = param_vals[0]
    B = param_vals[6]
    B0 = param_vals[8]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B*kwbc/ktr)
    model.setValue('Xtr', B*kwbc/ktr)
    model.setValue('Xwbc', B0)
    model.setValue('BM_eq', 4*B*kwbc/ktr)
    return model

def initialization_fn_m2b_with_b0(model, param_vals):
    # initializing some of the model params to equilibrium values?
    ktr = param_vals[0]
    B = param_vals[6]
    B0 = param_vals[8]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B*kwbc/ktr)
    model.setValue('Xtr', B*kwbc/ktr)
    model.setValue('Xwbc', B0)
    model.setValue('BM_eq', 4*B*kwbc/ktr)
    return model

def initialization_fn_m2b_2(model, param_vals):
    # initializing some of the model params to equilibrium values?
    ktr = param_vals[0]
    B = param_vals[6]
    B0 = param_vals[8]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B0*kwbc/ktr)
    model.setValue('Xtr', B0*kwbc/ktr)
    model.setValue('Xwbc', B0)
    model.setValue('BM_eq', 4*B*kwbc/ktr)
    return model


def build_pm_model_m2b(model, wbc, blasts, theta=None, n_samples=None, params_to_fit=None,
        initialization=None, use_b0=False, uniform_prior=False, **params):
    """
    Builds a PyMC model
    """
    if params_to_fit is None:
        params_to_fit = param_names_m2b
    if theta is None:
        model.resetToOrigin()
        default_params = [model.getValue(x) for x in params_to_fit]
        theta = default_params
    print('theta:', theta)
    pytensor_forward_function = generate_forward_function(model, wbc, blasts, n_samples=n_samples,
            params_to_fit=params_to_fit, initialization=initialization)
    wbc_data = wbc[1].to_numpy(dtype=np.float64)
    blast_data = blasts[1].to_numpy(dtype=np.float64)
    print('wbc_data:', wbc_data)
    print('blast_data:', blast_data)
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
            p1 = pm.TruncatedNormal('p1', mu=theta[9], sigma=theta[9]/2, lower=0, initval=theta[9])
            bi = pm.TruncatedNormal('bi', mu=theta[10], sigma=theta[10]/2, lower=0, upper=1, initval=theta[10])
        else:
            ktr = pm.Uniform("ktr", lower=0, upper=1, initval=theta[0])
            gam = pm.Uniform("gam", lower=0, upper=1, initval=theta[1])
            slope_ven_blast = pm.Uniform("slope_ven_blast", lower=0, upper=5, initval=theta[2])
            slope_aza_blast = pm.Uniform("slope_aza_blast", lower=0, upper=5, initval=theta[3])
            slope_ven_wbc = pm.Uniform("slope_ven_wbc", lower=0, upper=5, initval=theta[4])
            slope_aza_wbc = pm.Uniform("slope_aza_wbc", lower=0, upper=5, initval=theta[5])
            B = pm.Uniform("B", lower=0, upper=10, initval=theta[6])
            if use_b0:
                B0 = pm.Uniform('B0', lower=0, upper=10, initval=theta[7])
            p1 = pm.Uniform('p1', lower=0, upper=1, initval=theta[9])
            bi = pm.Uniform('bi', lower=0, upper=1, initval=theta[10])
        Xl1 = pm.Uniform('Xl1', lower=0, upper=10, initval=theta[7])
        sigma_wbc = pm.HalfNormal("sigma_wbc", 5)
        sigma_blasts = pm.HalfNormal("sigma_blasts", 5)

        # ODE solution function
        ode_solution = pytensor_forward_function(pm.math.stack([ktr, gam,
                                                               slope_ven_blast,
                                                               slope_aza_blast,
                                                               slope_ven_wbc,
                                                               slope_aza_wbc, B, Xl1, p1, bi]))
        if use_b0:
            ode_solution = pytensor_forward_function(pm.math.stack([ktr, gam,
                                                               slope_ven_blast,
                                                               slope_aza_blast,
                                                               slope_ven_wbc,
                                                               slope_aza_wbc, B, Xl1, B0, p1, bi]))
        
        # split up the blasts and WBCs
        wbc_ode = ode_solution[0,:len(wbc_data)]
        blast_ode = ode_solution[1,:len(blast_data)]

        # likelihood
        pm.Normal("Y_wbc_obs", mu=wbc_ode, sigma=sigma_wbc, observed=wbc_data)
        pm.Normal("Y_blast_obs", mu=blast_ode, sigma=sigma_blasts, observed=blast_data)
    return new_patient_model_pm


def build_pm_model_m2b_blast_only(model, wbc, blasts, theta=None, n_samples=None, params_to_fit=None,
        initialization=None, use_b0=False, uniform_prior=False, **params):
    """
    Builds a PyMC model
    """
    if params_to_fit is None:
        params_to_fit = param_names_m2b_blast_only
    if theta is None:
        model.resetToOrigin()
        default_params = [model.getValue(x) for x in params_to_fit]
        theta = default_params
    print('theta:', theta)
    pytensor_forward_function = generate_forward_function(model, wbc, blasts, n_samples=n_samples,
            params_to_fit=params_to_fit, initialization=initialization)
    blast_data = blasts[1].to_numpy(dtype=np.float64)
    print('blast_data:', blast_data)
    with pm.Model() as new_patient_model_pm:
        # priors
        # TODO: uniform priors for slope?
        if not uniform_prior:
            slope_ven_blast = pm.TruncatedNormal("slope_ven_blast", mu=theta[2], sigma=theta[2]/2, lower=0, initval=theta[2])
            slope_aza_blast = pm.TruncatedNormal("slope_aza_blast", mu=theta[3], sigma=theta[3]/2, lower=0, initval=theta[3])
            p1 = pm.TruncatedNormal('p1', mu=theta[9], sigma=theta[9]/2, lower=0, initval=theta[9])
        else:
            slope_ven_blast = pm.Uniform("slope_ven_blast", lower=0, upper=5, initval=theta[2])
            slope_aza_blast = pm.Uniform("slope_aza_blast", lower=0, upper=5, initval=theta[3])
            p1 = pm.Uniform('p1', lower=0, upper=1, initval=theta[9])
        Xl1 = pm.Uniform('Xl1', lower=0, upper=10, initval=theta[7])
        sigma_blasts = pm.HalfNormal("sigma_blasts", 5)

        # ODE solution function
        ode_solution = pytensor_forward_function(pm.math.stack([slope_ven_blast,
                                                               slope_aza_blast,
                                                               Xl1, p1]))
        if use_b0:
            ode_solution = pytensor_forward_function(pm.math.stack([slope_ven_blast,
                                                               slope_aza_blast,
                                                               Xl1, p1]))
        
        # split up the blasts and WBCs
        blast_ode = ode_solution[1,:len(blast_data)]

        # likelihood
        pm.Normal("Y_blast_obs", mu=blast_ode, sigma=sigma_blasts, observed=blast_data)
    return new_patient_model_pm



# ### M2b-WBC-only


model_desc_m2b_wbc_only = """
//Aza PK model

in_aza_treatment = 0;
in_ven_treatment = 0;

// WBCs (proliferating, transit, mature)
species $Xprol; species $Xtr; species $Xwbc; species $Xblasts_obs;

// simplified drug effect model
// effect
E_ven_wbc := in_ven_treatment*slope_ven_wbc; 
E_aza_wbc := in_aza_treatment*slope_aza_wbc;

// using the M3 model structure from Jost 2019
eps = 1e-8;
wbc_drug := 1 - E_ven_wbc - E_aza_wbc;
F := piecewise(wbc_drug*ktr, wbc_drug < 0, wbc_drug*ktr*((B+eps)/(Xwbc+eps))^gam);
G := ktr/(1+bi*Xblasts_obs);
D := (Xprol + d*Xl1);
BM_eq = 4*B*kwbc/ktr;
dc_piece := D - BM_eq;
dc := piecewise(0, dc_piece <= 0, d*dc_piece);

Xprol' = Xprol*(F - G - dc);
Xtr' =  G * Xprol - G * Xtr;
Xwbc' = G * Xtr - kwbc * Xwbc;


// PD model of WBCs and leukemic blasts
// source is https://link.springer.com/article/10.1007/s11095-014-1429-9
kwbc = 2.3765; 
keg = 0.592*24;
kANC = 5.64*24;
beta = 0.234;
// source is https://www.nature.com/articles/s41598-018-21115-4
Xl1 := Xblasts_obs/100;
a1 = 0.875;
d2 = 2.3;
c1 = 0.01;
c2 = 0.01;

// parameters to fit - default values can be found in table 1 of the original paper.
ktr = 0.236; // transition rate - unit: 1/day
gam = 0.651; // feedback of G-CSF on WBC proliferation - MAKE SURE THIS VARIABLE IS NAMED gam
B = 4.67; // unit: G/L
B0 = 4.67;

slope_ven_blast = 1.1;//7.94; // unit: L/micromol
slope_aza_blast = 1.1; //;
slope_ven_wbc = 1.1;//7.94; // unit: L/micromol
slope_aza_wbc = 1.1; //;
p1 = 0.1; // leukemic blast proliferation rate
d = 1;
bi = 0.1;

// initial values
Xprol = B*kwbc/ktr;
Xtr = B*kwbc/ktr;
Xwbc = B;
"""

param_names_m2b_wbc_only = ['ktr', 'gam', 'slope_ven_wbc', 'slope_aza_wbc', 'B', 'B0', 'bi']
param_bounds_m2b_wbc_only = [(0, 1), (0, 1), (0, 5), (0, 5), (0, 10), (0, 10), (0, 1)]


def initialization_fn_m2b_wbc_only(model, param_vals):
    # initializing some of the model params to equilibrium values?
    ktr = param_vals[0]
    B = param_vals[4]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B*kwbc/ktr)
    model.setValue('Xtr', B*kwbc/ktr)
    model.setValue('Xwbc', B)
    model.setValue('BM_eq', 4*B*kwbc/ktr)
    return model


def initialization_fn_m2b_2_wbc_only(model, param_vals):
    # initializing some of the model params to equilibrium values?
    ktr = param_vals[0]
    B = param_vals[4]
    B0 = param_vals[5]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B0*kwbc/ktr)
    model.setValue('Xtr', B0*kwbc/ktr)
    model.setValue('Xwbc', B0)
    model.setValue('BM_eq', 4*B*kwbc/ktr)
    return model


def build_pm_model_m2b_wbc_only(model, wbc, theta=None, n_samples=None, params_to_fit=None,
        initialization=None, use_b0=True, uniform_prior=False, **params):
    """
    Builds a PyMC model
    """
    if params_to_fit is None:
        params_to_fit = param_names_m2b_wbc_only
    if theta is None:
        model.resetToOrigin()
        default_params = [model.getValue(x) for x in params_to_fit]
        theta = default_params
    print('theta:', theta)
    pytensor_forward_function = generate_forward_function_wbc_only(model, wbc, n_samples=n_samples,
            params_to_fit=params_to_fit, initialization=initialization)
    wbc_data = wbc[1].to_numpy(dtype=np.float64)
    print('wbc_data:', wbc_data)
    with pm.Model() as new_patient_model_pm:
        # priors
        if not uniform_prior:
            ktr = pm.TruncatedNormal("ktr", mu=theta[0], sigma=theta[0]/2, lower=0, upper=1, initval=theta[0])
            gam = pm.TruncatedNormal("gam", mu=theta[1], sigma=theta[1]/2, lower=0, upper=1, initval=theta[1])
            slope_ven_wbc = pm.TruncatedNormal("slope_ven_wbc", mu=theta[2], sigma=theta[2]/2, lower=0, initval=theta[2])
            slope_aza_wbc = pm.TruncatedNormal("slope_aza_wbc", mu=theta[3], sigma=theta[3]/2, lower=0, initval=theta[3])
            B = pm.TruncatedNormal("B", mu=theta[4], sigma=theta[4]/2, lower=0, initval=theta[4])
            if use_b0:
                B0 = pm.TruncatedNormal('B0', mu=theta[5], sigma=theta[5]/2, lower=0, initval=theta[5])
            bi = pm.TruncatedNormal('bi', mu=theta[6], sigma=theta[6]/2, lower=0, upper=1, initval=theta[6])
        else:
            ktr = pm.Uniform("ktr", lower=0, upper=1, initval=theta[0])
            gam = pm.Uniform("gam", lower=0, upper=1, initval=theta[1])
            slope_ven_wbc = pm.Uniform("slope_ven_wbc", lower=0, upper=5, initval=theta[2])
            slope_aza_wbc = pm.Uniform("slope_aza_wbc", lower=0, upper=5, initval=theta[3])
            B = pm.Uniform("B", lower=0, upper=10, initval=theta[4])
            if use_b0:
                B0 = pm.Uniform('B0', lower=0, upper=10, initval=theta[5])
            bi = pm.Uniform('bi', lower=0, upper=1, initval=theta[6])
        sigma_wbc = pm.HalfNormal("sigma_wbc", 5)
        # ODE solution function
        ode_solution = pytensor_forward_function(pm.math.stack([ktr, gam,
                                                               slope_ven_wbc,
                                                               slope_aza_wbc, B, bi]))
        if use_b0:
            ode_solution = pytensor_forward_function(pm.math.stack([ktr, gam,
                                                               slope_ven_wbc,
                                                               slope_aza_wbc, B, B0, bi]))
        
        wbc_ode = ode_solution

        # likelihood
        pm.Normal("Y_wbc_obs", mu=wbc_ode, sigma=sigma_wbc, observed=wbc_data)
    return new_patient_model_pm


# ### M2c - simplified model with simplified blast dynamics


model_desc_m2c = """
//Aza PK model

in_aza_treatment = 0;
in_ven_treatment = 0;

// WBCs (proliferating, transit, mature)
species $Xprol; species $Xtr; species $Xwbc; 

// simplified drug effect model
// effect
E_ven_wbc := in_ven_treatment*slope_ven_wbc; 
E_aza_wbc := in_aza_treatment*slope_aza_wbc;

// using the M3 model structure from Jost 2019
eps = 1e-8;
wbc_drug := 1 - E_ven_wbc - E_aza_wbc;
F := piecewise(wbc_drug*ktr, wbc_drug < 0, wbc_drug*ktr*((B+eps)/(Xwbc+eps))^gam);
G := ktr/(1 + bi*Xblasts_obs);
D := (Xprol + d*Xl1);
BM_eq = 4*B*kwbc/ktr;
dc_piece := D - BM_eq;
dc := piecewise(0, dc_piece <= 0, d*dc_piece);

Xprol' = Xprol*(F - G - dc);
Xtr' =  G * Xprol - G * Xtr;
Xwbc' = G * Xtr - kwbc * Xwbc;

// leukemic blasts (bone marrow, blood)
species $Xl1; species $Xl2;
species $Xblasts; species $Xblasts_obs;

// PD model - leukemic blasts
E_ven_blast := slope_ven_blast * in_ven_treatment;
E_aza_blast := slope_aza_blast * in_aza_treatment;

klc := 1/(1 + c1*Xwbc + c2*Xl2);

Xl1' = p1*Xl1*(2*a1*klc - a1*klc - Xl1/1 - E_ven_blast - E_aza_blast);
Xl2' = 2*(1 - a1*klc) * p1 * Xl1 - d2 * Xl2;

// Xblasts_obs
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

slope_ven_blast = 1.1;//7.94; // unit: L/micromol
slope_aza_blast = 1.1; //;
slope_ven_wbc = 1.1;//7.94; // unit: L/micromol
slope_aza_wbc = 1.1; //;
p1 = 0.1; // leukemic blast proliferation rate
d = 1;
bi = 0.1;

// initial values
Xprol = B*kwbc/ktr;
Xtr = B*kwbc/ktr;
Xwbc = B;
Xl1 = 0.04;
"""

param_names_m2c = ['ktr', 'gam', 'slope_ven_blast', 'slope_aza_blast', 'slope_ven_wbc', 'slope_aza_wbc', 'B', 'Xl1', 'B0', 'p1', 'bi']
param_bounds_m2c = [(0, 1), (0, 1), (0, 5), (0, 5), (0, 5), (0, 5), (0, 10), (0, 10), (0, 10), (0, 1), (0, 1)]


def initialization_fn_m2c(model, param_vals):
    # initializing some of the model params to equilibrium values?
    ktr = param_vals[0]
    B = param_vals[6]
    B0 = param_vals[8]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B*kwbc/ktr)
    model.setValue('Xtr', B*kwbc/ktr)
    model.setValue('Xwbc', B0)
    model.setValue('BM_eq', 4*B*kwbc/ktr)
    return model

def initialization_fn_m2c_with_b0(model, param_vals):
    # initializing some of the model params to equilibrium values?
    ktr = param_vals[0]
    B = param_vals[6]
    B0 = param_vals[8]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B*kwbc/ktr)
    model.setValue('Xtr', B*kwbc/ktr)
    model.setValue('Xwbc', B0)
    model.setValue('BM_eq', 4*B*kwbc/ktr)
    return model

def initialization_fn_m2c_2(model, param_vals):
    # initializing some of the model params to equilibrium values?
    ktr = param_vals[0]
    B = param_vals[6]
    B0 = param_vals[8]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B0*kwbc/ktr)
    model.setValue('Xtr', B0*kwbc/ktr)
    model.setValue('Xwbc', B0)
    model.setValue('BM_eq', 4*B*kwbc/ktr)
    return model


def build_pm_model_m2c(model, wbc, blasts, theta=None, n_samples=None, params_to_fit=None,
        initialization=None, use_b0=False, uniform_prior=False, **params):
    """
    Builds a PyMC model
    """
    if params_to_fit is None:
        params_to_fit = param_names_m2c
    if theta is None:
        model.resetToOrigin()
        default_params = [model.getValue(x) for x in params_to_fit]
        theta = default_params
    print('theta:', theta)
    pytensor_forward_function = generate_forward_function(model, wbc, blasts, n_samples=n_samples,
            params_to_fit=params_to_fit, initialization=initialization)
    wbc_data = wbc[1].to_numpy(dtype=np.float64)
    blast_data = blasts[1].to_numpy(dtype=np.float64)
    print('wbc_data:', wbc_data)
    print('blast_data:', blast_data)
    with pm.Model() as new_patient_model_pm:
        # priors
        if not uniform_prior:
            ktr = pm.TruncatedNormal("ktr", mu=theta[0], sigma=theta[0]/2, lower=0, upper=1, initval=theta[0])
            gam = pm.TruncatedNormal("gam", mu=theta[1], sigma=theta[1]/2, lower=0, upper=1, initval=theta[1])
            slope_ven_blast = pm.TruncatedNormal("slope_ven_blast", mu=theta[2], sigma=theta[2]/2, lower=0, initval=theta[2])
            slope_aza_blast = pm.TruncatedNormal("slope_aza_blast", mu=theta[3], sigma=theta[3]/2, lower=0, initval=theta[3])
            slope_ven_wbc = pm.TruncatedNormal("slope_ven_wbc", mu=theta[4], sigma=theta[4]/2, lower=0, initval=theta[4])
            slope_aza_wbc = pm.TruncatedNormal("slope_aza_wbc", mu=theta[5], sigma=theta[5]/2, lower=0, initval=theta[5])
            B = pm.TruncatedNormal("B", mu=theta[6], sigma=theta[6]/2, lower=0, initval=theta[6])
            if use_b0:
                B0 = pm.TruncatedNormal('B0', mu=theta[8], sigma=theta[8]/2, lower=0, initval=theta[8])
            p1 = pm.TruncatedNormal('p1', mu=theta[9], sigma=theta[9]/2, lower=0, initval=theta[9])
            bi = pm.TruncatedNormal('bi', mu=theta[10], sigma=theta[10]/2, lower=0, upper=1, initval=theta[10])
        else:
            ktr = pm.Uniform("ktr", lower=0, upper=1, initval=theta[0])
            gam = pm.Uniform("gam", lower=0, upper=1, initval=theta[1])
            slope_ven_blast = pm.Uniform("slope_ven_blast", lower=0, upper=5, initval=theta[2])
            slope_aza_blast = pm.Uniform("slope_aza_blast", lower=0, upper=5, initval=theta[3])
            slope_ven_wbc = pm.Uniform("slope_ven_wbc", lower=0, upper=5, initval=theta[4])
            slope_aza_wbc = pm.Uniform("slope_aza_wbc", lower=0, upper=5, initval=theta[5])
            B = pm.Uniform("B", lower=0, upper=10, initval=theta[6])
            if use_b0:
                B0 = pm.Uniform('B0', lower=0, upper=10, initval=theta[8])
            p1 = pm.Uniform('p1', lower=0, upper=1, initval=theta[9])
            bi = pm.Uniform('bi', lower=0, upper=1, initval=theta[10])
        Xl1 = pm.Uniform('Xl1', lower=0, upper=10, initval=theta[7])
        sigma_wbc = pm.HalfNormal("sigma_wbc", 5)
        sigma_blasts = pm.HalfNormal("sigma_blasts", 5)

        # ODE solution function
        ode_solution = pytensor_forward_function(pm.math.stack([ktr, gam,
                                                               slope_ven_blast,
                                                               slope_aza_blast,
                                                               slope_ven_wbc,
                                                               slope_aza_wbc, B, Xl1, p1, bi]))
        if use_b0:
            ode_solution = pytensor_forward_function(pm.math.stack([ktr, gam,
                                                               slope_ven_blast,
                                                               slope_aza_blast,
                                                               slope_ven_wbc,
                                                               slope_aza_wbc, B, Xl1, B0, p1, bi]))
        
        # split up the blasts and WBCs
        wbc_ode = ode_solution[0,:len(wbc_data)]
        blast_ode = ode_solution[1,:len(blast_data)]

        # likelihood
        pm.Normal("Y_wbc_obs", mu=wbc_ode, sigma=sigma_wbc, observed=wbc_data)
        pm.Normal("Y_blast_obs", mu=blast_ode, sigma=sigma_blasts, observed=blast_data)
    return new_patient_model_pm


#####################################
# M2d - model with much simpler drug component - same killing rate for blasts and neutrophils, venetoclax-only without an aza component...


model_desc_m2d = """
//Aza PK model

in_aza_treatment = 0;
in_ven_treatment = 0;

// WBCs (proliferating, transit, mature)
species $Xprol; species $Xtr; species $Xwbc; 

// simplified drug effect model
// effect
E_ven_wbc := in_ven_treatment*slope_ven; 

// using the M3 model structure from Jost 2019
eps = 1e-8;
wbc_drug := 1 - E_ven_wbc;
F := piecewise(wbc_drug*ktr, wbc_drug < 0, wbc_drug*ktr*((B+eps)/(Xwbc+eps))^gam);
G := ktr/(1 + bi*Xblasts_obs);
D := (Xprol + d*Xl1);
BM_eq = 4*B*kwbc/ktr;
dc_piece := D - BM_eq;
dc := piecewise(0, dc_piece <= 0, d*dc_piece);

Xprol' = Xprol*(F - G - dc);
Xtr' =  G * Xprol - G * Xtr;
Xwbc' = G * Xtr - kwbc * Xwbc;

// leukemic blasts (bone marrow, blood)
species $Xl1; species $Xl2;
species $Xblasts; species $Xblasts_obs;

// PD model - leukemic blasts
E_ven_blast := slope_ven* in_ven_treatment;

//klc := 1/(1 + c1*Xwbc + c2*Xl2);

Xl1' = p1*Xl1*(1 - Xl1 - E_ven_blast);
// a;dfljdkf;sasdf; idc about Xl2
//Xl2' = 2*(1 - a1*klc) * p1 * Xl1 - d2 * Xl2;

// Xblasts_obs
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

slope_ven = 1.1;//7.94; // unit: L/micromol
p1 = 0.1; // leukemic blast proliferation rate
d = 1;
bi = 0.1;

// initial values
Xprol = B*kwbc/ktr;
Xtr = B*kwbc/ktr;
Xwbc = B;
Xl1 = 0.04;
"""

param_names_m2d = ['ktr', 'gam', 'slope_ven', 'B', 'Xl1', 'B0', 'p1', 'bi']
param_bounds_m2d = [(0, 1), (0, 1), (0, 5), (0, 10), (0, 10), (0, 10), (0, 1), (0, 1)]


def initialization_fn_m2d(model, param_vals):
    # initializing some of the model params to equilibrium values?
    ktr = param_vals[0]
    B = param_vals[3]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B*kwbc/ktr)
    model.setValue('Xtr', B*kwbc/ktr)
    model.setValue('Xwbc', B)
    model.setValue('BM_eq', 4*B*kwbc/ktr)
    return model

def initialization_fn_m2d_with_b0(model, param_vals):
    # initializing some of the model params to equilibrium values?
    ktr = param_vals[0]
    B = param_vals[3]
    B0 = param_vals[5]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B*kwbc/ktr)
    model.setValue('Xtr', B*kwbc/ktr)
    model.setValue('Xwbc', B0)
    model.setValue('BM_eq', 4*B*kwbc/ktr)
    return model

def initialization_fn_m2d_2(model, param_vals):
    # initializing some of the model params to equilibrium values?
    ktr = param_vals[0]
    B = param_vals[3]
    B0 = param_vals[5]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B0*kwbc/ktr)
    model.setValue('Xtr', B0*kwbc/ktr)
    model.setValue('Xwbc', B0)
    model.setValue('BM_eq', 4*B*kwbc/ktr)
    return model


def build_pm_model_m2d(model, wbc, blasts, theta=None, n_samples=None, params_to_fit=None,
        initialization=None, use_b0=False, uniform_prior=False, **params):
    """
    Builds a PyMC model
    """
    if params_to_fit is None:
        params_to_fit = param_names_m2d
    if initialization is None:
        initialization = initialization_fn_m2d_2
    if theta is None:
        model.resetToOrigin()
        default_params = [model.getValue(x) for x in params_to_fit]
        theta = default_params
    print('theta:', theta)
    pytensor_forward_function = generate_forward_function(model, wbc, blasts, n_samples=n_samples,
            params_to_fit=params_to_fit, initialization=initialization)
    wbc_data = wbc[1].to_numpy(dtype=np.float64)
    blast_data = blasts[1].to_numpy(dtype=np.float64)
    print('wbc_data:', wbc_data)
    print('blast_data:', blast_data)
    with pm.Model() as new_patient_model_pm:
        # priors
        if not uniform_prior:
            ktr = pm.TruncatedNormal("ktr", mu=theta[0], sigma=theta[0]/2, lower=0, upper=1, initval=theta[0])
            gam = pm.TruncatedNormal("gam", mu=theta[1], sigma=theta[1]/2, lower=0, upper=1, initval=theta[1])
            slope_ven = pm.TruncatedNormal("slope_ven", mu=theta[2], sigma=theta[2]/2, lower=0, initval=theta[2])
            B = pm.TruncatedNormal("B", mu=theta[3], sigma=theta[3]/2, lower=0, initval=theta[3])
            if use_b0:
                B0 = pm.TruncatedNormal('B0', mu=theta[5], sigma=theta[5]/2, lower=0, initval=theta[5])
            p1 = pm.TruncatedNormal('p1', mu=theta[6], sigma=theta[6]/2, lower=0, initval=theta[6])
            bi = pm.TruncatedNormal('bi', mu=theta[7], sigma=theta[7]/2, lower=0, upper=1, initval=theta[7])
        else:
            ktr = pm.Uniform("ktr", lower=0, upper=1, initval=theta[0])
            gam = pm.Uniform("gam", lower=0, upper=1, initval=theta[1])
            slope_ven = pm.Uniform("slope_ven", lower=0, upper=5, initval=theta[2])
            B = pm.Uniform("B", lower=0, upper=10, initval=theta[3])
            if use_b0:
                B0 = pm.Uniform('B0', lower=0, upper=10, initval=theta[5])
            p1 = pm.Uniform('p1', lower=0, upper=1, initval=theta[6])
            bi = pm.Uniform('bi', lower=0, upper=1, initval=theta[7])
        Xl1 = pm.Uniform('Xl1', lower=0, upper=10, initval=theta[4])
        sigma_wbc = pm.HalfNormal("sigma_wbc", 5)
        sigma_blasts = pm.HalfNormal("sigma_blasts", 5)

        # ODE solution function
        ode_solution = pytensor_forward_function(pm.math.stack([ktr, gam,
                                                               slope_ven,
                                                               B, Xl1, p1, bi]))
        if use_b0:
            ode_solution = pytensor_forward_function(pm.math.stack([ktr, gam,
                                                               slope_ven,
                                                               B, Xl1, B0, p1, bi]))
        
        # split up the blasts and WBCs
        wbc_ode = ode_solution[0,:len(wbc_data)]
        blast_ode = ode_solution[1,:len(blast_data)]

        # likelihood
        pm.Normal("Y_wbc_obs", mu=wbc_ode, sigma=sigma_wbc, observed=wbc_data)
        pm.Normal("Y_blast_obs", mu=blast_ode, sigma=sigma_blasts, observed=blast_data)
    return new_patient_model_pm

##################################

# ### M2e - basically the same as m2c except for removing Xl2. Same parameters and everything.

model_desc_m2e = """
//Aza PK model

in_aza_treatment = 0;
in_ven_treatment = 0;

// WBCs (proliferating, transit, mature)
species $Xprol; species $Xtr; species $Xwbc; 

// simplified drug effect model
// effect
E_ven_wbc := in_ven_treatment*slope_ven_wbc; 
E_aza_wbc := in_aza_treatment*slope_aza_wbc;

// using the M3 model structure from Jost 2019
eps = 1e-8;
wbc_drug := 1 - E_ven_wbc - E_aza_wbc;
F := piecewise(wbc_drug*ktr, wbc_drug < 0, wbc_drug*ktr*((B+eps)/(Xwbc+eps))^gam);
G := ktr/(1 + bi*Xblasts_obs);
D := (Xprol + d*Xl1);
BM_eq = 4*B*kwbc/ktr;
dc_piece := D - BM_eq;
dc := piecewise(0, dc_piece <= 0, d*dc_piece);

Xprol' = Xprol*(F - G - dc);
Xtr' =  G * Xprol - G * Xtr;
Xwbc' = G * Xtr - kwbc * Xwbc;

// leukemic blasts (bone marrow, blood)
species $Xl1;
species $Xblasts; species $Xblasts_obs;

// PD model - leukemic blasts
E_ven_blast := slope_ven_blast * in_ven_treatment;
E_aza_blast := slope_aza_blast * in_aza_treatment;

Xl1' = p1*Xl1*(a1 - Xl1/1 - E_ven_blast - E_aza_blast);

// Xblasts_obs
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

slope_ven_blast = 1.1;//7.94; // unit: L/micromol
slope_aza_blast = 1.1; //;
slope_ven_wbc = 1.1;//7.94; // unit: L/micromol
slope_aza_wbc = 1.1; //;
p1 = 0.1; // leukemic blast proliferation rate
d = 1;
bi = 0.1;

// initial values
Xprol = B*kwbc/ktr;
Xtr = B*kwbc/ktr;
Xwbc = B;
Xl1 = 0.04;
"""
##################################

# ### M2f - M2e but without an Aza treatment component, ven only.

model_desc_m2f = """
//Aza PK model

in_aza_treatment = 0;
in_ven_treatment = 0;

// WBCs (proliferating, transit, mature)
species $Xprol; species $Xtr; species $Xwbc; 

// simplified drug effect model
// effect
E_ven_wbc := in_ven_treatment*slope_ven_wbc; 
E_aza_wbc := in_aza_treatment*slope_aza_wbc;

// using the M3 model structure from Jost 2019
eps = 1e-8;
wbc_drug := 1 - E_ven_wbc - E_aza_wbc;
F := piecewise(wbc_drug*ktr, wbc_drug < 0, wbc_drug*ktr*((B+eps)/(Xwbc+eps))^gam);
G := ktr/(1 + bi*Xblasts_obs);
D := (Xprol + d*Xl1);
BM_eq = 4*B*kwbc/ktr;
dc_piece := D - BM_eq;
dc := piecewise(0, dc_piece <= 0, d*dc_piece);

Xprol' = Xprol*(F - G - dc);
Xtr' =  G * Xprol - G * Xtr;
Xwbc' = G * Xtr - kwbc * Xwbc;

// leukemic blasts (bone marrow, blood)
species $Xl1;
species $Xblasts; species $Xblasts_obs;

// PD model - leukemic blasts
E_ven_blast := slope_ven_blast * in_ven_treatment;
E_aza_blast := slope_aza_blast * in_aza_treatment;

Xl1' = p1*Xl1*(a1 - Xl1/1 - E_ven_blast - E_aza_blast);

// Xblasts_obs
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

slope_ven_blast = 1.1;//7.94; // unit: L/micromol
slope_aza_blast = 1.1; //;
slope_ven_wbc = 1.1;//7.94; // unit: L/micromol
slope_aza_wbc = 1.1; //;
p1 = 0.1; // leukemic blast proliferation rate
d = 1;
bi = 0.1;

// initial values
Xprol = B*kwbc/ktr;
Xtr = B*kwbc/ktr;
Xwbc = B;
Xl1 = 0.04;
"""


param_names_m2f = ['ktr', 'gam', 'slope_ven_blast', 'slope_ven_wbc',  'B', 'Xl1', 'B0', 'p1', 'bi']
param_bounds_m2f = [(0, 1), (0, 1), (0, 5), (0, 5),  (0, 10), (0, 10), (0, 10), (0, 1), (0, 1)]


def initialization_fn_m2f(model, param_vals):
    # initializing some of the model params to equilibrium values?
    ktr = param_vals[0]
    B = param_vals[4]
    B0 = param_vals[6]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B*kwbc/ktr)
    model.setValue('Xtr', B*kwbc/ktr)
    model.setValue('Xwbc', B0)
    model.setValue('BM_eq', 4*B*kwbc/ktr)
    return model

def initialization_fn_m2f_2(model, param_vals):
    # initializing some of the model params to equilibrium values?
    ktr = param_vals[0]
    B = param_vals[4]
    B0 = param_vals[6]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B0*kwbc/ktr)
    model.setValue('Xtr', B0*kwbc/ktr)
    model.setValue('Xwbc', B0)
    model.setValue('BM_eq', 4*B*kwbc/ktr)
    return model


def build_pm_model_m2f(model, wbc, blasts, theta=None, n_samples=None, params_to_fit=None,
        initialization=None, use_b0=False, uniform_prior=False, **params):
    """
    Builds a PyMC model
    """
    if params_to_fit is None:
        params_to_fit = param_names_m2f
    if theta is None:
        model.resetToOrigin()
        default_params = [model.getValue(x) for x in params_to_fit]
        theta = default_params
    print('theta:', theta)
    pytensor_forward_function = generate_forward_function(model, wbc, blasts, n_samples=n_samples,
            params_to_fit=params_to_fit, initialization=initialization)
    wbc_data = wbc[1].to_numpy(dtype=np.float64)
    blast_data = blasts[1].to_numpy(dtype=np.float64)
    print('wbc_data:', wbc_data)
    print('blast_data:', blast_data)
    with pm.Model() as new_patient_model_pm:
        # priors
        if not uniform_prior:
            ktr = pm.TruncatedNormal("ktr", mu=theta[0], sigma=theta[0]/2, lower=0, upper=1, initval=theta[0])
            gam = pm.TruncatedNormal("gam", mu=theta[1], sigma=theta[1]/2, lower=0, upper=1, initval=theta[1])
            slope_ven_blast = pm.TruncatedNormal("slope_ven_blast", mu=theta[2], sigma=theta[2]/2, lower=0, initval=theta[2])
            slope_ven_wbc = pm.TruncatedNormal("slope_ven_wbc", mu=theta[4], sigma=theta[4]/2, lower=0, initval=theta[3])
            B = pm.TruncatedNormal("B", mu=theta[6], sigma=theta[6]/2, lower=0, initval=theta[4])
            if use_b0:
                B0 = pm.TruncatedNormal('B0', mu=theta[8], sigma=theta[8]/2, lower=0, initval=theta[6])
            p1 = pm.TruncatedNormal('p1', mu=theta[9], sigma=theta[9]/2, lower=0, initval=theta[7])
            bi = pm.TruncatedNormal('bi', mu=theta[10], sigma=theta[10]/2, lower=0, upper=1, initval=theta[8])
        else:
            ktr = pm.Uniform("ktr", lower=0, upper=1, initval=theta[0])
            gam = pm.Uniform("gam", lower=0, upper=1, initval=theta[1])
            slope_ven_blast = pm.Uniform("slope_ven_blast", lower=0, upper=5, initval=theta[2])
            slope_ven_wbc = pm.Uniform("slope_ven_wbc", lower=0, upper=5, initval=theta[3])
            B = pm.Uniform("B", lower=0, upper=10, initval=theta[4])
            if use_b0:
                B0 = pm.Uniform('B0', lower=0, upper=10, initval=theta[6])
            p1 = pm.Uniform('p1', lower=0, upper=1, initval=theta[7])
            bi = pm.Uniform('bi', lower=0, upper=1, initval=theta[8])
        Xl1 = pm.Uniform('Xl1', lower=0, upper=10, initval=theta[5])
        sigma_wbc = pm.HalfNormal("sigma_wbc", 5)
        sigma_blasts = pm.HalfNormal("sigma_blasts", 5)

        # ODE solution function
        ode_solution = pytensor_forward_function(pm.math.stack([ktr, gam,
                                                               slope_ven_blast,
                                                               slope_ven_wbc,
                                                               B, Xl1, p1, bi]))
        if use_b0:
            ode_solution = pytensor_forward_function(pm.math.stack([ktr, gam,
                                                               slope_ven_blast,
                                                               slope_ven_wbc,
                                                               B, Xl1, B0, p1, bi]))
        
        # split up the blasts and WBCs
        wbc_ode = ode_solution[0,:len(wbc_data)]
        blast_ode = ode_solution[1,:len(blast_data)]

        # likelihood
        pm.Normal("Y_wbc_obs", mu=wbc_ode, sigma=sigma_wbc, observed=wbc_data)
        pm.Normal("Y_blast_obs", mu=blast_ode, sigma=sigma_blasts, observed=blast_data)
    return new_patient_model_pm



