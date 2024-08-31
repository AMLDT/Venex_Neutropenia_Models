import numpy as np

import scipy
import pymc as pm
from pytensor.compile.ops import as_op
import pytensor.tensor as pt

from tellurium_model_fitting import generate_objective_function_multiple_times
from new_patient_model import generate_forward_function


#############################
# model m0
# patient model with lenograstim
model_desc_m0 = """
//Aza PK model

species $X_aza_depot; species $X_aza_central; species $X_aza_peripheral; species $X_aza_eff;

// diffexps
X_aza_depot' = -KA_aza*X_aza_depot;
X_aza_central' = KA_aza*X_aza_depot - (CL_aza/V2_aza + k12_aza)*(X_aza_central) + k21_aza*X_aza_peripheral;
X_aza_peripheral' = k12_aza*X_aza_central - k21_aza*X_aza_peripheral;

// mostly for a unit conversion - to get it in units of ng/mL
X_aza_eff := X_aza_central/V2_aza*1000;

// initial values
Dose_aza = 0; // 75
X_aza_depot = Dose_aza;
X_aza_central = 0;
X_aza_peripheral = 0;

KA_aza = 17;
CL_aza = 4589;
V2_aza = 110;
k12_aza = 0;
k21_aza = 0;

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

// lenograstim (depot, transit compartment 1, transit compartment 2, exogenous G-CSF, infusion)
species $XD; species $Xexo1; species $Xexo2; species $Xg; species $ul;

// PK model for lenograstim: XD = depot lenograstim, Xexo1 and Xexo2 are transit compartments, Xg = external G-CSF
kin :=  (keg + kANC * B) * Bg + ka2 * Xexo1;
kout := (keg + kANC * (Xwbc + Xl2));

XD' = -ka1 * XD + ul * 1000 / (Vg * durl);
Xexo1' = ka1 * XD - ka2 * Xexo1;
Xexo2' = ka2 * Xexo1 - ka2 * Xexo2;
Xg' = kin - kout * Xg;


// WBCs (proliferating, transit, mature)
species $Xprol; species $Xtr; species $Xwbc; 

// PD model - WBCs
// this implies that WBCs are killed at the same rate as blasts...
E := slope * ln(1 + X_ven_central); // standard kinetic model...
E_aza := slope_aza * ln(1 + X_aza_central);

Xprol' = -(Xg/Bg)^beta * ktr * Xprol + (Xg/Bg)^gam * ktr * (1 - E - E_aza) * Xprol;
Xtr' = (Xg/Bg)^beta * ktr * Xprol - (Xg/Bg)^beta * ktr * Xtr;
Xwbc' = (Xg/Bg)^beta * ktr * Xtr - kwbc * Xwbc;

// leukemic blasts (bone marrow, blood)
species $Xl1; species $Xl2;
species $Xblasts; species $Xblasts_obs;

// PD model - leukemic blasts
klc := 1/(1 + c1*Xwbc + c2*Xl2);

Xl1' = (2*a1*klc - 1) * p1 * Xl1 - p1 * E * E_aza * Xl1;
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
Bg = 24.4;
keg = 0.592*24;
kANC = 5.64*24;
beta = 0.234;
// source is https://www.nature.com/articles/s41598-018-21115-4
a1 = 0.875;
d2 = 2.3;
c1 = 0.01;
c2 = 0.01;

// PK model of lenograstim
ka2 = 10*ka1/3;
Vg = 14.5;
durl = 0.0007;
BSA = 1.8; // between 1.61 and 2.07

// parameters to fit - default values can be found in table 1 of the original paper.
ka1 = 3.15; // absorption rate of lenograstim
ktr = 0.236; // transition rate - unit: 1/day
gam = 0.651; // feedback of G-CSF on WBC proliferation - MAKE SURE THIS VARIABLE IS NAMED gam
// CANNOT name this variable "gamma" bc that's a reserved function
B = 4.67; // unit: G/L
slope = 7;//7.94; // unit: L/micromol
slope_aza = 4;// 5;
p1 = 0.1; // leukemic blast proliferation rate

// initial values
Xexo1 = 0;
Xexo2 = 0;
Xprol = B*kwbc/ktr;
Xtr = B;
Xwbc = B;
Xl1 = 0.04;
"""

param_names_m0 = ['ka1', 'ktr', 'gam', 'slope_ven_blast', 'slope_aza_blast', 'slope_ven_wbc', 'slope_aza_wbc', 'B', 'Xl1']

##########
# model m1
# model without lenograstim - the default model...
model_desc_m1 = """
//Aza PK model

species $X_aza_depot; species $X_aza_central; species $X_aza_peripheral; species $X_aza_eff;

// diffexps
X_aza_depot' = -KA_aza*X_aza_depot;
X_aza_central' = KA_aza*X_aza_depot - (CL_aza/V2_aza + k12_aza)*(X_aza_central) + k21_aza*X_aza_peripheral;
X_aza_peripheral' = k12_aza*X_aza_central - k21_aza*X_aza_peripheral;

// mostly for a unit conversion - to get it in units of ng/mL
X_aza_eff := X_aza_central/V2_aza*1000;

// initial values
Dose_aza = 0; // 75
X_aza_depot = Dose_aza;
X_aza_central = 0;
X_aza_peripheral = 0;

KA_aza = 17;
CL_aza = 4589;
V2_aza = 110;
k12_aza = 0;
k21_aza = 0;

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

Xprol' = - G * Xprol + F * Xprol;
Xtr' =  G * Xprol - G * Xtr;
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

param_names_m1 = ['ktr', 'gam', 'slope_ven_blast', 'slope_aza_blast', 'slope_ven_wbc', 'slope_aza_wbc', 'B', 'Xl1']
# fitting the leukemic blast proliferation rate as a parameter
param_names_m1_with_prolif = ['ktr', 'gam', 'slope_ven_blast', 'slope_aza_blast', 'slope_ven_wbc', 'slope_aza_wbc', 'p1', 'B', 'Xl1']

def initialization_fn_m1(model, param_vals):
    # initializing some of the model params to equilibrium values?
    ktr = param_vals[0]
    B = param_vals[6]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B*kwbc/ktr)
    model.setValue('Xtr', B*kwbc/ktr)
    model.setValue('Xwbc', B)
    return model

##########
# model m1_wbc_only
# model with only a wbc component and no blast component...
model_desc_m1_wbc_only = """
//Aza PK model

species $X_aza_depot; species $X_aza_central; species $X_aza_peripheral; species $X_aza_eff;

// diffexps
X_aza_depot' = -KA_aza*X_aza_depot;
X_aza_central' = KA_aza*X_aza_depot - (CL_aza/V2_aza + k12_aza)*(X_aza_central) + k21_aza*X_aza_peripheral;
X_aza_peripheral' = k12_aza*X_aza_central - k21_aza*X_aza_peripheral;

// mostly for a unit conversion - to get it in units of ng/mL
X_aza_eff := X_aza_central/V2_aza*1000;

// initial values
Dose_aza = 0; // 75
X_aza_depot = Dose_aza;
X_aza_central = 0;
X_aza_peripheral = 0;

KA_aza = 17;
CL_aza = 4589;
V2_aza = 110;
k12_aza = 0;
k21_aza = 0;

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

// lmao a hack to deal with numerical instabilities
at Xwbc < 1e-10: Xwbc = 1e-10;

Xprol' = - G * Xprol + F * Xprol;
Xtr' =  G * Xprol - G * Xtr;
Xwbc' = G * Xtr - kwbc * Xwbc;

// leukemic blasts (bone marrow, blood)
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
slope_ven_wbc = 0.15;//7.94; // unit: L/micromol
slope_aza_wbc = 0.5; //;
p1 = 0.1; // leukemic blast proliferation rate

// initial values
Xprol = B*kwbc/ktr;
Xtr = B*kwbc/ktr;
Xwbc = B;
"""

param_names_m1_wbc_only = ['ktr', 'gam', 'slope_ven_wbc', 'slope_aza_wbc', 'B', 'B0']

def initialization_fn_m1_wbc_only(model, param_vals):
    # initializing some of the model params to equilibrium values?
    ktr = param_vals[0]
    B = param_vals[4]
    B0 = param_vals[5]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B*kwbc/ktr)
    model.setValue('Xtr', B*kwbc/ktr)
    model.setValue('Xwbc', B0)
    return model

def generate_residual_function_wbc_only(model, wbc, n_samples=None, params_to_fit=None, initialization=None):
    """
    Generates a residual function using leukocyte and blast data (assuming that the cycle info is already incorporated into the model)
    """
    if initialization is None:
        initialization = initialization_fn_m1_wbc_only
    if params_to_fit is None:
        params_to_fit = param_names_m1_wbc_only
    # n_samples should be max time * 10?
    if n_samples is None:
        n_samples = wbc.iloc[:,0].max()*20
        print('n_samples:', n_samples)
    resid_function = generate_objective_function_multiple_times(model,
                                    {'Xwbc': wbc.iloc[:,1].to_numpy()},
                                    {'Xwbc': wbc.iloc[:,0].to_numpy()},
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

def generate_forward_function_wbc_only(model, wbc, n_samples=None, params_to_fit=None, initialization=None):
    ode_out = generate_residual_function_wbc_only(model, wbc, n_samples=n_samples, params_to_fit=params_to_fit,
            initialization=initialization)
    @as_op(itypes=[pt.dvector], otypes=[pt.dvector])
    def pytensor_forward_model(theta):
        output = ode_out(theta)
        wbc = np.array(output['Xwbc'])
        return wbc
    return pytensor_forward_model


def build_pm_model_m1_wbc_only(model, wbc, theta=None, n_samples=None, params_to_fit=None, initialization=None, **params):
    """
    Builds a PyMC model
    """
    if params_to_fit is None:
        params_to_fit = param_names_m1_wbc_only
    if theta is None:
        model.resetToOrigin()
        default_params = [model.getValue(x) for x in params_to_fit]
        theta = default_params
    pytensor_forward_function = generate_forward_function_wbc_only(model, wbc, n_samples=n_samples, params_to_fit=params_to_fit, initialization=initialization)
    wbc_data = wbc[1].to_numpy(dtype=np.float64)
    print('theta:', theta)
    print('wbc_data:', wbc_data)
    with pm.Model() as new_patient_model_pm:
        # priors
        # TODO: uniform priors for slope?
        ktr = pm.TruncatedNormal("ktr", mu=theta[0], sigma=theta[0]/2, lower=0, initval=theta[0])
        gam = pm.TruncatedNormal("gam", mu=theta[1], sigma=theta[1]/2, lower=0, initval=theta[1])
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

def build_pm_model_m1_wbc_only_uniform_prior(model, wbc, theta=None, n_samples=None, params_to_fit=None,
        initialization=None, **params):
    """
    Builds a PyMC model
    """
    if params_to_fit is None:
        params_to_fit = param_names_m1_wbc_only
    if theta is None:
        model.resetToOrigin()
        default_params = [model.getValue(x) for x in params_to_fit]
        theta = default_params
    pytensor_forward_function = generate_forward_function_wbc_only(model, wbc, n_samples=n_samples, params_to_fit=params_to_fit, initialization=initialization)
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



#############################
# model m1c - M1 patterns for WBCs, but a logistic growth model for blasts, with blast growth being a parameter.
model_desc_m1c_logistic_growth = """
//Aza PK model

species $X_aza_depot; species $X_aza_central;  species $X_aza_eff;

// diffexps
X_aza_depot' = -KA_aza*X_aza_depot;
X_aza_central' = KA_aza*X_aza_depot - (CL_aza/V2_aza)*(X_aza_central);

// mostly for a unit conversion - to get it in units of ng/mL
X_aza_eff := X_aza_central/V2_aza*1000;

// initial values
Dose_aza = 0; // 75
X_aza_depot = Dose_aza;
X_aza_central = 0;

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
// TODO: change this to avoid the weirdly decreasing oscillations? During drug treatment the B/Xwbc feedback should stop...
wbc_drug := 1 - E_ven_wbc - E_aza_wbc;
// TODO: eps = 1e-2, 1e-4
eps = 1e-4;
F := piecewise(wbc_drug*ktr, wbc_drug < 0, wbc_drug*ktr*((B+eps)/(Xwbc+eps))^gam);
G := ktr;
D := (Xprol + d*Xl1);
BM_eq = 4*B*kwbc/ktr; // somewhat arbitrarily set maximum cellularity to 4 times the equilibrium number of proliferating cells...

Xprol' = Xprol*(F - G);
Xtr' =  G * Xprol - G * Xtr;
Xwbc' = G * Xtr - kwbc * Xwbc;

// leukemic blasts (bone marrow, blood)
species $Xl1; species $Xl2;
species $Xblasts; species $Xblasts_obs;

// PD model - leukemic blasts
E_ven_blast := slope_ven_blast * ln(1 + X_ven_central);
E_aza_blast := slope_aza_blast * ln(1 + X_aza_central);

// TODO: this is cytokine-dependent; we should make this cytokine-independent.
//klc := 1/(1 + c1*Xwbc + c2*Xl2);

// TODO: this should have logistic growth with a carrying capacity.
Xl1' = Xl1*(2*p1*a1 - p1 - p1*(E_ven_blast+E_aza_blast));
Xl2' = 2*(1 - a1) * p1 * Xl1 - d2 * Xl2;

// cellularity
CR = 0.5;
// DB is the "approximate maximal tumor burden"
DB = 10^12;
// Xblasts is the observed blast concentration.
//Xblasts := 100*(Xl1 + 0.005*Xtr)/(CR*DB);
//Xblasts0 = 100*(Xl1 + 0.005*Xtr)/(CR*DB);
// okay... I don't know what to do exactly with Xl1.
Xblasts_obs := 100*(Xl1/(Xl1 + BM_eq/4));

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
d = 1; // competition for space in the bone marrow compartment

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

param_names_m1c = ['ktr', 'gam', 'slope_ven_blast', 'slope_aza_blast', 'slope_ven_wbc', 'slope_aza_wbc', 'B', 'Xl1', 'B0', 'p1']

def initialization_fn_m1c(model, param_vals):
    "Initial WBC is a free parameter"
    ktr = param_vals[0]
    B = param_vals[6]
    B0 = param_vals[8]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B*kwbc/ktr)
    model.setValue('Xtr', B*kwbc/ktr)
    model.setValue('Xwbc', B0)
    model.setValue('BM_eq', 4*B*kwbc/ktr)
    return model

def initialization_fn_m1c_2(model, param_vals):
    "Initial WBC is a free parameter"
    ktr = param_vals[0]
    B = param_vals[6]
    B0 = param_vals[8]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B0*kwbc/ktr)
    model.setValue('Xtr', B0*kwbc/ktr)
    model.setValue('Xwbc', B0)
    model.setValue('BM_eq', 4*B*kwbc/ktr)
    return model


def build_pm_model_m1c(model, wbc, blasts, theta=None, n_samples=None, params_to_fit=None,
        initialization=None, use_b0=True, **params):
    """
    Builds a PyMC model
    """
    if params_to_fit is None:
        params_to_fit = param_names_m1c
    if theta is None:
        model.resetToOrigin()
        default_params = [model.getValue(x) for x in params_to_fit]
        theta = default_params
    pytensor_forward_function = generate_forward_function(model, wbc, blasts, n_samples=n_samples,
            params_to_fit=params_to_fit, initialization=initialization)
    wbc_data = wbc[1].to_numpy(dtype=np.float64)
    blast_data = blasts[1].to_numpy(dtype=np.float64)
    print('wbc_data:', wbc_data)
    print('blast_data:', blast_data)
    with pm.Model() as new_patient_model_pm:
        # priors
        # TODO: uniform priors for slope?
        ktr = pm.TruncatedNormal("ktr", mu=theta[0], sigma=theta[0]/2, lower=0, upper=1, initval=theta[0])
        gam = pm.TruncatedNormal("gam", mu=theta[1], sigma=theta[1]/2, lower=0, upper=1, initval=theta[1])
        slope_ven_blast = pm.TruncatedNormal("slope_ven_blast", mu=theta[2], sigma=theta[2]/2, lower=0, initval=theta[2])
        slope_aza_blast = pm.TruncatedNormal("slope_aza_blast", mu=theta[3], sigma=theta[3]/2, lower=0, initval=theta[3])
        slope_ven_wbc = pm.TruncatedNormal("slope_ven_wbc", mu=theta[4], sigma=theta[4]/2, lower=0, initval=theta[4])
        slope_aza_wbc = pm.TruncatedNormal("slope_aza_wbc", mu=theta[5], sigma=theta[5]/2, lower=0, initval=theta[5])
        B = pm.TruncatedNormal("B", mu=theta[6], sigma=theta[6]/2, lower=0, initval=theta[6])
        B0 = pm.TruncatedNormal('B0', mu=theta[8], sigma=theta[8]/2, lower=0, initval=theta[8])
        # Prior for initial blasts should go between 0.0 and 1
        Xl1 = pm.Uniform('Xl1', lower=0, upper=10, initval=theta[7])
        p1 = pm.TruncatedNormal('p1', mu=theta[9], sigma=theta[9]/2, lower=0, initval=theta[9])
        sigma_wbc = pm.HalfNormal("sigma_wbc", 5)
        sigma_blasts = pm.HalfNormal("sigma_blasts", 5)

        # ODE solution function
        ode_solution = pytensor_forward_function(pm.math.stack([ktr, gam,
                                                               slope_ven_blast,
                                                               slope_aza_blast,
                                                               slope_ven_wbc,
                                                               slope_aza_wbc, B, Xl1, B0, p1]))
        
        # split up the blasts and WBCs
        wbc_ode = ode_solution[0,:len(wbc_data)]
        blast_ode = ode_solution[1,:len(blast_data)]

        # likelihood
        pm.Normal("Y_wbc_obs", mu=wbc_ode, sigma=sigma_wbc, observed=wbc_data)
        pm.Normal("Y_blast_obs", mu=blast_ode, sigma=sigma_blasts, observed=blast_data)
    return new_patient_model_pm





######################
# m2 - model with simplified drug treatment compartments

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

// CANNOT name this variable "gamma" bc that's a reserved function
slope_ven_blast = 1.1;//7.94; // unit: L/micromol
slope_aza_blast = 1.1; //;
slope_ven_wbc = 0.9;//7.94; // unit: L/micromol
slope_aza_wbc = 0.9; //;
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


def generate_dosing_component_m2(venex_cycles, aza_cycles):
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
        #dose = row.dose
        output_str += f'at time >= {start} : in_ven_treatment = 1;\n'
        output_str += f'at time >= {end} : in_ven_treatment = 0;\n'
    output_str += '\n'
    for _, row in aza_cycles.iterrows():
        start = row.start
        end = row.end
        #dose = row.dose
        output_str += f'at time >= {start} : in_aza_treatment = 1;\n'
        output_str += f'at time >= {end} : in_aza_treatment = 0;\n'
    return output_str

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


##############
# model m3
# new model with a "resistant" compartment of blasts.
new_model_desc_resistance = """
//Aza PK model

species $X_aza_depot; species $X_aza_central; species $X_aza_peripheral; species $X_aza_eff;

// diffexps
X_aza_depot' = -KA_aza*X_aza_depot;
X_aza_central' = KA_aza*X_aza_depot - (CL_aza/V2_aza + k12_aza)*(X_aza_central) + k21_aza*X_aza_peripheral;
X_aza_peripheral' = k12_aza*X_aza_central - k21_aza*X_aza_peripheral;

// mostly for a unit conversion - to get it in units of ng/mL
X_aza_eff := X_aza_central/V2_aza*1000;

// initial values
Dose_aza = 0; // 75
X_aza_depot = Dose_aza;
X_aza_central = 0;
X_aza_peripheral = 0;

KA_aza = 17;
CL_aza = 4589;
V2_aza = 110;
k12_aza = 0;
k21_aza = 0;

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

Xprol' = - G * Xprol + F * Xprol;
Xtr' =  G * Xprol - G * Xtr;
Xwbc' = G * Xtr - kwbc * Xwbc;

// leukemic blasts (bone marrow, blood)
// Xl1_r is Xl1_resistant
species $Xl1; species $Xl2; speices $Xl1_r;
species $Xblasts; species $Xblasts_obs;

// PD model - leukemic blasts
E_ven_blast := slope_ven_blast * ln(1 + X_ven_central);
E_aza_blast := slope_aza_blast * ln(1 + X_aza_central);

klc := 1/(1 + c1*Xwbc + c2*Xl2);

Xl1' = (2*a1*klc - 1) * p1 * Xl1 - p1 * (E_ven_blast + E_aza_blast) * Xl1;
Xl2' = 2*(1 - a1*klc) * p1 * Xl1 - d2 * Xl2;

// kl_vr = conversion rate from vulnerable to resistant
// kl_rv = conversion rate from resistant to vulnerable
// growth and disposal rate of resistant blasts: should it be the same as vulnerable blasts? or is it 0?
// can we take some ideas from the Hoffmann model?

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


#########################
# model m4
# model with an inhibitory effect of blasts upon wbcs.

################# M4 - cytokine-independent model
model_desc_m4a_cytokine_independent = """
//Aza PK model

species $X_aza_depot; species $X_aza_central;  species $X_aza_eff;

// diffexps
X_aza_depot' = -KA_aza*X_aza_depot;
X_aza_central' = KA_aza*X_aza_depot - (CL_aza/V2_aza)*(X_aza_central);

// mostly for a unit conversion - to get it in units of ng/mL
X_aza_eff := X_aza_central/V2_aza*1000;

// initial values
Dose_aza = 0; // 75
X_aza_depot = Dose_aza;
X_aza_central = 0;

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
// TODO: change this to avoid the weirdly decreasing oscillations? During drug treatment the B/Xwbc feedback should stop...
wbc_drug := 1 - E_ven_wbc - E_aza_wbc;
// TODO: eps = 1e-2, 1e-4
eps = 1e-4;
F := piecewise(wbc_drug*ktr, wbc_drug < 0, wbc_drug*ktr*((B+eps)/(Xwbc+eps))^gam);
G := ktr;
D := (Xprol + d*Xl1);
BM_eq = 4*B*kwbc/ktr; // somewhat arbitrarily set maximum cellularity to 4 times the equilibrium number of proliferating cells...
dc_piece := D - BM_eq;
dc := piecewise(0, dc_piece <= 0, d*dc_piece);

Xprol' = Xprol*(F - G - dc);
Xtr' =  G * Xprol - G * Xtr;
Xwbc' = G * Xtr - kwbc * Xwbc;

// leukemic blasts (bone marrow, blood)
species $Xl1; species $Xl2;
species $Xblasts; species $Xblasts_obs;

// PD model - leukemic blasts
E_ven_blast := slope_ven_blast * ln(1 + X_ven_central);
E_aza_blast := slope_aza_blast * ln(1 + X_aza_central);

// TODO: this is cytokine-dependent; we should make this cytokine-independent.
//klc := 1/(1 + c1*Xwbc + c2*Xl2);

Xl1' = Xl1*(2*p1*a1 - p1 - p1*(E_ven_blast+E_aza_blast) - dc);
Xl2' = 2*(1 - a1) * p1 * Xl1 - d2 * Xl2;

// cellularity
CR = 0.5;
// DB is the "approximate maximal tumor burden"
DB = 10^12;
// Xblasts is the observed blast concentration.
//Xblasts := 100*(Xl1 + 0.005*Xtr)/(CR*DB);
//Xblasts0 = 100*(Xl1 + 0.005*Xtr)/(CR*DB);
// okay... I don't know what to do exactly with Xl1.
Xblasts_obs := 100*(Xl1/(Xl1 + BM_eq/4));

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
d = 1; // competition for space in the bone marrow compartment

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

param_names_m4a = ['ktr', 'gam', 'slope_ven_blast', 'slope_aza_blast', 'slope_ven_wbc', 'slope_aza_wbc', 'B', 'Xl1', 'B0', 'p1']
param_names_bounds_m4a = [(0, 1), (0, 1), (0, 2), (0, 2), (0, 2), (0, 2), (0, 10), (0, 10), (0, 10), (0, 1)]

def initialization_fn_m4a(model, param_vals):
    "Initial WBC is a free parameter"
    ktr = param_vals[0]
    B = param_vals[6]
    B0 = param_vals[8]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B*kwbc/ktr)
    model.setValue('Xtr', B*kwbc/ktr)
    model.setValue('Xwbc', B0)
    model.setValue('BM_eq', 4*B*kwbc/ktr)
    return model

def initialization_fn_m4a_2(model, param_vals):
    "Initial WBC is a free parameter"
    ktr = param_vals[0]
    B = param_vals[6]
    B0 = param_vals[8]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B0*kwbc/ktr)
    model.setValue('Xtr', B0*kwbc/ktr)
    model.setValue('Xwbc', B0)
    model.setValue('BM_eq', 4*B*kwbc/ktr)
    return model


def build_pm_model_m4a(model, wbc, blasts, theta=None, n_samples=None, params_to_fit=None,
        initialization=None, use_b0=True, uniform_prior=False, **params):
    """
    Builds a PyMC model
    """
    if params_to_fit is None:
        params_to_fit = param_names_m4a
    if theta is None:
        model.resetToOrigin()
        default_params = [model.getValue(x) for x in params_to_fit]
        theta = default_params
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
            B0 = pm.TruncatedNormal('B0', mu=theta[8], sigma=theta[8]/2, lower=0, initval=theta[8])
            p1 = pm.TruncatedNormal('p1', mu=theta[9], sigma=theta[9]/2, lower=0, initval=theta[9])
        else:
            ktr = pm.Uniform("ktr", lower=0, upper=1, initval=theta[0])
            gam = pm.Uniform("gam", lower=0, upper=1, initval=theta[1])
            slope_ven_blast = pm.Uniform("slope_ven_blast", lower=0, upper=2, initval=theta[2])
            slope_aza_blast = pm.Uniform("slope_aza_blast", lower=0, upper=2, initval=theta[3])
            slope_ven_wbc = pm.Uniform("slope_ven_wbc", lower=0, upper=2, initval=theta[4])
            slope_aza_wbc = pm.Uniform("slope_aza_wbc", lower=0, upper=2, initval=theta[5])
            B = pm.Uniform("B", lower=0, upper=10, initval=theta[6])
            B0 = pm.Uniform('B0', lower=0, upper=10, initval=theta[8])
            p1 = pm.Uniform('p1', lower=0, upper=1, initval=theta[9])
        # Prior for initial blasts should go between 0.0 and 1
        Xl1 = pm.Uniform('Xl1', lower=0, upper=10, initval=theta[7])
        sigma_wbc = pm.HalfNormal("sigma_wbc", 5)
        sigma_blasts = pm.HalfNormal("sigma_blasts", 5)

        # ODE solution function
        ode_solution = pytensor_forward_function(pm.math.stack([ktr, gam,
                                                               slope_ven_blast,
                                                               slope_aza_blast,
                                                               slope_ven_wbc,
                                                               slope_aza_wbc, B, Xl1, B0, p1]))
        
        # split up the blasts and WBCs
        wbc_ode = ode_solution[0,:len(wbc_data)]
        blast_ode = ode_solution[1,:len(blast_data)]

        # likelihood
        pm.Normal("Y_wbc_obs", mu=wbc_ode, sigma=sigma_wbc, observed=wbc_data)
        pm.Normal("Y_blast_obs", mu=blast_ode, sigma=sigma_blasts, observed=blast_data)
    return new_patient_model_pm

model_desc_m4b_direct_inhibition = """
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
X_aza_central = 0;

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
// TODO: change this to avoid the weirdly decreasing oscillations? During drug treatment the B/Xwbc feedback should stop...
wbc_drug := 1 - E_ven_wbc - E_aza_wbc;
// TODO: eps = 1e-2, 1e-4
eps = 1e-4;
F := piecewise(wbc_drug*G, wbc_drug < 0, wbc_drug*G*((B+eps)/(Xwbc+eps))^gam);
G := ktr/(1+bi*Xblasts_obs);
D := (Xprol + d*Xl1);
BM_eq = 4*B*kwbc/ktr; // somewhat arbitrarily set maximum cellularity to 4 times the equilibrium number of proliferating cells...
dc_piece := D - BM_eq;
dc := piecewise(0, dc_piece <= 0, d*dc_piece);

Xprol' = Xprol*(F - G - dc);
Xtr' =  G * Xprol - G * Xtr;
Xwbc' = G * Xtr - kwbc * Xwbc;

// leukemic blasts (bone marrow, blood)
species $Xl1; species $Xl2;
species $Xblasts; species $Xblasts_obs;

// PD model - leukemic blasts
E_ven_blast := slope_ven_blast * ln(1 + X_ven_central);
E_aza_blast := slope_aza_blast * ln(1 + X_aza_central);

// TODO: this is cytokine-dependent; we should make this cytokine-independent.
//klc := 1/(1 + c1*Xwbc + c2*Xl2);

Xl1' = Xl1*(2*p1*a1 - p1 - p1*(E_ven_blast+E_aza_blast) - dc);
Xl2' = 2*(1 - a1) * p1 * Xl1 - d2 * Xl2;

// cellularity
CR = 0.5;
// DB is the "approximate maximal tumor burden"
DB = 10^12;
// Xblasts is the observed blast concentration.
//Xblasts := 100*(Xl1 + 0.005*Xtr)/(CR*DB);
//Xblasts0 = 100*(Xl1 + 0.005*Xtr)/(CR*DB);
// okay... I don't know what to do exactly with Xl1.
// TODO: should Xblasts_obs be dependent on Xprol? Or should it just be like M4a?
// alternatively we could just have a different model description
Xblasts_obs := 100*(Xl1/(Xl1 + Xprol + 1e-2));
//Xblasts_obs := 100*(Xl1/(Xl1 + BM_eq/4));

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
d = 1; // competition for space in the bone marrow compartment
bi = 0.1;

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

param_names_m4b = ['ktr', 'gam', 'slope_ven_blast', 'slope_aza_blast', 'slope_ven_wbc', 'slope_aza_wbc', 'B', 'Xl1', 'B0', 'p1', 'bi']
param_names_bounds_m4b = [(0, 1), (0, 1), (0, 2), (0, 2), (0, 2), (0, 2), (0, 10), (0, 10), (0, 10), (0, 1), (0, 1)]

def initialization_fn_m4b(model, param_vals):
    "Initial WBC is a free parameter"
    ktr = param_vals[0]
    B = param_vals[6]
    B0 = param_vals[8]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B*kwbc/ktr)
    model.setValue('Xtr', B*kwbc/ktr)
    model.setValue('Xwbc', B0)
    model.setValue('BM_eq', 4*B*kwbc/ktr)
    return model

def initialization_fn_m4b_2(model, param_vals):
    "Initial WBC is a free parameter"
    ktr = param_vals[0]
    B = param_vals[6]
    B0 = param_vals[8]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B0*kwbc/ktr)
    model.setValue('Xtr', B0*kwbc/ktr)
    model.setValue('Xwbc', B0)
    model.setValue('BM_eq', 4*B*kwbc/ktr)
    return model


def build_pm_model_m4b(model, wbc, blasts, theta=None, n_samples=None, params_to_fit=None,
        initialization=None, use_b0=True, uniform_prior=False, **params):
    """
    Builds a PyMC model
    """
    if params_to_fit is None:
        params_to_fit = param_names_m4b
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
            B0 = pm.TruncatedNormal('B0', mu=theta[8], sigma=theta[8]/2, lower=0, initval=theta[8])
            # Prior for initial blasts should go between 0.0 and 1
            p1 = pm.TruncatedNormal('p1', mu=theta[9], sigma=theta[9]/2, lower=0, initval=theta[9])
            bi = pm.TruncatedNormal('bi', mu=theta[10], sigma=theta[10]/2, lower=0, upper=1, initval=theta[10])
        else:
            ktr = pm.Uniform("ktr", lower=0, upper=1, initval=theta[0])
            gam = pm.Uniform("gam", lower=0, upper=1, initval=theta[1])
            slope_ven_blast = pm.Uniform("slope_ven_blast", lower=0, upper=2, initval=theta[2])
            slope_aza_blast = pm.Uniform("slope_aza_blast", lower=0, upper=2, initval=theta[3])
            slope_ven_wbc = pm.Uniform("slope_ven_wbc", lower=0, upper=2, initval=theta[4])
            slope_aza_wbc = pm.Uniform("slope_aza_wbc", lower=0, upper=2, initval=theta[5])
            B = pm.Uniform("B", lower=0, upper=10, initval=theta[6])
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
                                                               slope_aza_wbc, B, Xl1, B0, p1, bi]))
        
        # split up the blasts and WBCs
        wbc_ode = ode_solution[0,:len(wbc_data)]
        blast_ode = ode_solution[1,:len(blast_data)]

        # likelihood
        pm.Normal("Y_wbc_obs", mu=wbc_ode, sigma=sigma_wbc, observed=wbc_data)
        pm.Normal("Y_blast_obs", mu=blast_ode, sigma=sigma_blasts, observed=blast_data)
    return new_patient_model_pm


########################################################################### Model4b - WBC only (using interpolated blasts)

model_desc_m4b_wbc_only = """
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
X_aza_central = 0;

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
// TODO: change this to avoid the weirdly decreasing oscillations? During drug treatment the B/Xwbc feedback should stop...
wbc_drug := 1 - E_ven_wbc - E_aza_wbc;
// TODO: eps = 1e-2, 1e-4
eps = 1e-4;
F := piecewise(wbc_drug*G, wbc_drug < 0, wbc_drug*G*((B+eps)/(Xwbc+eps))^gam);
G := ktr/(1+bi*Xblasts_obs);
D := (Xprol + d*Xl1);
BM_eq = 4*B*kwbc/ktr; // somewhat arbitrarily set maximum cellularity to 4 times the equilibrium number of proliferating cells...
dc_piece := D - BM_eq;
dc := piecewise(0, dc_piece <= 0, d*dc_piece);

Xprol' = Xprol*(F - G - dc);
Xtr' =  G * Xprol - G * Xtr;
Xwbc' = G * Xtr - kwbc * Xwbc;

// leukemic blasts (bone marrow, blood)
species $Xblasts_obs;

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
d = 1; // competition for space in the bone marrow compartment
bi = 0.1;

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

param_names_m4b_wbc_only = ['ktr', 'gam', 'slope_ven_wbc', 'slope_aza_wbc', 'B', 'B0', 'bi']
param_names_bounds_m4b_wbc_only = [(0, 1), (0, 1), (0, 10), (0, 10), (0, 10), (0, 10), (0, 1)]

def initialization_fn_m4b_wbc_only(model, param_vals):
    "Initial WBC is a free parameter"
    ktr = param_vals[0]
    B = param_vals[4]
    B0 = param_vals[5]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B*kwbc/ktr)
    model.setValue('Xtr', B*kwbc/ktr)
    model.setValue('Xwbc', B0)
    model.setValue('BM_eq', 4*B*kwbc/ktr)
    return model

def initialization_fn_m4b_2_wbc_only(model, param_vals):
    "Initial WBC is a free parameter"
    ktr = param_vals[0]
    B = param_vals[4]
    B0 = param_vals[5]
    kwbc = model.getValue('kwbc')
    model.setValue('Xprol', B0*kwbc/ktr)
    model.setValue('Xtr', B0*kwbc/ktr)
    model.setValue('Xwbc', B0)
    model.setValue('BM_eq', 4*B*kwbc/ktr)
    return model


def generate_blast_interpolation(blast_table):
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
                    poly_i += f'{c[j][i]}*(time-{start_time})^{3-j} + '
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
    print(output)
    return output


def build_pm_model_m4b_wbc_only(model, wbc, theta=None, n_samples=None, params_to_fit=None,
        initialization=None, use_b0=True, uniform_prior=False, **params):
    """
    Builds a PyMC model
    """
    if params_to_fit is None:
        params_to_fit = param_names_m4b_wbc_only
    if theta is None:
        model.resetToOrigin()
        default_params = [model.getValue(x) for x in params_to_fit]
        theta = default_params
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
            B0 = pm.TruncatedNormal('B0', mu=theta[5], sigma=theta[5]/2, lower=0, initval=theta[5])
            bi = pm.TruncatedNormal('bi', mu=theta[6], sigma=theta[6]/2, lower=0, upper=1, initval=theta[6])
        else:
            ktr = pm.Uniform("ktr", lower=0, upper=1, initval=theta[0])
            gam = pm.Uniform("gam", lower=0, upper=1, initval=theta[1])
            slope_ven_wbc = pm.Uniform("slope_ven_wbc", lower=0, upper=2, initval=theta[2])
            slope_aza_wbc = pm.Uniform("slope_aza_wbc", lower=0, upper=2, initval=theta[3])
            B = pm.Uniform("B", lower=0, upper=10, initval=theta[4])
            B0 = pm.Uniform('B0', lower=0, upper=10, initval=theta[5])
            bi = pm.Uniform('bi', lower=0, upper=1, initval=theta[6])
        # Prior for initial blasts should go between 0.0 and 1
        sigma_wbc = pm.HalfNormal("sigma_wbc", 5)

        # ODE solution function
        ode_solution = pytensor_forward_function(pm.math.stack([ktr, gam,
                                                               slope_ven_wbc,
                                                               slope_aza_wbc, B, B0, bi]))
        
        # split up the blasts and WBCs
        wbc_ode = ode_solution

        # likelihood
        pm.Normal("Y_wbc_obs", mu=wbc_ode, sigma=sigma_wbc, observed=wbc_data)
    return new_patient_model_pm

#########################################
# m4c
# Note: the only difference between this model and M4b is that here, the blasts are independent of WBCs.
# The initialization functions are the same as M4b.

model_desc_m4c_direct_inhibition_independent_blasts = """
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
X_aza_central = 0;

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
// TODO: change this to avoid the weirdly decreasing oscillations? During drug treatment the B/Xwbc feedback should stop...
wbc_drug := 1 - E_ven_wbc - E_aza_wbc;
// TODO: eps = 1e-2, 1e-4
eps = 1e-4;
F := piecewise(wbc_drug*G, wbc_drug < 0, wbc_drug*G*((B+eps)/(Xwbc+eps))^gam);
G := ktr/(1+bi*Xblasts_obs);
D := (Xprol + d*Xl1);
BM_eq = 4*B*kwbc/ktr; // somewhat arbitrarily set maximum cellularity to 4 times the equilibrium number of proliferating cells...
dc_piece := D - BM_eq;
dc := piecewise(0, dc_piece <= 0, d*dc_piece);

Xprol' = Xprol*(F - G - dc);
Xtr' =  G * Xprol - G * Xtr;
Xwbc' = G * Xtr - kwbc * Xwbc;

// leukemic blasts (bone marrow, blood)
species $Xl1; species $Xl2;
species $Xblasts; species $Xblasts_obs;

// PD model - leukemic blasts
E_ven_blast := slope_ven_blast * ln(1 + X_ven_central);
E_aza_blast := slope_aza_blast * ln(1 + X_aza_central);

// TODO: this is cytokine-dependent; we should make this cytokine-independent.
//klc := 1/(1 + c1*Xwbc + c2*Xl2);

Xl1' = Xl1*(2*p1*a1 - p1 - p1*(E_ven_blast+E_aza_blast) - dc);
Xl2' = 2*(1 - a1) * p1 * Xl1 - d2 * Xl2;

// cellularity
CR = 0.5;
// DB is the "approximate maximal tumor burden"
DB = 10^12;
// Xblasts is the observed blast concentration.
//Xblasts := 100*(Xl1 + 0.005*Xtr)/(CR*DB);
//Xblasts0 = 100*(Xl1 + 0.005*Xtr)/(CR*DB);
// okay... I don't know what to do exactly with Xl1.
// alternatively we could just have a different model description
//Xblasts_obs := 100*(Xl1/(Xl1 + Xprol + 1e-2));
Xblasts_obs := 100*(Xl1/(Xl1 + BM_eq/4));

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
d = 1; // competition for space in the bone marrow compartment
bi = 0.1;

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


