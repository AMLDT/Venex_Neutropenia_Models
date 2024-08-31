# TODO: implement a thread pool with multiprocessing for systematic comparisons...

import os
import multiprocessing
import traceback

import numpy as np
import pandas as pd
import time

from new_patient_model import new_model_desc_no_leno, build_pm_model_from_dataframes, fit_model_mc, extract_data_from_tables_new,\
        run_model, plot_data, plot_runs, calculate_errors, split_train_test, generate_forward_function,\
        plot_runs_area, param_names_with_initial, initialization_fn_with_initial_2,\
        param_names_with_initial_bounds, split_cycles

from new_patient_additional_model_descs import model_desc_m4a_cytokine_independent,\
        param_names_m4a, param_names_bounds_m4a, initialization_fn_m4a_2, build_pm_model_m4a,\
        model_desc_m4b_direct_inhibition, param_names_m4b, param_names_bounds_m4b, initialization_fn_m4b_2,\
        build_pm_model_m4b, model_desc_m4c_direct_inhibition_independent_blasts,\
        param_names_m4b_wbc_only, initialization_fn_m4b_2_wbc_only, param_names_bounds_m4b_wbc_only,\
        build_pm_model_m4b_wbc_only, model_desc_m4b_wbc_only

from simplified_models import model_desc_m2, build_pm_model_m2, initialization_fn_m2,\
        initialization_fn_m2_2, param_names_m2_with_b0, generate_dosing_component_m2,\
        param_bounds_m2, model_desc_m2b, build_pm_model_m2b, initialization_fn_m2b_2,\
        param_names_m2b, model_desc_m2c, build_pm_model_m2c, initialization_fn_m2c_2,\
        param_names_m2c, param_bounds_m2b, param_bounds_m2c,\
        model_desc_m2b_wbc_only, build_pm_model_m2b_wbc_only, initialization_fn_m2b_2_wbc_only,\
        param_names_m2b_wbc_only, param_bounds_m2b_wbc_only,\
        model_desc_m2d, param_names_m2d, param_bounds_m2d, initialization_fn_m2d_2,\
        build_pm_model_m2d,\
        model_desc_m2e,\
        model_desc_m2f, param_names_m2f, param_bounds_m2f, initialization_fn_m2f_2,\
        build_pm_model_m2f

from hoffmann_like_models import model_desc_hoffmann_like_blast_only,\
        model_desc_hoffmann_like, param_names_hoffmann_like_blast_only,\
        param_bounds_hoffmann_like_blast_only, initialization_fn_hoffmann_like_blast_only,\
        build_pm_model_hoffmann_like_blast_only, param_names_hoffmann_like,\
        param_bounds_hoffmann_like, initialization_fn_hoffmann_like,\
        generate_dosing_component_hoffmann_like, build_pm_model_hoffmann_like

from find_map import fit_model_map, pybobyqa_wrapper

# this contains the key descriptions and functions for each model.
MODELS = {
        'm1b': {
            'param_names': param_names_with_initial,
            'model_desc': new_model_desc_no_leno,
            'initialization_fn': initialization_fn_with_initial_2,
            'build_pm_model': None,
            'param_bounds': param_names_with_initial_bounds + [(0, 10), (0, 10)],
            'dosing_component': None,
            'wbc_only': False
        },
        'm2a': {
            'param_names': param_names_m2_with_b0,
            'model_desc': model_desc_m2,
            'initialization_fn': initialization_fn_m2_2,
            'build_pm_model': build_pm_model_m2,
            'param_bounds': param_bounds_m2 + [(0, 10), (0, 10)],
            'dosing_component': generate_dosing_component_m2,
            'wbc_only': False
        },
        'm2b': {
            'param_names': param_names_m2b,
            'model_desc': model_desc_m2b,
            'initialization_fn': initialization_fn_m2b_2,
            'build_pm_model': build_pm_model_m2b,
            'param_bounds': param_bounds_m2b + [(0, 10), (0, 10)],
            'dosing_component': generate_dosing_component_m2,
            'wbc_only': False
        },
        'm2b_wbc_only': {
            'param_names': param_names_m2b_wbc_only,
            'model_desc': model_desc_m2b_wbc_only,
            'initialization_fn': initialization_fn_m2b_2_wbc_only,
            'build_pm_model': build_pm_model_m2b_wbc_only,
            'param_bounds': param_bounds_m2b_wbc_only + [(0, 10)],
            'dosing_component': generate_dosing_component_m2,
            'wbc_only': True
        },
        'm2b_w': {
            'param_names': param_names_m2b_wbc_only,
            'model_desc': model_desc_m2b_wbc_only,
            'initialization_fn': initialization_fn_m2b_2_wbc_only,
            'build_pm_model': build_pm_model_m2b_wbc_only,
            'param_bounds': param_bounds_m2b_wbc_only + [(0, 10)],
            'dosing_component': generate_dosing_component_m2,
            'wbc_only': True
        },
        'm2c': {
            'param_names': param_names_m2c,
            'model_desc': model_desc_m2c,
            'initialization_fn': initialization_fn_m2c_2,
            'build_pm_model': build_pm_model_m2c,
            'param_bounds': param_bounds_m2c + [(0, 10), (0, 10)],
            'dosing_component': generate_dosing_component_m2,
            'wbc_only': False
        },
        'm2d': {
            'param_names': param_names_m2d,
            'model_desc': model_desc_m2d,
            'initialization_fn': initialization_fn_m2d_2,
            'build_pm_model': build_pm_model_m2d,
            'param_bounds': param_bounds_m2d + [(0, 10), (0, 10)],
            'dosing_component': generate_dosing_component_m2,
            'wbc_only': False
        },
        'm2e': {
            'param_names': param_names_m2b,
            'model_desc': model_desc_m2e,
            'initialization_fn': initialization_fn_m2b_2,
            'build_pm_model': build_pm_model_m2b,
            'param_bounds': param_bounds_m2b + [(0, 10), (0, 10)],
            'dosing_component': generate_dosing_component_m2,
            'wbc_only': False
        },
        'm2f': {
            'param_names': param_names_m2f,
            'model_desc': model_desc_m2f,
            'initialization_fn': initialization_fn_m2f_2,
            'build_pm_model': build_pm_model_m2f,
            'param_bounds': param_bounds_m2f + [(0, 10), (0, 10)],
            'dosing_component': generate_dosing_component_m2,
            'wbc_only': False
        },
        'm4a': {
            'param_names': param_names_m4a,
            'model_desc': model_desc_m4a_cytokine_independent,
            'initialization_fn': initialization_fn_m4a_2,
            'build_pm_model': build_pm_model_m4a,
            'param_bounds': param_names_bounds_m4a + [(0, 10), (0, 10)],
            'dosing_component': None,
            'wbc_only': False
        },
        'm4b': {
            'param_names': param_names_m4b,
            'model_desc': model_desc_m4b_direct_inhibition,
            'initialization_fn': initialization_fn_m4b_2,
            'build_pm_model': build_pm_model_m4b,
            'param_bounds': param_names_bounds_m4b + [(0, 10), (0, 10)],
            'dosing_component': None,
            'wbc_only': False
        },
        'm4b_wbc_only': {
            'param_names': param_names_m4b_wbc_only,
            'model_desc': model_desc_m4b_wbc_only,
            'initialization_fn': initialization_fn_m4b_2_wbc_only,
            'build_pm_model': build_pm_model_m4b_wbc_only,
            'param_bounds': param_names_bounds_m4b_wbc_only + [(0, 10)],
            'dosing_component': None,
            'wbc_only': True
        },
        'm4b_w': {
            'param_names': param_names_m4b_wbc_only,
            'model_desc': model_desc_m4b_wbc_only,
            'initialization_fn': initialization_fn_m4b_2_wbc_only,
            'build_pm_model': build_pm_model_m4b_wbc_only,
            'param_bounds': param_names_bounds_m4b_wbc_only + [(0, 10)],
            'dosing_component': None,
            'wbc_only': True
        },
        'm4c': {
            'param_names': param_names_m4b,
            'model_desc': model_desc_m4c_direct_inhibition_independent_blasts,
            'initialization_fn': initialization_fn_m4b_2,
            'build_pm_model': build_pm_model_m4b,
            'param_bounds': param_names_bounds_m4b + [(0, 10), (0, 10)],
            'dosing_component': None,
            'wbc_only': False
        },
}


def fit_model_results(blood_counts, bm_blasts, cycle_days, patient_id,
        model_name, prior_params=None, uniform_prior=True):
    print('Patient ID:', patient_id)
    model_functions = MODELS[model_name]
    model_desc = model_functions['model_desc']
    build_pm_model = model_functions['build_pm_model']
    initialization_fn = model_functions['initialization_fn']
    param_names = model_functions['param_names']
    param_bounds = model_functions['param_bounds']
    dosing_component = None
    if 'dosing_component' in model_functions:
        dosing_component = model_functions['dosing_component']
    use_blasts = True
    if 'wbc_only' in model_functions:
        use_blasts = not model_functions['wbc_only']
    t0 = time.time()
    cycle_info, leuk_table, blast_table = extract_data_from_tables_new(blood_counts,
                                                                             bm_blasts, cycle_days, patient_id, use_neut=True)
    te_model, pm_model = build_pm_model_from_dataframes(cycle_info, leuk_table, blast_table,
            use_neut=True, use_b0=True,
            model_desc=model_desc,
            build_model_function=build_pm_model,
            initialization=initialization_fn,
            params_to_fit=param_names,
            dosing_component_function=dosing_component,
            uniform_prior=uniform_prior,
            use_initial_guess=True,
            use_blasts=use_blasts,
            use_blasts_interpolation=not use_blasts,
            theta=prior_params
            )
    try:
        map_params = fit_model_map(pm_model, params_to_fit=param_names, maxeval=50000,
                                   method=pybobyqa_wrapper,
                                   bounds=param_bounds,
                                   options={'maxfun':20000},
                                   progressbar=False)
        print('time:', time.time() - t0)
        map_params_dict = {k: v for k, v in zip(param_names, map_params)}
        rmse_leuk, rmse_blasts, results = calculate_errors(te_model, map_params, cycle_info, leuk_table, blast_table,
                                                              use_neut=True,
                                                              error_fn='rmse',
                                                            initialization=initialization_fn,
                                                            params_to_fit=param_names,
                                                            wbc_only=not use_blasts)
        rmse_results = {'blasts_train': rmse_blasts, 'leuk_train': rmse_leuk}
        map_rmse = rmse_results
        print('RMSE results:', rmse_results)
        print('nRMSE_blasts:', rmse_blasts/blast_table.bm_blasts.std(), 'nRMSE_neut:', rmse_leuk/leuk_table.b_neut.std())
    except Exception as e:
        print('Unable to run patient')
        print(e)
        print(traceback.format_exc())
        map_rmse = {'blasts_train': np.inf, 'leuk_train': np.inf}
        map_params_dict = {k: np.nan for k in param_names}
    return map_rmse, map_params_dict


def fit_model_train_test(blood_counts, bm_blasts, cycle_days, patient_id,
        model_name, n_cycles_test=4, n_cycles_train=2,
        prior_params=None, uniform_prior=True):
    print('Patient ID:', patient_id)
    model_functions = MODELS[model_name]
    model_desc = model_functions['model_desc']
    build_pm_model = model_functions['build_pm_model']
    initialization_fn = model_functions['initialization_fn']
    param_names = model_functions['param_names']
    param_bounds = model_functions['param_bounds']
    dosing_component = None
    if 'dosing_component' in model_functions:
        dosing_component = model_functions['dosing_component']
    use_blasts = True
    if 'wbc_only' in model_functions:
        use_blasts = not model_functions['wbc_only']
    t0 = time.time()
    cycle_info, leuk_table, blast_table = extract_data_from_tables_new(blood_counts,
                                                                             bm_blasts, cycle_days, patient_id, use_neut=True)
    # TODO: split cycles into train and test
    leuk_train, leuk_test, blast_train, blast_test, leuk_remainder, blast_remainder = split_cycles(leuk_table, blast_table, cycle_info, n_cycles_train, n_cycles_test)
    te_model, pm_model = build_pm_model_from_dataframes(cycle_info, leuk_train, blast_train,
            use_neut=True, use_b0=True,
            model_desc=model_desc,
            build_model_function=build_pm_model,
            initialization=initialization_fn,
            params_to_fit=param_names,
            dosing_component_function=dosing_component,
            uniform_prior=uniform_prior,
            use_initial_guess=True,
            use_blasts=use_blasts,
            theta=prior_params
            )
    try:
        map_params = fit_model_map(pm_model, params_to_fit=param_names, maxeval=50000,
                                   method=pybobyqa_wrapper,
                                   bounds=param_bounds,
                                   options={'maxfun':20000},
                                   progressbar=False)
        print('time:', time.time() - t0)
        map_params_dict = {k: v for k, v in zip(param_names, map_params)}
        rmse_leuk_train, rmse_blasts_train, _ = calculate_errors(te_model, map_params, cycle_info, leuk_train, blast_train,
                                                            use_neut=True,
                                                            error_fn='rmse',
                                                            initialization=initialization_fn,
                                                            params_to_fit=param_names,
                                                            wbc_only=not use_blasts)
        rmse_leuk_test, rmse_blasts_test, _ = calculate_errors(te_model, map_params, cycle_info, leuk_test, blast_test,
                                                            use_neut=True,
                                                            error_fn='rmse',
                                                            initialization=initialization_fn,
                                                            params_to_fit=param_names,
                                                            wbc_only=not use_blasts)
        leuk_additional = pd.concat([leuk_test, leuk_remainder])
        blast_additional = pd.concat([blast_test, blast_remainder])
        rmse_leuk_additional, rmse_blasts_additional, _ = calculate_errors(te_model, map_params, cycle_info,
                                                            leuk_additional, blast_additional,
                                                            use_neut=True,
                                                            error_fn='rmse',
                                                            initialization=initialization_fn,
                                                            params_to_fit=param_names,
                                                            wbc_only=not use_blasts)
        rmse_results = {'blasts_train': rmse_blasts_train, 'leuk_train': rmse_leuk_train,
                'blasts_test': rmse_blasts_test, 'leuk_test': rmse_leuk_test,
                'blasts_additional': rmse_blasts_additional, 'leuk_additional': rmse_leuk_additional}
        map_rmse = rmse_results
        print('RMSE results:', rmse_results)
        print('nRMSE_blasts:', rmse_blasts_train/blast_table.bm_blasts.std(), 'nRMSE_neut:', rmse_leuk_train/leuk_table.b_neut.std())
    except Exception as e:
        print('Unable to run patient')
        print(e)
        print(traceback.format_exc())
        map_rmse = {'blasts_train': np.inf, 'leuk_train': np.inf,
                'blasts_test': np.inf, 'leuk_test': np.inf,
                'blasts_additional': np.inf, 'leuk_additional': np.inf}
        map_params_dict = {k: np.nan for k in param_names}
    return map_rmse, map_params_dict



def run_all_patients(model_name, params_filename=None, rmse_filename=None,
        n_threads=None, train_test=False,
        n_cycles_train=4, n_cycles_test=2):
    """
    For a given model, runs all patients using a multiprocessing pool...
    """
    # load data
    blood_counts = pd.read_excel('patient_data_venex/Ven_blood_counts_16042023.xlsx', sheet_name='Blood_counts')
    bm_blasts = pd.read_excel('patient_data_venex/Ven_blood_counts_16042023.xlsx', sheet_name='Bone_marrow_blasts')
    cycle_days = pd.read_excel('patient_data_venex/Ven_blood_counts_16042023.xlsx', sheet_name='Cycle_days')
    patient_ids = np.loadtxt('id_samples_3_cycles.txt', dtype=int)
    if n_threads is None:
        n_threads = os.cpu_count()
    pool = multiprocessing.Pool(n_threads)
    if train_test:
        args = [(blood_counts, bm_blasts, cycle_days, p, model_name, n_cycles_train, n_cycles_test) for p in patient_ids]
        results = pool.starmap(fit_model_train_test, args, int(len(patient_ids)/n_threads))
    else:
        args = [(blood_counts, bm_blasts, cycle_days, p, model_name) for p in patient_ids]
        results = pool.starmap(fit_model_results, args, int(len(patient_ids)/n_threads))
    pool.close()
    pool.terminate()
    rmse_results = {patient_id: x[0] for patient_id, x in zip(patient_ids, results)}
    param_results = {patient_id: x[1] for patient_id, x in zip(patient_ids, results)}
    rmse_data = pd.DataFrame(rmse_results).T
    param_data = pd.DataFrame(param_results).T
    if rmse_filename is None:
        if train_test:
            rmse_filename = f'systematic_comparison_results_simplified_models/{model_name}_train_test_mp_rmse_data.csv'
        else:
            rmse_filename = f'systematic_comparison_results_simplified_models/{model_name}_mp_rmse_data.csv'
    if params_filename is None:
        if train_test:
            params_filename = f'systematic_comparison_results_simplified_models/{model_name}_train_test_mp_param_data.csv'
        else:
            params_filename = f'systematic_comparison_results_simplified_models/{model_name}_mp_param_data.csv'
    rmse_data.to_csv(rmse_filename)
    param_data.to_csv(params_filename)


def run_all_patients_1_vs_rest(model_name, prior_params_filename,
        params_filename=None, rmse_filename=None, n_threads=None,
        uniform_prior=False, train_test=False,
        n_cycles_train=4, n_cycles_test=2):
    """
    Runs all patients using a prior parameter set constructed from all other patients.
    """
    # load data
    blood_counts = pd.read_excel('patient_data_venex/Ven_blood_counts_16042023.xlsx', sheet_name='Blood_counts')
    bm_blasts = pd.read_excel('patient_data_venex/Ven_blood_counts_16042023.xlsx', sheet_name='Bone_marrow_blasts')
    cycle_days = pd.read_excel('patient_data_venex/Ven_blood_counts_16042023.xlsx', sheet_name='Cycle_days')
    patient_ids = np.loadtxt('id_samples_3_cycles.txt', dtype=int)
    # load prior params
    prior_params = pd.read_csv(prior_params_filename, index_col=0)
    prior_params_dict = {}
    # for every patient, create a prior based on all other patients.
    for p in patient_ids:
        other_patient_params = prior_params[prior_params.index != p].mean(0)
        prior_params_dict[p] = other_patient_params.to_list()
    if n_threads is None:
        n_threads = os.cpu_count()
    pool = multiprocessing.Pool(n_threads)
    if train_test:
        args = [(blood_counts, bm_blasts, cycle_days, p, model_name, n_cycles_train, n_cycles_test, prior_params_dict[p], uniform_prior) for p in patient_ids]
        results = pool.starmap(fit_model_train_test, args, int(len(patient_ids)/n_threads))
    else:
        args = [(blood_counts, bm_blasts, cycle_days, p, model_name, prior_params_dict[p], uniform_prior) for p in patient_ids]
        results = pool.starmap(fit_model_results, args, int(len(patient_ids)/n_threads))
    pool.close()
    pool.terminate()
    rmse_results = {patient_id: x[0] for patient_id, x in zip(patient_ids, results)}
    param_results = {patient_id: x[1] for patient_id, x in zip(patient_ids, results)}
    rmse_data = pd.DataFrame(rmse_results).T
    param_data = pd.DataFrame(param_results).T
    if rmse_filename is None:
        if train_test:
            rmse_filename = f'systematic_comparison_results_simplified_models/{model_name}_train_test_mp_priors_rmse_data.csv'
        else:
            rmse_filename = f'systematic_comparison_results_simplified_models/{model_name}_mp_priors_rmse_data.csv'
    if params_filename is None:
        if train_test:
            params_filename = f'systematic_comparison_results_simplified_models/{model_name}_train_test_mp_priors_param_data.csv'
        else:
            params_filename = f'systematic_comparison_results_simplified_models/{model_name}_mp_priors_param_data.csv'
    rmse_data.to_csv(rmse_filename)
    param_data.to_csv(params_filename)

# TODO: priors constructed using linear regression instead of stuff
def run_all_patients_clinical_prior(model_name, prior_params_filename,
        clinical_data_filename='patient_data_venex/ven_responses_052023.txt',
        clinical_features=[],
        params_filename=None, rmse_filename=None, n_threads=None,
        uniform_prior=False, train_test=False,
        n_cycles_train=4, n_cycles_test=2):
    """
    Runs all patients using a prior parameter set constructed from clinical data.

    For each patient, we will construct a linear regression model using all
    other patients to predict each model parameter.
    """
    import statsmodels.api as sm
    # load data
    blood_counts = pd.read_excel('patient_data_venex/Ven_blood_counts_16042023.xlsx', sheet_name='Blood_counts')
    bm_blasts = pd.read_excel('patient_data_venex/Ven_blood_counts_16042023.xlsx', sheet_name='Bone_marrow_blasts')
    cycle_days = pd.read_excel('patient_data_venex/Ven_blood_counts_16042023.xlsx', sheet_name='Cycle_days')
    patient_ids = np.loadtxt('id_samples_3_cycles.txt', dtype=int)
    # load prior params
    prior_params = pd.read_csv(prior_params_filename, index_col=0)
    # load clinical data
    clinical_data = pd.read_csv(clinical_data_filename)
    # TODO: build a model using statsmodels
    prior_params_dict = {}
    # for every patient, create a prior based on all other patients.
    for p in patient_ids:
        other_patient_params = prior_params[prior_params.index != p].mean(0)
        prior_params_dict[p] = other_patient_params.to_list()
    if n_threads is None:
        n_threads = os.cpu_count()
    pool = multiprocessing.Pool(n_threads)
    if train_test:
        args = [(blood_counts, bm_blasts, cycle_days, p, model_name, n_cycles_train, n_cycles_test, prior_params_dict[p], uniform_prior) for p in patient_ids]
        results = pool.starmap(fit_model_train_test, args, int(len(patient_ids)/n_threads))
    else:
        args = [(blood_counts, bm_blasts, cycle_days, p, model_name, prior_params_dict[p], uniform_prior) for p in patient_ids]
        results = pool.starmap(fit_model_results, args, int(len(patient_ids)/n_threads))
    pool.close()
    pool.terminate()
    rmse_results = {patient_id: x[0] for patient_id, x in zip(patient_ids, results)}
    param_results = {patient_id: x[1] for patient_id, x in zip(patient_ids, results)}
    rmse_data = pd.DataFrame(rmse_results).T
    param_data = pd.DataFrame(param_results).T
    if rmse_filename is None:
        if train_test:
            rmse_filename = f'systematic_comparison_results_simplified_models/{model_name}_train_test_mp_priors_rmse_data.csv'
        else:
            rmse_filename = f'systematic_comparison_results_simplified_models/{model_name}_mp_priors_rmse_data.csv'
    if params_filename is None:
        if train_test:
            params_filename = f'systematic_comparison_results_simplified_models/{model_name}_train_test_mp_priors_param_data.csv'
        else:
            params_filename = f'systematic_comparison_results_simplified_models/{model_name}_mp_priors_param_data.csv'
    rmse_data.to_csv(rmse_filename)
    param_data.to_csv(params_filename)




if __name__ == '__main__':
    """
    run_all_patients('m2b', params_filename='systematic_comparison_results_simplified_models/m2b_mp_param_data.csv',
            rmse_filename='systematic_comparison_results_simplified_models/m2b_mp_rmse_data.csv',
            n_threads=8)
    run_all_patients('m2c', params_filename='systematic_comparison_results_simplified_models/m2c_mp_param_data.csv',
            rmse_filename='systematic_comparison_results_simplified_models/m2c_mp_rmse_data.csv',
            n_threads=8)
    run_all_patients('m2d', n_threads=8)
    run_all_patients('m4a', n_threads=8)
    run_all_patients('m4b', n_threads=8)
    run_all_patients('m2b_wbc_only', n_threads=8)
    run_all_patients('m1b', n_threads=8)
    """
    run_all_patients('m4b_wbc_only', n_threads=12)
    run_all_patients('m2b_wbc_only', n_threads=12)
    # train-test
    """
    run_all_patients('m2c', train_test=True, n_cycles_train=1, n_cycles_test=2,
            rmse_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_rmse_data_1_cycle_train.csv',
            param_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_param_data_1_cycle_train.csv')
    run_all_patients('m2c', train_test=True, n_cycles_train=2, n_cycles_test=2,
            rmse_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_rmse_data_2_cycle_train.csv',
            param_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_param_data_2_cycle_train.csv')
    run_all_patients('m2c', train_test=True, n_cycles_train=3, n_cycles_test=2,
            rmse_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_rmse_data_3_cycle_train.csv',
            param_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_param_data_3_cycle_train.csv')
    run_all_patients('m2c', train_test=True, n_cycles_train=4, n_cycles_test=2,
            rmse_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_rmse_data_4_cycle_train.csv',
            param_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_param_data_4_cycle_train.csv')
    run_all_patients('m2c', train_test=True, n_cycles_train=5, n_cycles_test=2,
            rmse_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_rmse_data_5_cycle_train.csv',
            param_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_param_data_5_cycle_train.csv')
    # train-test with priors
    run_all_patients_1_vs_rest('m2c', 'systematic_comparison_results_simplified_models/m2c_mp_param_data.csv', train_test=True, n_cycles_train=1, n_cycles_test=2,
            rmse_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_rmse_data_1_cycle_train.csv',
            param_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_param_data_1_cycle_train.csv')
    run_all_patients_1_vs_rest('m2c', train_test=True, n_cycles_train=2, n_cycles_test=2,
            rmse_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_rmse_data_2_cycle_train.csv',
            param_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_param_data_2_cycle_train.csv')
    run_all_patients_1_vs_rest('m2c', train_test=True, n_cycles_train=3, n_cycles_test=2,
            rmse_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_rmse_data_3_cycle_train.csv',
            param_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_param_data_3_cycle_train.csv')
    run_all_patients_1_vs_rest('m2c', train_test=True, n_cycles_train=4, n_cycles_test=2,
            rmse_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_rmse_data_4_cycle_train.csv',
            param_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_param_data_4_cycle_train.csv')
    run_all_patients_1_vs_rest('m2c', train_test=True, n_cycles_train=5, n_cycles_test=2,
            rmse_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_rmse_data_5_cycle_train.csv',
            param_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_param_data_5_cycle_train.csv')
    """
