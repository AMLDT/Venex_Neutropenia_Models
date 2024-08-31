# TODO: some general utilities for fitting tellurium models.
# using scipy's optimization routines - nonlinear least squares

import numpy as np
import scipy

def abs_percent_error(model_val, base_val):
    """
    This is symmetric absolute percentage error
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    """
    return np.abs(model_val - base_val)/(np.abs(model_val) + np.abs(base_val))

def rmse(model_output, model_times, data_vals, data_times, error_fn='rmse'):
    """
    single-dimensional RMSE between the model output and the data.

    Args:
        model_output: 1d np array
        model_times: 1d np array of same shape as model_output, indicating the times at which the model outputs were taken.
        data_vals: 1d np array
        data_times: 1d np array of same shape as data_vals, indicating the times in which the measurements were taken.
        error_fn: str - could be 'rmse' or 'mape' (for mean absolute percentage error)
    """
    start_time = model_times[0]
    end_time = model_times[-1]
    time_scale = (end_time - start_time)/(len(model_times))
    rmse = 0
    model_vals = []
    for j, t in enumerate(data_times):
        index0 = int((t - start_time)/time_scale)
        index1 = index0 + 1
        if index0 <= 0:
            index0 = 0
            index1 = 0
            model_val = model_output[0]
        else:
            current_time = model_times[index1]
            prev_time = model_times[index0]
            factor = (t - prev_time)/(current_time - prev_time)
            model_val = factor*model_output[index1] + (1-factor)*model_output[index0]
        if error_fn == 'rmse':
            rmse += (model_val - data_vals[j])**2
        elif error_fn == 'mape' or error_fn == 'mean absolute percentage error':
            rmse += np.abs(model_val - data_vals[j])/data_vals[j]
        elif error_fn == 'mae' or error_fn == 'mean absolute error':
            rmse += np.abs(model_val - data_vals[j])
        model_vals.append(model_val)
    if error_fn == 'rmse':
        rmse = np.sqrt(rmse/len(data_times))
    elif error_fn == 'nrmse_mean':
        rmse = np.sqrt(rmse/len(data_times))/data_vals.mean()
    elif error_fn == 'nrmse_range':
        rmse = np.sqrt(rmse/len(data_times))/(data_vals.max() - data_vals.mean())
    elif error_fn == 'nrmse_std' or error_fn == 'nrmse_std':
        rmse = np.sqrt(rmse/len(data_times))/data_vals.std()
    elif error_fn == 'corrcoef' or error_fn == 'corr':
        rmse = np.corrcoef(model_vals, data_vals)[0,1]
    else:
        rmse = rmse/len(data_times)
    return rmse


def get_model_results_times(model_output, model_times, data_times):
    "This returns a list of model results that are matched to the times in the data. "
    start_time = model_times[0]
    end_time = model_times[-1]
    time_scale = (end_time - start_time)/(len(model_times))
    model_vals = []
    for _, t in enumerate(data_times):
        index0 = int((t - start_time)/time_scale)
        index1 = index0 + 1
        if index0 <= 0:
            index0 = 0
            index1 = 0
            model_val = model_output[0]
        else:
            current_time = model_times[index1]
            prev_time = model_times[index0]
            factor = (t - prev_time)/(current_time - prev_time)
            model_val = factor*model_output[index1] + (1-factor)*model_output[index0]
        model_vals.append(model_val)
    # TODO
    return model_vals


def generate_objective_function(model, var_vals, times, params_to_fit,
                                use_val_ranges=False, set_initial_vals=True,
                                n_samples=100, var_weights=None,
                                global_param_vals=None,
                                time_range=None,
                                initialization_fn=None,
                                scale_fn=None,
                                error_fn=None,
                                error_combo_fn=None,
                                additional_fn=None,
                                return_values=False,
                                handle_errors=False,
                                print_errors=True):
    """
    Generates an objective function for a Tellurium model with one or more measured/output variables, where all measurements for the variables are taken at the same times.

    Args:
        model: a tellurium model
        var_vals: dict of variable name: numpy array of vals (this is derived from the data, and each array should have the same length as times, and should be sorted by time). Basically this is the training data - these are the output values of the model, and are used to calculate the errors, residuals, etc.
        times: array of times at which the measurements were taken (should be sorted by increasing time)
        params_to_fit (list of str): independent variables - names of parameters that should be fitted
        use_val_ranges: whether to scale the values before calculating error (to adjust for the varying ranges of different variables)
        set_initial_vals: if true, initial conditions are set to the data. if false, initial conditions are treated as variables.
        n_samples: total number of samples for the simulation.
        var_weights: weight assigned to the variables - dict of param_name : number.
        global_param_vals: dict of param_name : value
        time_range: pair of values (start, end) for the simulation time. Default: using the data minimum and maximum
        initialization_fn: a function taking in the model and parameters as an argument (in the same order), and initializes the model in-place.
        scale_fn: a vectorized numpy function on an array (or number), to be called on both the data input and output.
        error_fn: function that takes in the model and ground-truth outputs, and returns an error value. By default, it's just a square.
        error_combo_fn: function that takes a list of errors for each observation, and returns a single number.
        additional_fn: an additional function added to the error after everything is done, such as a regularization term. Takes in the model.
        return_values (bool): whether or not to return a full array of results (default: false - returns the RMSE). If true, this returns the values of the function at the given times.
        handle_errors (bool): whether or not to somehow deal with errors in the model simulation process (CVODE). default: False
    """
    val_ranges = {p: max(val) - min(val) for p, val in var_vals.items()}
    if var_weights is None:
        var_weights = {p: 1.0/len(var_vals) for p in var_vals.keys()}
    if time_range is None:
        time_range = (times.min(), times.max() + 1)
    start_time = time_range[0]
    end_time = time_range[1]
    n_samples = int(n_samples)
    time_scale = (end_time - start_time)/(n_samples)
    if scale_fn is not None:
        var_vals = {k: scale_fn(v) for k, v in var_vals.items()}
    data_vars = list(var_vals.keys())
    variables_to_simulate = ['time'] + data_vars
    def objective(x, *args):
        "x is a 1d array of the variable to optimize"
        x = np.around(x, 8)
        model.resetToOrigin()
        # set global vals
        if global_param_vals:
            for p, v in global_param_vals.items():
                model.setValue(p, v)
        if initialization_fn is not None:
            initialization_fn(model, x)
        # set initial values
        if set_initial_vals:
            for p, vals in var_vals.items():
                model.setValue(p, vals[0])
        for p, v in zip(params_to_fit, x):
            model.setValue(p, v)
        # run simulation
        #print('running simulation')
        # (maybe the number of time points per day could be increased/parameterized?)
        if handle_errors:
            try:
                results = model.simulate(start_time, end_time, n_samples+1, selections=variables_to_simulate)
            except RuntimeError:
                if print_errors:
                    print('Error simulating model at', x)
                results = {'time': np.linspace(start_time, end_time, n_samples+1)}
                for p in var_vals.keys():
                    results[p] = np.zeros(n_samples+1)
        else:
            results = model.simulate(start_time, end_time, n_samples+1, selections=variables_to_simulate)
        # compute errors
        errors = []
        # return the model's simulated values
        model_vals = {}
        result_times = results['time']
        for p, vals in var_vals.items():
            model_var_val = []
            rp = results[p]
            if scale_fn is not None:
                rp = scale_fn(rp)
            for j, t in enumerate(times):
                index0 = int((t - start_time)/time_scale)
                index1 = index0 + 1
                if index0 < 0:
                    index0 = 0
                    index1 = 0
                    model_val = results[p][0]
                else:
                    current_time = result_times[index1]
                    prev_time = result_times[index0]
                    factor = (t - prev_time)/(current_time - prev_time)
                    model_val = factor*rp[index1] + (1-factor)*rp[index0]
                model_var_val.append(model_val)
                base_val = vals[j]
                if use_val_ranges:
                    errors.append(var_weights[p]*((model_val - base_val)/val_ranges[p])**2)
                else:
                    if error_fn is not None:
                        errors.append(error_fn(model_val, base_val))
                    else:
                        errors.append(var_weights[p]*(model_val - base_val)**2)
                # scale the ranges so that they're roughly equal?
            model_vals[p] = model_var_val
        if return_values:
            return model_vals
        #print('errors calculated')
        if error_combo_fn is None:
            rmse = np.sqrt(sum(errors)/len(times))
        else:
            rmse = error_combo_fn(errors)
        if additional_fn is not None:
            rmse += additional_fn(model, x)
        return rmse
    return objective


# this is almost the same as generate_objective_function (see above), 
# different variables have different times in which they're measured.
def generate_objective_function_multiple_times(model, var_vals, times, params_to_fit,
                                use_val_ranges=False, set_initial_vals=True,
                                n_samples=100, var_weights=None,
                                global_param_vals=None,
                                time_range=None,
                                initialization_fn=None,
                                scale_fn=None,
                                error_fn=None,
                                error_combo_fn=None,
                                additional_fn=None,
                                return_values=False,
                                handle_errors=False,
                                print_errors=False):
    """
    Generates an objective function for multiple variables and multiple measurement times.

    Args:
        model: a tellurium model
        var_vals: dict of variable name: numpy array of vals (this is derived from the data, and each array should have the same length as times, and should be sorted by time). Basically this is the training data - these are the output values of the model, and are used to calculate the errors, residuals, etc.
        times: dict of variable name : array of times, in which case the variable names should correspond to the variables in var_vals.
        params_to_fit (list of str): independent variables - names of parameters that should be fitted
        use_val_ranges: whether to scale the values before calculating error (to adjust for the varying ranges of different variables)
        set_initial_vals: if true, initial conditions are set to the data. if false, initial conditions are treated as variables.
        n_samples: total number of samples for the simulation.
        var_weights: weight assigned to the variables - dict of param_name : number.
        global_param_vals: dict of param_name : value
        time_range: pair of values (start, end) for the simulation time. Default: using the data minimum and maximum
        initialization_fn: a function taking in the model and parameters as an argument (in the same order), and initializes the model in-place.
        scale_fn: a vectorized numpy function on an array (or number), to be called on both the data input and output.
        error_fn: function that takes in the model and ground-truth outputs, and returns an error value. By default, it's just a square.
        error_combo_fn: function that takes a list of errors for each observation, and returns a single number.
        additional_fn: an additional function added to the error after everything is done, such as a regularization term. Takes in the model.
        return_values (bool): whether or not to return a full array of results (default: false - returns the RMSE). If true, this returns the values of the function at the given times.
        handle_errors (bool): whether or not to somehow deal with errors in the model simulation process (CVODE). default: False
    """
    val_ranges = {p: max(val) - min(val) for p, val in var_vals.items()}
    if var_weights is None:
        var_weights = {p: 1.0/len(var_vals) for p in var_vals.keys()}
    # single time-range max for everything...???
    if time_range is None:
        min_time = 1e10;
        max_time = -1e10;
        for _, t in times.items():
            tmin = t.min()
            tmax = t.max()
            if tmin < min_time:
                min_time = tmin
            if tmax > max_time:
                max_time = tmax
        min_time = min(min_time, 0)
        time_range = (min_time, max_time + 1)
    start_time = time_range[0]
    end_time = time_range[1]
    n_samples = int(n_samples)
    time_scale = (end_time - start_time)/(n_samples)
    if scale_fn is not None:
        var_vals = {k: scale_fn(v) for k, v in var_vals.items()}
    data_vars = list(var_vals.keys())
    variables_to_simulate = ['time'] + data_vars
    def objective(x, *args):
        "x is a 1d array of the variable to optimize"
        x = np.around(x, 8)
        model.resetToOrigin()
        # set global vals
        if global_param_vals:
            for p, v in global_param_vals.items():
                model.setValue(p, v)
        if initialization_fn is not None:
            initialization_fn(model, x)
        # set initial values
        if set_initial_vals:
            for p, vals in var_vals.items():
                model.setValue(p, vals[0])
        for p, v in zip(params_to_fit, x):
            model.setValue(p, v)
        # run simulation
        #print('running simulation')
        # (maybe the number of time points per day could be increased/parameterized?)
        if handle_errors:
            try:
                results = model.simulate(start_time, end_time, n_samples+1, selections=variables_to_simulate)
            except RuntimeError:
                if print_errors:
                    print('Error simulating model at', x)
                results = {'time': np.linspace(start_time, end_time, n_samples+1)}
                for p in var_vals.keys():
                    results[p] = np.zeros(n_samples+1)
        else:
            results = model.simulate(start_time, end_time, n_samples+1, selections=variables_to_simulate)
        # compute errors
        errors = []
        # return the model's simulated values
        model_vals = {}
        result_times = results['time']
        for p, vals in var_vals.items():
            model_var_val = []
            # getting the times for the measured variable
            var_times = times[p]
            rp = results[p]
            if scale_fn is not None:
                rp = scale_fn(rp)
            for j, t in enumerate(var_times):
                index0 = int((t - start_time)/time_scale)
                index1 = index0 + 1
                if index0 < 0:
                    index0 = 0
                    index1 = 0
                    model_val = results[p][0]
                else:
                    current_time = result_times[index1]
                    prev_time = result_times[index0]
                    factor = (t - prev_time)/(current_time - prev_time)
                    model_val = factor*rp[index1] + (1-factor)*rp[index0]
                model_var_val.append(model_val)
                base_val = vals[j]
                if use_val_ranges:
                    errors.append(var_weights[p]*((model_val - base_val)/val_ranges[p])**2)
                else:
                    if error_fn is not None:
                        errors.append(error_fn(model_val, base_val))
                    else:
                        errors.append(var_weights[p]*(model_val - base_val)**2)
                # scale the ranges so that they're roughly equal?
            model_vals[p] = model_var_val
        if return_values:
            return model_vals
        #print('errors calculated')
        if error_combo_fn is None:
            rmse = np.sqrt(sum(errors)/len(times))
        else:
            rmse = error_combo_fn(errors)
        if additional_fn is not None:
            rmse += additional_fn(model, x)
        return rmse
    return objective


def personalize_model_global(data_subset, model,
                      use_val_ranges,
                      model_vars,
                      data_vars,
                      optim_vars,
                      initial_vals,
                      bounds,
                      var_weights=None,
                      time_column='time',
                      optim_function=scipy.optimize.dual_annealing,
                      **optim_kwargs):
    """
    Uses global nonlinear optimization to try to find the parameters of a model that minimizes the error between the model's output and a set of observations.

    Args:
        data_subset: pandas dataframe
        model: compiled tellurium model
        use_val_ranges:
        model_vars (list): names of the dependent/output variables in the model.
        data_vars (list): column names for the dependent/output variables in data_subset.
        optim_vars (list): names of the (independent) variables in the model that we're optimizing.
        initial_vals (list): initial values of the independent variables (optim_vars).
        bounds (list of tuples): list of lower and upper bounds for the optim_vars.
        var_weights:
        time_column (str): the column of data_subset that represents time.
        optim_function (function): something from scipy.optimize - default: dual_annealing
        **optim_kwargs: additional arguments to optim_function

    Returns:
        Results of running optim_function
    """
    var_vals = {v1: data_subset[v2].to_numpy() for v1, v2 in zip(model_vars, data_vars)}
    objective = generate_objective_function(model,
            var_vals,
            data_subset[time_column].to_numpy(),
            optim_vars,
            use_val_ranges=use_val_ranges,
            var_weights=var_weights)
    optim_results = optim_function(objective, bounds=bounds,
                            x0=initial_vals, **optim_kwargs)
    return optim_results


def personalize_model_multiple_inits(data_subset, model,
                      use_val_ranges,
                      model_vars,
                      data_vars,
                      optim_vars,
                      bounds,
                      n_starts=100,
                      var_weights=None,
                      time_column='time',
                      optim_function=scipy.optimize.least_squares,
                      **optim_kwargs):
    """
    Uses Latin hypercube sampling to generate n_starts different starting points, and runs least squares on each of the starting points,
    returning the optimization results for all initializations.

    Args:
        data_subset: pandas dataframe
        model: compiled tellurium model
        use_val_ranges:
        model_vars (list): names of the dependent/output variables in the model.
        data_vars (list): column names for the dependent/output variables in data_subset.
        optim_vars (list): names of the (independent) variables in the model that we're optimizing.
        bounds (list of length 2): list of lower and upper bounds for the optim_vars.
        n_starts (int): number of initializations. Default: 100
        var_weights:
        time_column (str): the column of data_subset that represents time.
        optim_function (function): something from scipy.optimize - default: least_squares
        **optim_kwargs: additional arguments to optim_function

    Returns:
        Results of running optim_function
    """
    var_vals = {v1: data_subset[v2].to_numpy() for v1, v2 in zip(model_vars, data_vars)}
    objective = generate_objective_function(model,
            var_vals,
            data_subset[time_column].to_numpy(),
            optim_vars,
            use_val_ranges=use_val_ranges,
            var_weights=var_weights)
    return personalize_model_multiple_inits_with_objective(objective, bounds,
            n_starts, optim_function, **optim_kwargs)


def personalize_model_multiple_inits_with_objective(obj, bounds, n_starts,
        optim_function=scipy.optimize.least_squares, print_errors=False,
        **optim_kwargs):
    """
    Given an objective function as generated by generate_objective_function,
    this runs an optimization to try to find the values of the parameters with
    the lowest cost.

    Args:
        obj: objective function
        bounds: two lists, indicating the lower and upper bounds of each of
            the variables.
        n_starts: number of starting points (for the Latin Hypercube sampling)
        optim_function: from scipy.optimize - default is least_squares

    Returns:
        best_x (lowest-cost parameter values),
        best_cost (lowest cost),
        optim_results (list of all optimization results from each run)
    """
    # run latin squares sampling
    sampler = scipy.stats.qmc.LatinHypercube(len(bounds[0]))
    samples = sampler.random(n_starts)
    optim_results = []
    best_cost = np.inf
    best_x = None
    for i in range(n_starts):
        # move samples to bounds
        initial_vals = samples[i, :]
        for j in range(len(bounds[0])):
            b0 = bounds[0][j]
            b1 = bounds[1][j]
            initial_vals[j] = b0 + initial_vals[j]*(b1 - b0)  
        # If there's an error, just skip the current initialization
        try:
            result = optim_function(obj, bounds=bounds,
                                    x0=initial_vals, **optim_kwargs)
            optim_results.append(result)
            if result.fun[0] < best_cost:
                best_cost = result.fun[0]
                best_x = result.x
            #print(i, initial_vals, result.fun, result.x)
        except RuntimeError as e:
            if print_errors:
                print('Error:', e)
                print('Initialization:', initial_vals)
    return best_x, best_cost, optim_results


def run_model(model, time_bounds, params, best_vals, n_samples=500):
    """
    This runs the model with the given data (used only for setting treatment times) and parameters.
    
    Args:
        model - a tellurium model
        time_bounds: (min_time, max_time)
        params: list of parameter names in the model.
        best_vals: list/array of model parameters
    """
    model.resetToOrigin()
    for k, v in zip(params, best_vals[:4]):
        model.setValue(k, v)
    results = model.simulate(time_bounds[0], time_bounds[1], n_samples)
    return results


def plot_model(model, time_bounds, params, output_var_name, all_param_vals=None, data_times=None, data_vals=None):
    """
    Plots the model and fitted trajectories. optim_results is a list of all optimization results, returned by the optimization function.
    """
    import matplotlib.pyplot as plt
    # if optim_results is provided, this plots all of the parameters from all the samples.
    if all_param_vals is not None:
        for r in all_param_vals:
            # very arbitrarily, we're limiting the cost of all the displayed samples.
            best_vals = r
            results_temp, model_params = run_model(model, time_bounds, params, best_vals)
            plt.plot(results_temp['time'], results_temp[output_var_name], linewidth=0.1, color='gray')
    if data_times is not None and data_vals is not None:
        plt.scatter(data_times, data_vals)
    plt.title('Best fitted parameters')
