import numpy as np
import pandas as pd
import xarray as xr
import warnings

def check_and_verify_dimensions(ds, vars, dim):
    """
    Check if the dimensions of the specified variables in the dataset match the provided dimension.
    Raise an exception and stop if any variable does not match the expected dimensions.
    
    Parameters:
    - ds (xr.Dataset): The xarray dataset containing the variables.
    - vars (list of str): List of variable names to check.
    - dim (tuple of str): The expected dimensions.
    
    Returns:
    - None: If all variables match the expected dimensions.
    
    Raises:
    - ValueError: If any variable does not match the expected dimensions.
    """
    if isinstance(dim, list):
        dim = tuple(dim)
    if isinstance(dim, str):
        dim = tuple(dim)
        
    for var in vars:
        if var in ds:
            var_dims = ds[var].dims
            if var_dims != dim:
                raise ValueError(f"Variable {var} dimensions {var_dims} do not match expected dimensions {dim}.")
        else:
            raise ValueError(f"Variable {var}  that is defined in var_extra not found in the dataset.")

    #print("All variables have the expected dimensions.")

def keep_selected_vars(ds, selected_vars):
    # Get the list of variables in the dataset
    all_vars = list(ds.variables)

    # Variables to drop
    vars_to_drop = [var for var in all_vars if var not in selected_vars]

    # Drop the variables
    ds_selected = ds.drop_vars(vars_to_drop)

    return ds_selected

def ObjectiveFunction(obs,
                      sim,
                      info_obs={'var': 'Discharge',
                                'var_id': 'COMID',
                                'dim_id': 'COMID',
                                'var_time': 'time',
                                'dim_time': 'time'},
                      info_sim={'var': 'Discharge',
                                'var_id': 'COMID',
                                'dim_id': 'COMID',
                                'var_time': 'time',
                                'dim_time': 'time'},
                      TimeStep='daily'):
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # check for var_extra and check if their dimension is dim_id
    # observation
    list_var_obs = [info_obs['var'], info_obs['var_id'], info_obs['var_time']]
    if info_obs.get('var_extra') is not None:
        check_and_verify_dimensions(obs, info_obs['var_extra'], info_obs['dim_id'])
        list_var_obs.extend(info_obs['var_extra'])
    # simulation
    list_var_sim = [info_sim['var'], info_sim['var_id'], info_sim['var_time']]
    if info_sim.get('var_extra') is not None:
        check_and_verify_dimensions(sim, info_sim['var_extra'], info_sim['dim_id'])
        list_var_sim.extend(info_sim['var_extra'])
    
    # slice the obs and sim based on varibale, time and ID
    obs = keep_selected_vars(obs, list_var_obs)
    sim = keep_selected_vars(sim, list_var_sim)
    
    # rename var_id to ID and time_var to time
    obs = obs.rename({info_obs['var']: 'obs', info_obs['var_id']: 'ID', info_obs['var_time']: 'time'})
    if info_obs['dim_id'] in obs.dims:
        obs = obs.rename({info_obs['dim_id']: 'ID'})
    if info_obs['dim_time'] in obs.dims:
        obs = obs.rename({info_obs['dim_time']: 'time'})
    
    # rename var_id to ID and time_var to time
    sim = sim.rename({info_sim['var']: 'sim', info_sim['var_id']: 'ID', info_sim['var_time']: 'time'})
    if info_sim['dim_id'] in sim.dims:
        sim = sim.rename({info_sim['dim_id']: 'ID'})
    if info_sim['dim_time'] in sim.dims:
        sim = sim.rename({info_sim['dim_time']: 'time'})
    
    #print(obs)
    if 'ID' not in obs.coords:
        obs = obs.set_coords('ID')
    if 'time' not in obs.coords:
        obs = obs.set_coords('time')
    obs = obs.set_index(ID='ID')
    obs = obs.set_index(time='time')

    #print(sim)
    if 'ID' not in sim.coords:
        sim = sim.set_coords('ID')
    if 'time' not in sim.coords:
        sim = sim.set_coords('time')
    sim = sim.set_index(ID='ID')
    sim = sim.set_index(time='time')
    
    # Round to the daily or hourly time steps
    if TimeStep == 'daily':
        obs['time'] = obs['time'].to_index().floor('d')
        sim['time'] = sim['time'].to_index().floor('d')
    elif TimeStep == 'hourly':
        obs['time'] = obs['time'].to_index().floor('h')
        sim['time'] = sim['time'].to_index().floor('h')

    # Find overlapping time
    common_time = np.intersect1d(obs['time'].values, sim['time'].values)

    # Slice data based on overlapping time
    obs_overlap = obs.sel({'time': common_time})
    sim_overlap = sim.sel({'time': common_time})

    # Get shared IDs
    common_ids = np.intersect1d(obs_overlap['ID'].values, sim_overlap['ID'].values)

    # Slice data based on shared IDs
    obs_overlap = obs_overlap.sel({'ID': common_ids})
    sim_overlap = sim_overlap.sel({'ID': common_ids})

    # Sort data based on IDs
    obs_overlap_sorted = obs_overlap.sortby('ID')
    sim_overlap_sorted = sim_overlap.sortby('ID')
    
    # print
    #print(obs_overlap_sorted)
    #print(sim_overlap_sorted)

    # Create new xarray object to store efficiency values
    ds = xr.Dataset()
    
    # populate the ds and pass the extra var with simulation and observation prefix
    ds ['obs'] = obs_overlap_sorted['obs']
    ds ['sim'] = sim_overlap_sorted['sim']
    ds ['time'] = obs_overlap['time']
    ds ['ID'] = obs_overlap['ID']
    if info_obs.get('var_extra') is not None:
        for var in info_obs.get('var_extra'):
            ds ['obs_'+var] = obs_overlap_sorted[var]
    if info_sim.get('var_extra') is not None:
        for var in info_sim.get('var_extra'):
            ds ['sim_'+var] = sim_overlap_sorted[var]    
    
    # Create empty lists to store efficiency values for each variable
    kge_values = []
    nse_values = []
    rmse_values = []

    for ID in ds['ID'].values:
        observed = ds['obs'].sel(ID=ID).values
        simulated = ds['sim'].sel(ID=ID).values

        # Remove NaN values
        observed, simulated = filter_nan(observed, simulated)

        # Calculate efficiency metrics
        if (observed is not np.nan) and (simulated is not np.nan):
            kge = calculate_kge(observed, simulated)
            nse = calculate_nse(observed, simulated)
            rmse = calculate_rmse(observed, simulated)
        else:
            kge = np.nan
            nse = np.nan
            rmse = np.nan
            
        # Append efficiency values to lists
        kge_values.append(kge)
        nse_values.append(nse)
        rmse_values.append(rmse)

    # Add efficiency values as variables to the dataset
    ds['KGE'] = (('ID'), kge_values)
    ds['NSE'] = (('ID'), nse_values)
    ds['RMSE'] = (('ID'), rmse_values)
    
    # Warning back to default
    warnings.filterwarnings("default")
    
    return ds


def filter_nan(s, o):
    """
    Removes NaN values from simulated and observed data and returns NaN if either data is empty.

    Parameters:
        s (numpy.ndarray): Simulated data.
        o (numpy.ndarray): Observed data.

    Returns:
        tuple: Tuple containing filtered simulated and observed data arrays.
    """
    # Combine simulated and observed data
    data = np.array([s.flatten(), o.flatten()]).T
    
    # Remove rows containing NaN values
    data = data[~np.isnan(data).any(axis=1)]
    
    # If data is empty, return NaN
    if len(data) == 0:
        return np.nan, np.nan
    
    # Separate filtered simulated and observed data
    s_filtered = data[:, 0]
    o_filtered = data[:, 1]
    
    return s_filtered, o_filtered

def calculate_kge(observed, simulated):
    mean_obs = np.mean(observed)
    mean_sim = np.mean(simulated)
    std_obs = np.std(observed)
    std_sim = np.std(simulated)
    correlation = np.corrcoef(observed, simulated)[0, 1]
    kge = 1 - np.sqrt((correlation - 1) ** 2 + (std_sim / std_obs - 1) ** 2 + (mean_sim / mean_obs - 1) ** 2)
    return kge

def calculate_nse(observed, simulated):
    mean_obs = np.mean(observed)
    nse = 1 - np.sum((observed - simulated) ** 2) / np.sum((observed - mean_obs) ** 2)
    return nse

def calculate_rmse(observed, simulated):
    rmse = np.sqrt(np.mean((observed - simulated) ** 2))
    return rmse
