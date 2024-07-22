import numpy as np
import pandas as pd
import xarray as xr


def rename_vars_and_dims(ds, var_id, dim_id, var_time, dim_time):
    
    ds_renamed = ds
    
    # Check if variable and dimension have the same name
    if var_id == dim_id:
        ds_renamed = ds_renamed.rename({var_id: 'ID'})
    else:
        ds_renamed = ds_renamed.rename({var_id: 'ID', dim_id: 'ID'})
    
    # Check if variable and dimension have the same name
    if var_time == dim_time:
        ds_renamed = ds_renamed.rename({var_time: 'time'})
    else:
        ds_renamed = ds_renamed.rename({var_time: 'time', dim_time: 'time'})
    
    # return
    return ds_renamed

def rename_ds_variables(ds, info_ds):
    ds_renamed = rename_vars_and_dims(ds, info_ds['var_id'], info_ds['dim_id'], info_ds['var_time'], info_ds['dim_time'])
    return ds_renamed

def ObjectiveFunction(obs,
                      sim,
                      info_obs={'var': 'Discharge', 'var_id': 'COMID', 'dim_id': 'COMID', 'var_time': 'time', 'dim_time': 'time'},
                      info_sim={'var': 'Discharge', 'var_id': 'COMID', 'dim_id': 'COMID', 'var_time': 'time', 'dim_time': 'time'},
                      TimeStep='daily'):
    
    # Extract relevant information
    var_name_obs = info_obs['var']
    var_name_sim = info_sim['var']
    
    # rename
    obs = rename_ds_variables(obs, info_obs)
    sim = rename_ds_variables(sim, info_sim)
    
    # Round to the daily time steps
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

    # Create new xarray object to store efficiency values
    ds = xr.Dataset()
    
    # create the 
    ds [var_name_obs+'_obs'] = obs_overlap_sorted[var_name_obs]
    ds [var_name_sim+'_sim'] = sim_overlap_sorted[var_name_sim]
    ds ['time'] = obs_overlap['time']
    ds ['ID'] = obs_overlap['ID']
    
    # Create empty lists to store efficiency values for each variable
    kge_values = []
    nse_values = []
    rmse_values = []

    for ID in ds['ID'].values:
        observed = ds[var_name_obs+'_obs'].sel(ID=ID).values
        simulated = ds[var_name_sim+'_sim'].sel(ID=ID).values

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
        
        # # print
        # print(ID)
        # kge = calculate_kge(observed, simulated)
        # print(kge)
        # nse = calculate_nse(observed, simulated)
        # print(nse)
        # rmse = calculate_rmse(observed, simulated)
        # print(rmse)
        # print('+++++++')

    # Add efficiency values as variables to the dataset
    ds['KGE'] = (('ID'), kge_values)
    ds['NSE'] = (('ID'), nse_values)
    ds['RMSE'] = (('ID'), rmse_values)
    
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
