import numpy as np
import pandas as pd
import xarray as xr
import warnings
from   scipy.spatial import ConvexHull
from   itertools import product

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
                      TimeStep=None,
                      sort=[False],
                      quantile=[None],
                      depth=[None],
                      time_agg=[None]):
    
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
    
    # Round to the daily, hourly, or monthly time steps
    if TimeStep is not None:
        if TimeStep.lower() in ['monthly','m','month','months']:
            obs['time'] = obs['time'].to_index().floor('M')  # 'M' for month
            sim['time'] = sim['time'].to_index().floor('M')
        elif TimeStep.lower() in ['daily','d','day','days']:
            obs['time'] = obs['time'].to_index().floor('D')  # 'D' for day
            sim['time'] = sim['time'].to_index().floor('D')
        elif TimeStep.lower() in ['hourly','h','hour','hours']:
            obs['time'] = obs['time'].to_index().floor('H')  # 'H' for hour
            sim['time'] = sim['time'].to_index().floor('H')
        else:
            sys.exit("TimeStep is not recognized; the TimeStep should be 'monthly', 'daily', or 'hourly'")

    # Find overlapping time
    common_time = np.intersect1d(obs['time'].values, sim['time'].values)
    
    # Get shared IDs
    common_ids = np.intersect1d(obs['ID'].values, sim['ID'].values)
    
    if common_time.size != 0 and common_ids.size != 0:

        # Slice data based on overlapping time
        obs_overlap = obs.sel({'time': common_time})
        sim_overlap = sim.sel({'time': common_time})

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
        ds ['time'] = obs_overlap_sorted['time']
        ds ['ID'] = obs_overlap_sorted['ID']
        if info_obs.get('var_extra') is not None:
            for var in info_obs.get('var_extra'):
                ds ['obs_'+var] = obs_overlap_sorted[var]
        if info_sim.get('var_extra') is not None:
            for var in info_sim.get('var_extra'):
                ds ['sim_'+var] = sim_overlap_sorted[var]


        # Check if all lists have length equal to 1
        if not (len(sort) == 1 and len(quantile) == 1 and len(depth) == 1 and len(time_agg) == 1):
            raise ValueError("Expected all input lists (sort, quantile, depth, time_agg) to have a length of 1.")

        # Generate combinations of parameters
        combinations = product(sort, quantile, depth, time_agg)

        for s, q, d, t in combinations:

            # Create a descriptive string for each combination
            combination_str = '_'
            if s:
                combination_str += "s_"
            if q is not None:
                combination_str += f"q_{q}_"
            if d is not None:
                combination_str += f"d_{d}_"
            if t is not None:
                combination_str += f"a_{t}"
            # Strip trailing underscore if needed
            combination_str = combination_str.rstrip('_')
            
            # TEMPORARY: TO BE REMOVED WHEN USING MULTIPLE CONDITIONS
            combination_str = ''

            # Create a detailed explanation for the combination
            explanation_str = ''
            if s:
                explanation_str += "Observed and simulated values are sorted. "
            if q is not None:
                explanation_str += f"Observed values below the quantile of {q} are removed. "
            if d is not None:
                explanation_str += f"Events are selected based on a depth threshold of {d}. "
            if t is not None:
                explanation_str += f"Time is aggregated to {t} time steps."

            # Calculate objective functions for the given combination
            nse_values, kge_values, rmse_values = calculate_all_objective(
                ds,
                sort=s,
                quantile=q,
                depth=d, 
                time_agg=t
            )

            # Add efficiency values as variables to the dataset
            ds[f'KGE{combination_str}'] = (('ID'), kge_values)
            ds[f'KGE{combination_str}'].attrs['long_name'] = f"KGE: Kling–Gupta efficiency. {explanation_str}"

            ds[f'NSE{combination_str}'] = (('ID'), nse_values)
            ds[f'NSE{combination_str}'].attrs['long_name'] = f"NSE: Nash–Sutcliffe efficiency. {explanation_str}"

            ds[f'RMSE{combination_str}'] = (('ID'), rmse_values)
            ds[f'RMSE{combination_str}'].attrs['long_name'] = f"RMSE: Root Mean Squared Error. {explanation_str}"

            # Warning back to default
            warnings.filterwarnings("default")
        
    else:
        # should be populated
        ds = None
    
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

def depth_function(observed, simulated, depth):
    """
    Removes values that are not part of the convex hull in a sequential way and replaces them with np.nan.
    
    Parameters:
    observed (numpy array): The input data (1D array).
    simulated (numpy array): The input data (1D array).
    depth (int): The depth for the iterative convex hull calculation.
    
    Returns:
    numpy array: The modified data with non-convex hull values replaced by np.nan for observed and simulated.
    """
    df = pd.DataFrame()
    df ['Qt'] = observed
    df ['Qs'] = simulated
    df = df.dropna(subset=['Qt', 'Qs']).reset_index(drop=True)
    df['Qt+1'] = df['Qt'].shift(-1)
    df = df.dropna(subset=['Qt+1']).reset_index(drop=True)
    df['Depth'] = np.nan  # Initialize depth column with NaNs
    
    # Copy data for iterative depth calculation
    Qt = df['Qt'].values
    Qt_plus_1 = df['Qt+1'].values
    original_indices = np.arange(len(Qt))

    # Loop to calculate depth and update the DataFrame
    for m in range(1, depth + 1):
        if len(Qt) < 3:
            # Break if fewer than 3 points remain
            break

        # Stack data for ConvexHull
        points = np.column_stack((Qt, Qt_plus_1))

        # Calculate the convex hull
        hull = ConvexHull(points, qhull_options='QJ')

        # Update the depth for the convex hull points
        df.iloc[original_indices[hull.vertices], df.columns.get_loc('Depth')] = m

        # Remove the convex hull points from Qt, Qt_plus_1, and original_indices
        Qt = np.delete(Qt, hull.vertices)
        Qt_plus_1 = np.delete(Qt_plus_1, hull.vertices)
        original_indices = np.delete(original_indices, hull.vertices)
    
    # remove the nan values and pass the values to oserved and simulated
    df['Qt'] = np.where(df['Depth'].isna(), np.nan, df['Qt'])
    df['Qs'] = np.where(df['Depth'].isna(), np.nan, df['Qs'])
    observed = df['Qt'].values
    simulated = df['Qs'].values
        
    return observed, simulated

def calculate_all_objective(ds,
                            sort = None,
                            quantile = None,
                            depth = None, 
                            time_agg = None):
    
    # Create empty lists to store efficiency values for each variable
    nse_values = []
    kge_values = []
    rmse_values = []
    
    # if time_agg
    if time_agg is not None:
        ds = ds.resample(time=time_agg).mean()

    # Loop over the ID of the 
    for ID in ds['ID'].values:
        observed = ds['obs'].sel(ID=ID).values
        simulated = ds['sim'].sel(ID=ID).values
        
        # sort if sort is true
        if sort is True:
            observed = np.sort(observed)
            simulated = np.sort(simulated)
            
        # quntile
        if quantile is not None:
            
            # observed
            observed = np.array(observed, dtype=float)
            # Get the non-NaN values
            non_nan_values = observed[~np.isnan(observed)]    
            # Calculate the quantile value
            quantile_value = np.quantile(non_nan_values, quantile)
            # Set values below the quantile to NaN
            observed[observed < quantile_value] = np.nan
            
            # simulated
            simulated = np.array(simulated, dtype=float)
            # Get the non-NaN values
            non_nan_values = simulated[~np.isnan(simulated)]    
            # Calculate the quantile value
            quantile_value = np.quantile(non_nan_values, quantile)
            # Set values below the quantile to NaN
            simulated[simulated < quantile_value] = np.nan
            
        # # print
        # print(observed)
        # print(simulated)
        # print(type(observed))
        # print(type(simulated))
        # print(len(observed))
        # print(len(simulated))
        
        # find the depth and put the rest as nan values
        if depth is not None:
            print(depth)
            observed, simulated = depth_function(observed, simulated, depth)
            
#         # print
#         print(observed)
#         print(simulated)
#         print(type(observed))
#         print(type(simulated))
#         print(len(observed))
#         print(len(simulated))
            
        # allocate nse, kge, rmse
        nse = np.nan
        kge = np.nan
        rmse = np.nan
        
        if isinstance(observed, np.ndarray) and isinstance(simulated, np.ndarray):
            observed, simulated = filter_nan(observed, simulated)

            # Calculate efficiency metrics
            if (observed is not np.nan) and (simulated is not np.nan):
                nse = calculate_nse(observed, simulated)
                kge = calculate_kge(observed, simulated)
                rmse = calculate_rmse(observed, simulated)
            
        # Append efficiency values to lists
        nse_values.append(nse)
        kge_values.append(kge)
        rmse_values.append(rmse)
        
    # return
    return nse_values, kge_values, rmse_values
