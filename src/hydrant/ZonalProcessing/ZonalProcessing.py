"""
Common functions for manupulating the gis outfput from other sources
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, List
from itertools import product
import sys
import xarray as xr
import re



def Fraction_to_Xarray(df,
                       info,
                       mapping,
                       remove_zero_columns=True,
                       majority=False,
                       threshold = None):

    # Sort the mapping dictionary based on keys
    mapping = dict(sorted(mapping.items()))

    # set the info ID as index
    df.set_index(info.get('ID'), inplace=True)
    #print(df)

    # keep the column that start from prefix and their int is in mapping keys
    # Regular expression pattern to match numeric part
    pattern = r'\d+'
    # Extract numeric part from each column name starting with 'frac'
    frac_columns = [col for col in df.columns if col.startswith(info.get('prefix'))]
    numeric_parts = [int(re.search(pattern, col).group()) for col in frac_columns]
    # Filter columns based on whether their numeric part is in the mapping keys
    filtered_columns = [col for col, num_part in zip(frac_columns, numeric_parts) if num_part in mapping]
    df = df[filtered_columns]
    #print(df)

    # supress values smaller than a value to 0
    if threshold is not None:
        df[df < threshold] = 0.00


    # Get the complete fraction part of the columns
    # Regular expression pattern to extract the non-digit part
    pattern = r'\D+'
    # Extract the non-digit part of each string using regular expressions
    non_digit_parts = [re.search(pattern, col).group() for col in df.columns]
    non_digit_parts = list(set(non_digit_parts))
    if len (non_digit_parts) > 1:
        sys.exit('There are multiple values of'+info.get('prefix'))
    else:
        non_digit_parts = non_digit_parts[0]
        #print ('Non-digit part of the dataframe is: '+ non_digit_parts)

    # remove the non-integer part from column name and sort based on integer
    # Define regular expression pattern to remove non-digit part
    pattern = r'\D+'
    # Extract only the digit part of the column names using regular expressions
    new_columns = [int(re.sub(pattern, '', col)) for col in df.columns]
    # Convert integer column numbers to strings
    new_columns = [str(col) for col in new_columns]
    df.columns = new_columns
    # Rename the columns
    new_columns = sorted(new_columns, key=lambda x: int(x))
    df = df[new_columns]
    #print(df)

    # add zeros contributions
    # Iterate over rows and add columns for missing fractions
    for idx, row in df.iterrows():
        for integer in mapping.keys():
            if f"{integer}" not in row.index:
                df.at[idx, f"{integer}"] = 0
    #print(df)

    # remove the non-integer part from column name and sort based on integer
    # Define regular expression pattern to remove non-digit part
    # Rename the columns
    new_columns = sorted(df.columns, key=lambda x: int(x))
    # Sort the columns based on their integer values
    df = df[new_columns]
    #print(df)

    # renormalizing
    # Calculate the sum of fractions for each fraction number
    fraction_sums = df.sum()
    # Find the fraction number with the maximum sum
    max_fraction_number = fraction_sums.idxmax()
    #print("Fraction number with the maximum portion:", max_fraction_number)
    for index, row in df.iterrows():
        # Calculate the sum of frac columns
        frac_sum = row.sum()
        # Check if frac_sum is negative or zero
        if frac_sum < 0:
            print('sum is negative')
        elif frac_sum == 0:
            df.loc[index, max_fraction_number] = 1
        else:
            # Normalize the frac columns
            df.loc[index, row.index] /= frac_sum
    #print(df)

    # possible removal of the zeros
    if remove_zero_columns is True:
        df = df.loc[:, (df != 0).any(axis=0)]
        # also filter the mapping dictionary for the non zero column:
        mapping = {key: value for key, value in mapping.items() if int(key) in map(int, df.columns)}

    # add fraction to column names
    new_columns = [non_digit_parts + col for col in df.columns]
    df.columns = new_columns

    # ==============
    # convert the fractions into 0 and 1 for majority
    if majority:
        df = df.eq(df.max(axis=1), axis=0).astype(int)
    # create xarray for majority and frac_
    data_array_frac = xr.DataArray(df.values, dims=(info.get('ID'), info.get('name_of_variable')),\
                                   coords={info.get('ID'): df.index, info.get('name_of_variable'): list(mapping.keys())})
    # Get the column name with the maximum value (excluding 'majority' column)
    df['majority'] = df.idxmax(axis=1)
    df['majority'] = df['majority'].str.extract(r'(\d+)').astype(int)
    data_array_majority = xr.DataArray(df['majority'].values, dims=(info.get('ID'),),\
                                       coords={info.get('ID'): df.index})
    # pass the names from the mapping info
    data_array_name = xr.DataArray(list(mapping.values()), dims=(info.get('name_of_variable'),),\
                                   coords={info.get('name_of_variable'): list(mapping.keys())})

    # Combine into xarray Dataset
    ds = xr.Dataset({info.get('name_of_variable')+'_majority': data_array_majority,\
                     info.get('name_of_variable')+'_frac': data_array_frac,\
                     info.get('name_of_variable')+'_names': data_array_name})

    #print(ds)

    return ds, df


def Stat_to_Xarray(df, info, mapping):

    # Set 'COMID' column as index
    df.set_index(info['ID'], inplace=True)

    # Initialize an empty dictionary to store variables and their attributes
    variables = {}

    # Iterate over each column (except the ID) and create a variable with the COMID as coordinate
    for col in df.columns:
        if col != info['ID']:
            variable = xr.DataArray(df[col].values, dims=[info['ID']], coords={info['ID']: df.index})
            if col in mapping:
                variable.attrs = mapping[col]
            variables[col] = variable

    # Create xarray Dataset with the variables
    ds = xr.Dataset(variables)

    return ds


def intersect_df(*dfs: pd.DataFrame,
                 df_mappings: Union[Dict[str, Dict[str, str]], None] = None,
                 output_dim={'id':'ID'},
                 threshold = None,
                 remove_zero_combinations: bool = True):

    # turn input into list
    dfs = list(dfs)

    # Check if mappings are provided
    if df_mappings is None:
        df_mappings = {}

    # Loop through each DataFrame
    for idx, df in enumerate(dfs):

        mapping = df_mappings.get(f'df{idx+1}', {})
        ID = mapping.get('id')
        prefix = mapping.get('prefix', '')
        data_name = mapping.get('data_name', f'Data_{idx+1}')

        # Set the ID column to index for the DataFrame
        if ID is not None:
            df.index = df[ID].astype(int)
            df.index.name = None
            df.drop(columns=[ID], inplace=True)
            df = df.sort_index()

        # Select columns based on prefix and replace them with data_name
        prefix_cols = [col for col in df.columns if col.startswith(prefix)]
        df = df[prefix_cols] # keep the columns with prefix
        for col in df.columns:
            new_col_name = col.replace(prefix, data_name)
            df[new_col_name] = df[col]
            df = df.drop(columns=[col])

        # update the dataframe
        dfs[idx] = df

    # Compare the indexes
    index_comparison = all(dfs[i].index.equals(dfs[i + 1].index) for i in range(len(dfs) - 1))
    if index_comparison:
        print("The indexes of all DataFrames are exactly the same with the same order.")
    else:
        sys.exit("The indexes of DataFrames are different or in a different order.")

    # Generate combinations and calculate products
    combinations = list(product(*(df.columns for df in dfs)))
    result = pd.DataFrame(columns=[f'{" ".join(cols)}' for cols in combinations], \
                          index=dfs[0].index)
    combination_list = []
    max_fraction_number = result.sum().idxmax()

    # loop over combination
    for cols in combinations:
        col_product = dfs[0][cols[0]] #initialize the product
        for i in range(1, len(dfs)):
            col_product = col_product * dfs[i][cols[i]] # multiply the product
        # update the results
        if threshold is not None:
            col_product[col_product < threshold] = 0
        result[f'{" ".join(cols)}'] = col_product

        # keep the positive or existing combinations
        if (sum(col_product)>0):
            combination_list.append(' '.join(cols))
        elif (sum(col_product)==0):
            if not remove_zero_combinations:
                combination_list.append(' '.join(cols))
        else:
            sys.exit("sum of columns cannot be negative")

    # normalize
    for index, row in result.iterrows():
        if row.sum() == 0:
            result.loc[index, max_fraction_number] = 1
    result = result.div(result.sum(axis=1), axis=0)

    # total number of non zero combinations:
    print('total number of non zero combinations: ',len(combination_list))

    # Report combinations with non-zero values
    report = pd.DataFrame(combination_list, columns=['comb_explanation'])
    # Split combinations into separate columns based on DataFrame columns
    for idx, df in enumerate(dfs, start=0):
        mapping = df_mappings.get(f'df{idx+1}', {})
        data_name = mapping.get('data_name', f'Data_{idx+1}')
        report[data_name] = report['comb_explanation'].apply(lambda x: x.split()[idx])
        report[data_name] = report[data_name].str.extract(r'(\d+)')  # f'{chr(65 + i)}'
        report[data_name] = report[data_name].astype(int)
    report ['comb_int'] = np.arange(len(report))+1

    # remove the zero combination from result dataframe
    if remove_zero_combinations:
        zeros_columns = result.columns[(result == 0).all()]  # Get columns with all zeros
        # Drop columns that consist entirely of zeros
        result = result.drop(zeros_columns, axis=1)

    # rename the result header to 1 to n
    for i, col in enumerate(result.columns, start=1):
        new_col_name = f'comb_{i:04d}'  # Using f-string to format column names
        result.rename(columns={col: new_col_name}, inplace=True)

    fraction = xr.DataArray(result.values,
                            dims=('id', 'comb'),
                            coords={'id': result.index,
                                    'comb': result.columns})

    max_fraction = xr.DataArray(result.idxmax(axis=1).values,
                                dims=('id'),
                                coords={'id': result.index})

    # convert report to xarray
    total_ds = report.to_xarray()
    total_ds = total_ds.drop_vars(['index'])
    total_ds = total_ds.swap_dims({'index':'comb'})

    # add other variables to this
    total_ds['fraction'] = fraction
    total_ds['max_fraction'] = max_fraction

    # rename the output_dim if provided
    total_ds = total_ds.rename_dims(output_dim)

    # return
    return result , report, total_ds
