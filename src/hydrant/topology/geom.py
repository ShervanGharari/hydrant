"""
Common functions for working with river and sub-basin geometries
in the context of hydrological modelling
"""

import numpy      as np
import networkx   as nx
import geopandas  as gpd
import pandas     as pd
import os
import sys
import xarray as xr
import pint_xarray

from typing import (
    Optional,
    Dict,
    Tuple,
    Union,
    Sequence,
    Iterable,
    List,
)

from .river_graph import find_upstream

def merit_read_file (
    pfafs: list,
    path_riv: str,
    riv_file_template: str,
    path_cat: str,
    cat_file_template: str,
    path_cst: Optional [str] = None,
    cst_file_template: Optional [str] = None):


    # initializaing empty geodataframe
    cat_all = gpd.GeoDataFrame()
    riv_all = gpd.GeoDataFrame()

    for pafaf in pfafs:

        # read files cat, riv, cst
        riv = gpd.read_file(os.path.join(path_riv, riv_file_template.replace('*', pafaf)))
        cat = gpd.read_file(os.path.join(path_cat, cat_file_template.replace('*', pafaf)))
        if not path_cst is None and not cst_file_template is None:
            cst = gpd.read_file(os.path.join(path_cst, cst_file_template.replace('*', pafaf)))
            # add cat and cst
            cst = merit_cst_prepare(cst,
                                    {'id':'COMID','area':'unitarea'},
                                    cat = cat,
                                    cat_col_id = 'COMID')
        else:
            cst = None

        # merge the cat and cst
        cat = merit_cat_cst_merge (cat,
                                   cst = cst)

        # append the files
        riv_all = pd.concat([riv_all, riv])
        cat_all = pd.concat([cat_all, cat])

    # sort COMID
    riv_all.sort_values(by='COMID', axis='index', inplace=True)
    riv_all.reset_index(drop=True, inplace=True)

    # sort COMID
    cat_all.sort_values(by='COMID', axis='index', inplace=True)
    cat_all.reset_index(drop=True, inplace=True)

    # set the projection
    riv_all.set_crs(epsg=4326, inplace=True, allow_override=True)
    cat_all.set_crs(epsg=4326, inplace=True, allow_override=True)

    return riv_all, cat_all


def merit_cst_prepare(
    cst: gpd.GeoDataFrame,
    cst_col: Optional[Dict[str, str]] = None,
    cat: Optional[gpd.GeoDataFrame] = None,
    cat_col_id: Optional[str] = None,
    cst_col_id_reset: bool = True,
    crs: int = 4326,
    *args,
    ) -> gpd.GeoDataFrame:

    # get the possible existing id, area if exists
    cst_col_id = 'COMID'
    cst_col_area = 'unitarea'
    if cst_col is not None:
        cst_col_id = cst_col.get('id')
        cst_col_area = cst_col.get('area')

    if not cst.crs:
        cst.set_crs(epsg=4326, inplace=True, allow_override=True)
        warnings.warn('CRS of the coastal hillslope Shapefile has been assumed to be EPSG:4326')

    if cst_col_id_reset:
        max_cat_id = 0
        if cat is not None:
            max_cat_id = cat[cat_col_id].max()
        cst[cst_col_id] = range(max_cat_id+1,
                                max_cat_id+1+len(cst))
    else:
        if not cst_col_id in cst.columns:
            sys.exit('the corresponding id is not given for cosatl hillslope')
        else:
            max_cat_id = 0
            if cat is not None:
                max_cat_id = cat[cat_col_id].max()
            min_cst_id = cst[cst_col_id].min()
            if min_cst_id < max_cat_id:
                sys.exit('there is some mixed up COMID between the cat and costal hillslope')

    if not cst_col_area in cst.columns: # then we need to populate the id
        cst[cst_col_area] = cst.to_crs(epsg=6933).area / 1e6

    # assign `coastal_hillslope` flag
    cst['hillslope'] = 1

    # drop FID column
    cst = cst.drop(columns = ['FID'])

    # return
    return cst

def merit_cat_cst_merge (cat,
                         cst: Optional[gpd.GeoDataFrame] = None,
                         crs: int = 4326) -> gpd.GeoDataFrame:
    '''add the hillslope flag and merge the cat and costal hillslope
    into one geodataframe
    Parameters
    ----------
    cat: geopandas.GeoDataFrame
        The catchment object derived from `merit_read_file`.
    cst: geopandas.GeoDataFrame
        The costal hillslope object read in `merit_read_file`

    Returns
    -------
    catchment: geopandas.GeoDataFrame
        combined cat and costal hillslope with their hillslope flag
    '''

    cat['hillslope'] = 0
    if not cst is None:
        cst['hillslope'] = 1
        catchment = gpd.GeoDataFrame(pd.concat([cat, cst]))
    else:
        catchment = cat

    # assign crs
    catchment.set_crs(epsg=crs, inplace=True, allow_override=True)
    catchment.reset_index(drop=True, inplace=True)

    return catchment

def hdma_read_file(
    regions: list,
    path_riv: str,
    riv_file_template: str,
    path_cat: str,
    cat_file_template: str):

    # initializaing empty geodataframe
    cat_all = gpd.GeoDataFrame()
    riv_all = gpd.GeoDataFrame()

    for region in regions:

        # read files cat, riv, cst
        riv = gpd.read_file(os.path.join(path_riv, riv_file_template.replace('*', region)))
        cat = gpd.read_file(os.path.join(path_cat, cat_file_template.replace('*', region)))

        # slice based on the region for cat
        riv = riv[(riv['seg_id'] >= 1000000*int(region)) & (riv['seg_id'] <= 1000000*int(region)+999999)]

        # identifying hillslope catchments
        cat = hdma_assining_cst(riv,
                                {'id':'seg_id'},
                                cat,
                                {'id':'hruid'})

        # append the files
        riv_all = pd.concat([riv_all, riv])
        cat_all = pd.concat([cat_all, cat])

    # return
    return riv_all, cat_all


def hdma_assining_cst(riv,
                      riv_id,
                      cat,
                      cat_id):

    # get the name of the colomns
    riv_col_id = riv_id.get('id')
    cat_col_id = cat_id.get('id')

    # check if the IDs from river all in the cat
    if not riv[riv_col_id].isin(cat[cat_col_id]).all():
        sys.exit('it seems the river passed has river segments that are not in cat')

    # assign hillslope flag to ids that are not
    cat ['hillslope'] = 0
    cat.loc[~cat[cat_col_id].isin(riv[riv_col_id]), 'hillslope'] = 1

    # return
    return cat

def tdx_read_file  (codes:list,
                    path_riv: str,
                    riv_file_template: str,
                    path_cat: str,
                    cat_file_template: str,
                    prepare_for_ntopo = True):

    # initializaing empty geodataframe
    cat_all = gpd.GeoDataFrame()
    riv_all = gpd.GeoDataFrame()

    for code in codes:

        # read files cat, riv, cst
        riv = gpd.read_file(os.path.join(path_riv, riv_file_template.replace('*', code)))
        cat = gpd.read_file(os.path.join(path_cat, cat_file_template.replace('*', code)))

        # check if the length of riv is larger or equal to cat
        if len(riv) < len(cat):
            sys.exit('the length of river is smaller than the length of cat')

        # add hillslope
        cat ['hillslope'] = 0

        # add pfaf to riv and cat
        riv['pfafcode'] = int(code)
        cat['pfafcode'] = int(code)

        if prepare_for_ntopo:

            # identify the linkage
            riv, cat = tdx_identify_linkage(riv, cat)

            # remove of linkages
            riv = tdx_contraction_of_linkage(riv)

            # check if the cat and riv are the same
            if set(riv['LINKNO']) != set(cat['streamID']):
                sys.exit('cat and riv have different LINKNO and streamID for code: '+ code)

            # sort based on ID and pfaf
            riv.sort_values(by=['LINKNO', 'pfafcode'], axis='index', inplace=True)
            riv.reset_index(drop=True, inplace=True)

            # sort based on ID and pfaf
            cat.sort_values(by=['streamID', 'pfafcode'], axis='index', inplace=True)
            cat.reset_index(drop=True, inplace=True)

            # pass area from the riv to cat
            riv['area'] = riv.apply(lambda x: x['DSContArea'] if x['strmOrder'] == 1 else x['DSContArea'] - x['USContArea'], axis=1)
            cat['area'] = riv ['area']

        # append the files
        riv_all = pd.concat([riv_all, riv])
        cat_all = pd.concat([cat_all, cat])

    # sort based on ID and pfaf
    riv_all.sort_values(by=['LINKNO', 'pfafcode'], axis='index', inplace=True)
    riv_all.reset_index(drop=True, inplace=True)

    # sort based on ID and pfaf
    cat_all.sort_values(by=['streamID', 'pfafcode'], axis='index', inplace=True)
    cat_all.reset_index(drop=True, inplace=True)

    # set the projection
    riv_all.set_crs(epsg=4326, inplace=True, allow_override=True)
    cat_all.set_crs(epsg=4326, inplace=True, allow_override=True)

    return riv_all, cat_all

def tdx_identify_linkage(riv,
                         cat):

    # identify the linkage in the river
    riv_slice = riv.copy()
    riv_slice = riv_slice[~riv_slice['LINKNO'].isin(cat['streamID'])]

    # check if the riv len, riv_slice add up to cat
    if len(riv_slice)+len(cat) != len(riv):
        sys.exit('There are differences between the river and cat')

    # check the ids
    if set(riv_slice['LINKNO'].values) | set(cat['streamID'].values) != set(riv['LINKNO'].values):
        sys.exit('inside identifying linkage, the LINKNO and streamID are not the same as riv LINKNO')

    # add linkage flag
    riv['linkage'] = 0
    riv.loc[riv['LINKNO'].isin(riv_slice['LINKNO'].values), 'linkage'] = 1

    # return
    return riv, cat

def tdx_contraction_of_linkage  (df,
                                 mapping = {'id':'LINKNO','next_id':'DSLINKNO','linkage_flag':'linkage'},
                                 column_to_drop = ['USLINKNO1', 'USLINKNO2']):

    # get the name of id, next_id, linkage_flag
    downID = mapping.get('next_id')
    ID = mapping.get('id')
    linkage_flag = mapping.get('linkage_flag')

    # make sure the id, next_id, linkage_flag is int
    df[downID] = df[downID].astype(int)
    df[ID] = df[ID].astype(int)
    df[linkage_flag] = df[linkage_flag].astype(int)

    for index, row in df.iterrows():
        if row[linkage_flag] == 1: # if linkage
            # find the upstream segmenet in which drain into
            idx = np.array(df[df[downID]==row[ID]].index)
            df.loc[idx,downID]=row[downID] # correct the downstream for upstream segment to skip linkage

    # remove the linakges
    df = df[df[linkage_flag]==0].reset_index(drop=True)

    # drop the colomn
    df = df.drop(columns = column_to_drop)

    # df
    df = add_immediate_upstream (df,
                                 mapping = mapping)

    # return
    return df


#     # keep the needed columns and drop the columns that will be replace
#     attributes_df = df.copy()
#     attributes_df = attributes_df.drop(columns = column_to_drop)
#     attributes_df = attributes_df.drop(columns = [downID, linkage_flag])

#     # keep the needed columns only
#     df = df [[ID, downID, linkage_flag]]

#     # Create a directed graph
#     G = nx.DiGraph()

#     # Add edges from DataFrame with attributes
#     for _, row in df.iterrows():
#         if row[downID] > -0.01:  # Ignore nodes with negative downstream
#             G.add_edge(row[downID], row[ID], linkage_flag_G=row[linkage_flag])

#     # Contract edges based on linkage attribute being 1; in tdx hydro they are just linking
#     edges_to_contract = [(u, v) for u, v, d in G.edges(data=True) if d.get('linkage_flag_G') == 1]
#     for u, v in edges_to_contract:
#         G = nx.contracted_edge(G, (u, v), self_loops=False)

#     # convert to pandas dataframe
#     updated_df = [(u, v, d) for u, v, d in G.edges(data=True)]
#     updated_df = pd.DataFrame(updated_df, columns=[downID, ID, 'attributes'])
#     updated_df = updated_df.drop(columns = ['attributes'])
#     updated_df[ID] = updated_df[ID].astype(int)

#     print(len(updated_df))
#     print(len(attributes_df))

#     # add attributes
#     updated_df = pd.merge(updated_df, attributes_df, on=ID, how='inner')

#     print(len(updated_df))
#     print(len(attributes_df))

#     # add immidiate upstream and update
#     updated_df = add_immediate_upstream (updated_df)

    # # return
    # return updated_df

def prepare_ntopo(
    riv: gpd.GeoDataFrame,
    riv_cols: Dict[str, str],
    cat: gpd.GeoDataFrame,
    cat_cols: Dict[str, str],
    network: str,
    outlet_val: int = -9999,
    crs: int = 4326,
    calc_rank: Optional[bool] = False,
    *args,
    **kwargs,
) -> gpd.GeoDataFrame:
    '''prepares and match the river with costal and other hillslope which are presented in cat
    but not presented in riv. It it so make sure the routing model has the needed variables
    Parameters
    ----------
    riv: geopandas.GeoDataFrame
        The river network object containing river segments
    riv_cols: dict
        A dictionary containing 'id', 'next_id', keys and values
        corresponding to the river segement IDs and their downstream
        river segments, respectively.
    cat: geopandas.GeoDataFrame
        The catchment topology object derived from `merit_read_file`
        that takes care of costal hillslopes, COMIDs, etc using the
        function `merit_cst_prepare` and `merit_cat_cst_merge`.
    cat_cols: dict
        A dictionary containing 'id', 'hillslope', and 'unitarea'
        keys and values corresponding to the catchment IDs and also
        the flag indicating whether the catchment is a coastal
        hillslope one (1) or not (0)
    network: str
        A string that define the input network topology into the function
        This can be one of `merit`, 'hdma', or `tdx`
    outlet_val: int, optional [defaults to -9999]
        Value being assigned to the 'next_id' cell of the each
        river segments indicating the segment is an outlet for the network;
        before any subsetting, if any coastal hillslope segments exist, they
        will only be set to the outlet value.

    Returns
    -------
    river: geopandas.GeoDataFrame
        The prepared river network topology GeoDataFrame object
    cat: geopandas.GeoDataFrame
        For consistency with other functions passes the cat as
        output.
    '''

    # necessary assertions
    assert isinstance(riv, gpd.GeoDataFrame), "`riv` object must be of type geopandas.GeoDataFrame"
    assert isinstance(riv_cols, dict), "`riv_cols` object must be of type dict"
    assert isinstance(cat, gpd.GeoDataFrame), "`cat` object must be of type geopandas.GeoDataFrame"
    assert isinstance(cat_cols, dict), "`cat_cols` object must be of type dict"

    # `id` and `next_id` keys must be included in the `riv_cols` dictionary
    if not 'id' and 'next_id' in riv_cols:
        raise ValueError("`id` and `next_id` must be included in the `riv_cols` dictionary")

    # `id`, `geom`, and `hillslope` keys must be included in the `cat_cols` dictionary
    if not 'id' and 'geom' and 'hillslope' in cat_cols:
        raise ValueError("`id`, `geom` and `hillslope` must be included in the `cat_cols` dictionary")

    # necessary definitions
    riv_col_id = riv_cols.get('id')
    riv_col_next_id = riv_cols.get('next_id')
    hil_col_id = cat_cols.get('hillslope')
    cat_col_id = cat_cols.get('id')
    cat_col_area = cat_cols.get('area')

    # extract the hillslope column from `cat` and assign `id` as index
    hills = cat.set_index(cat_col_id)
    hills = hills[hil_col_id]

    # add the hillslope flag from `cat`
    # `cat` has necessary greater than or equal number of items than `riv`
    river = gpd.GeoDataFrame(riv.join(other=hills, on=riv_col_id, how='outer'),
                             geometry='geometry')

    # assign `outlet_val` to the `hillslope_id` river segments
    river.loc[river[hil_col_id] == 1, riv_col_next_id] = outlet_val

    # sort based on `riv_col_id` and reset_index
    river.sort_values(by=riv_col_id, axis='index', inplace=True)
    river.reset_index(drop=True, inplace=True)

    # sort based on `cat_col_id` and reset_index
    cat.sort_values(by=cat_col_id, axis='index', inplace=True)
    cat.reset_index(drop=True, inplace=True)

    # assign crs
    river.set_crs(epsg=crs, inplace=True, allow_override=True)

    if network.lower() == 'merit':

        # fix cicular segemnts, in other words, those whose 'id' and
        # 'next_id's are the same
        river.loc[river[riv_col_id] == river[riv_col_next_id], riv_col_next_id] = outlet_val

        # next ID is zero! replace with outlet_val
        river.loc[river[riv_col_next_id] == 0, riv_col_next_id] = outlet_val

        # pass the unit area from cat
        river['unitarea'] = 0
        river['unitarea'] = cat['unitarea']

        # centroid of cat
        river['latitude'] = 0
        river['longitude'] = 0
        river['latitude'] = cat.centroid.y
        river['longitude'] = cat.centroid.x

        # fill NAs with `riv_na_val` values in `rivers`
        river.loc[river[hil_col_id] == 1, ['maxup','up1','up2','up3','up4']] = 0
        river.loc[river[hil_col_id] == 1, ['lengthkm','lengthdir','sinuosity','slope','slope_taud']] = 0.001
        river.loc[river[hil_col_id] == 1, ['strmDrop_t','order']] = 1
        river.loc[river[hil_col_id] == 1, 'uparea'] = river.loc[river[hil_col_id] == 1, 'unitarea']

        # values that cannot be negative to a minimume value
        river.loc[river['lengthkm'] <= 0.0, 'lengthkm'] = 0.001 # km
        river.loc[river['slope'] <= 0.0, 'slope'] = 0.0001 # m/m
        river.loc[river['slope_taud'] <= 0.0, 'slope_taud'] = 0.0001 # m/m
        river.loc[river['unitarea'] <= 0.0, 'unitarea'] = 0.001 # km2
        river.loc[river['uparea'] <= 0.0, 'uparea'] = 0.001 # km2

        # approximate width based on uparea based on Eq. 5 of Mizukami et al., 2016
        # with changes to have at least 1 meter of channel
        # area assumed to be in meter square
        river['width'] = 0.0
        river['width'] = 0.001 * ((river['uparea']*10**6)**0.5) + 1

        # dictionary
        data_types = {
            'COMID': int,
            'order': int,
            'NextDownID': int,
            'maxup': int,
            'up1': int,
            'up2': int,
            'up3': int,
            'up4': int,
            'hillslope': int,
        }

        # Convert columns to specified data types
        river = river.astype(data_types)

    elif network.lower() == 'hdma':

        # pass the unit area from cat
        river['area_org'] = 0
        river['area_org'] = cat['area_org']

        # centroid of cat
        river['latitude'] = 0
        river['longitude'] = 0
        river['latitude'] = cat.centroid.y
        river['longitude'] = cat.centroid.x

        # pass the area of cat as flow acc to the river
        river.loc[river[hil_col_id] == 1, 'flow_acc'] = river.loc[river[hil_col_id] == 1, 'area_org']/(10**6)

        #
        river.loc[river[hil_col_id] == 1, 'Length'] = 0
        river.loc[river[hil_col_id] == 1, 'Slope'] = 0.0001

        # non negative values
        river.loc[river['Length'] <= 0.0, 'Length'] = 1 # m
        river.loc[river['Slope'] <= 0.0, 'Slope'] = 0.0001 # m/m

        # replace nan with -9999 except geometry
        columns_to_fill = [col for col in river.columns if col not in ['geometry']]
        river[columns_to_fill] = river[columns_to_fill].fillna(-9999)

        # approximate width based on uparea based on Eq. 5 of Mizukami et al., 2016
        # with changes to have at least 1 meter of channel
        # area assumed to be in meter square
        river['width'] = 0.0
        river['width'] = 0.001 * ((river['flow_acc']*10**6)**0.5) + 1

        # dictionary
        data_types = {
            'OBJECTID': int,
            'PFAF': int,
            'PFAF_CODE': int,
            'PF_TYPE': int,
            'Tosegment': int,
            'seg_id': int,
            'hillslope': int,
        }
        # Convert columns to specified data types
        river = river.astype(data_types)

    elif network.lower() == 'tdx':
        river = river
        cat = cat

    else:
        sys.exit('The network topology is not recognized; it should be merit, hdma, tdx')

    # add rank, which is define in routing model <add name> and MESH model
    if calc_rank:
        river = extract_rank(river,
                             mapping = riv_cols,
                             outlet_value = outlet_val)

    # return
    return river, cat


def subset_ntopo(
    cat: gpd.GeoDataFrame,
    cat_cols: Dict[str, str],
    riv: gpd.GeoDataFrame,
    riv_cols: Dict[str, str],
    shapefile: gpd.GeoDataFrame = None,
    touch_shapefile: Optional[bool] = False,
    outlet_id: Optional[Union[int, Sequence[int]]] = None,
    outlet_val: Optional[int] = -9999,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    '''Intesecting `cat` and `riv` with a given `shapefile`
    or a specified `outlet_id` of interest.

    Parameters
    ----------
    cat: geopandas.GeoDataFrame
        The catchment GeoDataFrame output of `prepare_cat` function
    cat_cols: dict
        A dictionary with 'id' key corresponding to the `cat` ID
        column
    riv: geopandas.GeoDataFrame
        The river GeoDataFrame output of `prepare_riv` function
    riv_cols: int
        A dictionary with 'id' and 'next_id' keys corresponding to
        the `riv` ID and next downstream ID column
    shapefile: geopandas.GeoDataFrame, optional [defaults to None]
        A GeoDataFrame specifying the limits of the river network
        topology of interest
    outlet_id: int, or sequence of ints, optional [defaults to None]
        the ID or list of IDs specifying the most downstream river segment
        to be specified as the outlet of the river network topology
        in addition to any non-contributing subbasins
    outlet_val: int, optional [defaults to -9999]
        An integer being set to the most downstream river segments

    Returns
    -------
    cat_clipped: geopandas.GeoDataFrame
        The clipped catchments object
    riv_clipped: geopandas.GeoDataFrame
        The clipped rivers object
    '''
    # necessary definitions
    cat_col_id = cat_cols.get('id')
    riv_col_id = riv_cols.get('id')
    riv_col_next_id = riv_cols.get('next_id')
    riv_col_upstreamarea = riv_cols.get('upstream_area')
    
    # initialize
    ids = set()
    intersected_ids = set()
    upstream_ids = set()
    
    if shapefile is None and outlet_id is None:
        raise NotImplementedError("Either `shapefile` or `outlet_id` must be specified")

    if shapefile is not None:
        # check `shapefile` dtype
        assert isinstance(shapefile, gpd.GeoDataFrame), "`shapefile` must be of type geopandas.GeoDataFrame"
        # intersection with `cat`
        intersected_ids = cat.overlay(shapefile,
                                      how='intersection',
                                      keep_geom_type=True)[cat_col_id]
        intersected_ids = set(intersected_ids)
        
        # expand outlet_id to include all the intersected COMIDs as well it touch true
        if touch_shapefile:
            print('touch shapefile flag is activated, this may take some time!')
            if outlet_id is None:
                outlet_id = set(intersected_ids)
            else:
                outlet_id = set(outlet_id)
                outlet_id.update(intersected_ids)

    if outlet_id is not None:
        # if it is a single ID, make a set out of it
        outlet_id = set(outlet_id)
        
        # check if outlet_id is included in the `riv` IDs
        assert riv[riv_col_id].isin(outlet_id).any(), "`outlet_id` must be chosen from segments included in `riv`"

        if riv_col_upstreamarea is not None:
            riv_temp = riv[riv[riv_col_id].isin(outlet_id)]
            riv_temp = riv_temp.sort_values(by=riv_col_upstreamarea, ascending=False)
            outlet_id = set(riv_temp[riv_col_id].tolist())
            
        # TBD: this takes time with touch_shapefile flag on as it search all the upstream
        # for the intersected shapefile, it should start from segments that are most downstream
        # this needs uparea to be provided, perhpas optional, to this code and the set is arranged
        # based on the decending order of up area to avoid recalculation.
        # find upstream segments
        for element in outlet_id:
            if element not in upstream_ids:
                upstream_ids.update(find_upstream(gdf=riv,
                                                  target_id=element,
                                                  main_id=riv_col_id,
                                                  ds_main_id=riv_col_next_id))
    
    # update ids with intersected and upstreams
    ids.update(intersected_ids)
    ids.update(upstream_ids)
    
    # index `cat` and `riv` and return `cat_clipped` and `riv_clipped`
    cat_clipped = cat.loc[cat[cat_col_id].isin(ids), :].copy()
    riv_clipped = riv.loc[riv[riv_col_id].isin(ids), :].copy()

    # assign `outlet_val` to the segments without downstream segments
    riv_clipped.loc[~riv_clipped[riv_col_next_id].isin(riv_clipped[riv_col_id]), riv_col_next_id] = outlet_val

    # reset the index
    cat_clipped.reset_index(drop=True, inplace=True)
    riv_clipped.reset_index(drop=True, inplace=True)

    # return
    return cat_clipped, riv_clipped


def box_to_geometry(
    lon_min: float = None,
    lat_min: float = None,
    lon_max: float = None,
    lat_max: float = None,
    lims_box: list = [],
    crs: int = 4326,
    **kwargs,
) -> gpd.GeoDataFrame:
    '''Converts a latitude and longitude bounds to a geopandas
    GeoDataFrame object

    Parameters
    ----------
    lon_min: float, optional [defaults to None]
        Minimum longitude value
    lat_min: float, optional [defaults to None]
        Minimum latitude value
    lon_max: float, optional [defaults to None]
        Maximum longitude value
    lon_min: float, optional [defaults to None]
        Maximum latitude value
    box: list of floats, optional [defaults to [] ]
        List of spatial bounds in the ``[lon_min, lat_min, lon_max, lat_max]``
        form
    crs: int, optional [defaults to 4326]
        Coordinate Reference System EPSG numeber, defaults to
        WGS84 [EPSG:4326]

    Returns
    -------
    geom: geopandas.GeoDataFrame
        a GeoDataFrame object containing the geometry of the
        specified spatial bounds
    '''

    if lon_min and lat_min and lon_max and lat_max is not None:
        geometry = box(lon_min, lat_min, lon_max, lat_max, ccw=True)
    elif any(lims_box):
        geometry = box(*lims_box)
    else:
        raise ValueError("Either the spatial limits or a list of them must be provided")

    return gpd.GeoDataFrame(index=[0], crs=f'epsg:{crs}', geometry=[geometry])


def create_xr_ntopo(riv,
                    cat,
                    network): # can be hdma, or tdx

    if network.lower() == 'merit':

        # sort by COMID
        riv.sort_values(by='COMID').reset_index(drop=True)
        cat.sort_values(by='COMID').reset_index(drop=True)

        # check if the riv and cat IDs are similar
        if sum(riv['COMID'].values - cat['COMID'].values) != 0:
            sys.exit('The COMID of riv and cat should be the same')

        # rename lengthkm to length
        riv = riv.rename(columns={'lengthkm': 'length'})

        # drop geometry to make gdf to df
        ntopo = riv.drop(columns = 'geometry')

        # convert index to n
        ntopo = ntopo.rename_axis('n')

        # to xarray object,
        ntopo = ntopo.to_xarray()

        # add the units to the variables using pint
        ntopo = ntopo.pint.quantify({'length': 'km',
                                     'unitarea': 'km**2',
                                     'uparea': 'km**2'})

        # convert dictionary
        convert = {'length': 'm',
                   'unitarea': 'm**2',
                   'uparea': 'm**2'}

        # covert the units in ntopo
        ntopo = ntopo.pint.to(convert)
        ntopo = ntopo.pint.dequantify()

    elif network.lower() == 'hdma':

        # sort by COMID
        riv.sort_values(by='seg_id').reset_index(drop=True)
        cat.sort_values(by='hruid').reset_index(drop=True)

        # check if the riv and cat IDs are similar
        if sum(riv['seg_id'].values - cat['hruid'].values) != 0:
            sys.exit('The seg_id and hruid of riv and cat should be the same')

        # drop geometry to make gdf to df
        ntopo = riv.drop(columns = 'geometry')

        # convert index to n
        ntopo = ntopo.rename_axis('n')

        # to xarray object,
        ntopo = ntopo.to_xarray()

    elif network.lower() == 'tdx':

        #
        ntopo = riv.drop(columns = 'geometry')

        # to xarray object,
        ntopo = ntopo.to_xarray()

    # return the ntopo xarray object
    return ntopo

def add_immediate_upstream (df,
                            mapping = {'id':'LINKNO','next_id':'DSLINKNO'}):

    # this function add immediate segment of upstream for a river network if not provided
    # it first convert the df into a networkx derected graph, finds the sucessores for
    # river segments, provide the maximume existing upstream segments in column called maxup
    # and the values in up1, up2, up3, etc

    # get the name of ID and downID
    downID = mapping.get('next_id')
    ID = mapping.get('id')

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges from the DataFrame (reversing the direction)
    for _, row in df.iterrows():
        if row[downID] > -0.01:  # Skip nodes with negative downstream
            G.add_edge(row[downID], row[ID])

    # Find immediate upstream nodes for each node
    immediate_upstream = {}
    for node in G.nodes():
        immediate_upstream[node] = list(G.successors(node))

    # Create a new column containing lists of immediate upstream nodes
    df['upstream'] = df[ID].apply(lambda x: immediate_upstream[x] if x in immediate_upstream else [])

    # Find the maximum length of the lists in the 'upstream' column
    df['maxup'] = 0
    df['maxup'] = df['upstream'].apply(len)

    # Create new columns 'maxup', 'up1', 'up2', 'up3', etc.
    max_length = df['maxup'].max()
    if max_length > 0:
        for i in range(max_length):
            df[f'up{i + 1}'] = df['upstream'].apply(lambda x: x[i] if i < len(x) else 0)
    else:
        print('It seems there is no upstream segment for the provided river network. '+\
              'This may mean the river network you are working may have first order rivers '+\
              'that are not connected.')

    # drop upstream
    df = df.drop(columns = 'upstream')

    return df

def longest_branch(
    riv: gpd.GeoDataFrame,
    main_id: Union[str, int] = None,
    ds_main_id: Union[str, int] = None,
) -> List[int]:
    """Returns nodes of the longest branch of a river network

    Parameters
    ----------
    riv : gpd.GeoDataFrame
        a geopandas.GeoDataFrame object containing the geometry,
        `main_id`, and `ds_main_id` of a river network
    main_id : str or int, defaults to `None`
        column label within `riv` corresponding to ID values of river
        segments
    ds_main_id : str or int, defaults to `None`
        column label within `riv` corresponding to downstream segments of
        each river

    Returns
    -------
    nodes : set
        a set object containing river segment `main_id` calues of the
        longest branch found in `riv`
    """
    riv_graph = nx.from_pandas_edgelist(riv,
                                        source=main_id,
                                        target=ds_main_id,
                                        create_using=nx.DiGraph)

    nodes = nx.dag_longest_path(riv_graph)

    return nodes

def _adjust_ids(
    seg_id: np.ndarray,
    ds_seg_id: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    [Temporary solution]: readjusts segments (and therefore downstream
    segments) to allow all segments to be in between the nodes of the
    longest branch found in a given river network
    """
    # function limited names for the segments and downstream
    # river segments
    main_id_str = 'main_id'
    ds_main_id_str = 'ds_main_id'

    # creating a pandas DataFrame of `seg_id` and `ds_seg_id`
    riv_df = pd.concat([pd.Series(arr) for arr in [seg_id, ds_seg_id]],
                       names=[main_id_str, ds_main_id_str],
                       axis=1)
    # naming columns, in case needed
    riv_df.columns = [main_id_str, ds_main_id_str]

    # extracting the longest branch out of "hydrant"
    longest_branch_out = longest_branch(riv=riv_df,
                                    main_id=main_id_str,
                                    ds_main_id=ds_main_id_str)

    # selecting first and last (while not counting the outlet value, so -2)
    first_node = longest_branch_out[0]
    last_node = longest_branch_out[-2]

    # extracting the index of first and last nodes
    first_idx = riv_df.index[riv_df[main_id_str] == first_node]
    last_idx = riv_df.index[riv_df[main_id_str] == last_node]

    # building new `riv_df` with new index values
    idx = pd.Index(first_idx.to_list() + riv_df.index.to_list())
    idx = idx.drop_duplicates(keep='first')
    idx = pd.Index(idx.to_list() + last_idx.to_list())
    idx = idx.drop_duplicates(keep='last')
    riv_df = riv_df.loc[idx]

    # reseting index, just to reassure
    riv_df.reset_index(drop=True, inplace=True)

    # extracting np.ndarrays
    new_seg_id = riv_df[main_id_str].to_numpy()
    new_ds_seg_id = riv_df[ds_main_id_str].to_numpy()

    return new_seg_id, new_ds_seg_id


def extract_rank(ntopo,
                 mapping: Dict[str, str] = {'id':'COMID','next_id':'NextDownID'},
                 outlet_value: int = -9999):
    '''Producing rank_var and next_var variables needed for
    MESH modelling

    Parameters
    ----------
    seg : array-like or list-like
        The ordered list (or array) of segment IDs corresponding
        to river reaches presented in an area of interest
    ds_seg : array-like or list-like
        The ordered list (or array) of downstream segment IDs
        corresponding to those of `seg_id` elements
    outlet_value : int, [defaults to -9999]
        The outlet value assigned to `to_segment` indicating
        sinking from the system


    Returns
    -------
    rank_var: numpy.array of int
        The 'rank_var' of each segment ID produced based on
        MESH modelling standards
    next_var: numpy.array of int
        The 'next_var' variable indicating the downstream segment
        of river reaches corresponding to 'rank_var'
    seg_id: numpy.array of int
        The 'seg_id' that has been reordered to match values of
        `rank_var` and `next_var`.
    to_segment: numpy.array of int
        The 'to_segment' that has been reordered to match values
        of `rank_var` and `next_var`.


    Notes
    -----
    The function is mainly developed by Dr. Ala Bahrami at
    <ala.bahrami@usask.ca> and Cooper Albano <cooper.albano@usask.ca>
    as part of the North American MESH model workflow development.
    Minor changes have been implemented by Kasra Keshavarz
    <kasra.keshavarz1@ucalgary.ca>.

    The original workflow is located at the following link:
    https://github.com/MESH-Model/MESH-Scripts
    <last accessed on August 29th, 2023>
    '''

    # # sort based on up area
    # ntopo = ntopo.sort_values(by = 'uparea')
    # ntopo = ntopo.reset_index(drop = True)
    # ntopo['NextDownID'].iloc[38715] = -9999
    # print(ntopo)

    # ID column
    ID_column = mapping.get('id')
    ID_next_column = mapping.get('next_id')

    # extracting numpy array out of input iterables
    seg_arr = np.array(ntopo[ID_column])
    ds_seg_arr = np.array(ntopo[ID_next_column])

    # print(seg_arr)
    # print(ds_seg_arr)

    # re-order ids to match MESH's requirements
    seg_id, to_segment = _adjust_ids(seg_arr, ds_seg_arr)

    # Count the number of outlets
    outlets = np.where(to_segment == outlet_value)[0]

    # Search over to extract the subbasins drain into each outlet
    rank_var_id_domain = np.array([]).astype(int)
    outlet_number = np.array([]).astype(int)

    for k in range(len(outlets)):
        # initial step
        seg_id_target = seg_id[outlets[k]]
        # set the rank_var of the outlet
        rank_var_id = outlets[k]

        # find upstream seg_ids draining into the chosen outlet [indexed `k`]
        while (np.size(seg_id_target) >= 1):
            if (np.size(seg_id_target) == 1):
                r = np.where(to_segment == seg_id_target)[0]
            else:
                r = np.where(to_segment == seg_id_target[0])[0]
            # updated the target seg_id
            seg_id_target = np.append(seg_id_target, seg_id[r])
            # remove the first searched target
            seg_id_target = np.delete(seg_id_target, 0, 0)
            if (len(seg_id_target) == 0):
                break
            # update the rank_var_id
            rank_var_id = np.append(rank_var_id, r)
        rank_var_id = np.flip(rank_var_id)
        if (np.size(rank_var_id) > 1):
            outlet_number = np.append(outlet_number,
                                      (k)*np.ones((len(rank_var_id), 1)).astype(int))
        else:
            outlet_number = np.append(outlet_number, (k))
        rank_var_id_domain = np.append(rank_var_id_domain, rank_var_id)
        rank_var_id = []

    # reorder seg_id and to_segment
    seg_id = seg_id[rank_var_id_domain]
    to_segment = to_segment[rank_var_id_domain]

    # rearrange outlets to be consistent with MESH outlet structure
    # In MESH outlets should be placed at the end of the `NEXT` variable
    na = len(rank_var_id_domain)
    fid1 = np.where(to_segment != outlet_value)[0]
    fid2 = np.where(to_segment == outlet_value)[0]
    fid = np.append(fid1, fid2)

    rank_var_id_domain = rank_var_id_domain[fid]
    seg_id = seg_id[fid]
    to_segment = to_segment[fid]
    outlet_number = outlet_number[fid]

    # construct rank_var and next_var variables
    next_var = np.zeros(na).astype(np.int32)

    for k in range(na):
        if (to_segment[k] != outlet_value):
            r = np.where(to_segment[k] == seg_id)[0] + 1
            next_var[k] = r
        else:
            next_var[k] = 0

    # Construct rank_var from 1:na
    rank_var = np.arange(1, na+1).astype(np.int32)


    # pass the varibales into a dataframe
    df = pd.DataFrame()

    # Add additional variables to the DataFrame
    df['Rank'] = rank_var
    df['Rank_Next'] = next_var
    df['seg_id'] = seg_id
    df['to_segment'] = to_segment

    # sort df based on seg_id
    df = df.sort_values(by='seg_id')
    df = df.reset_index(drop=True)

    #print(df)

    # sort the input based on the seg_id also
    ntopo = ntopo.sort_values(by = ID_column)
    ntopo = ntopo.reset_index(drop=True)

    #print(ntopo)

    # Check if 'seg_id' and 'id' columns have exactly the same values after sorting
    ids_equal = sum (np.array(df['seg_id']) - np.array(ntopo[ID_column]) )

    if ids_equal != 0:
        sys.exit("The 'seg_id' in df and 'id' in ntopo do not have the same values after sorting.")

    # sort based on Rank and Next index
    ntopo['Rank'] = df['Rank']
    ntopo['Rank_Next'] = df['Rank_Next']
    ntopo = ntopo.sort_values(by = 'Rank')
    ntopo = ntopo.reset_index(drop = True)
    # Convert columns to specified data types
    data_types = {
        'Rank': int,
        'Rank_Next': int
    }
    ntopo = ntopo.astype(data_types)

    return ntopo

