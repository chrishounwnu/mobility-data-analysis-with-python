# %%
import numpy as np
import pandas as pd
import geopandas as gpd

import st_toolkit as stt

import os
import time
import argparse
from tqdm.auto import tqdm


CFG_EPS = 1e-9
UNSPECIFIED_SHIPTYPE_TOKEN = '<UNK>'


def subsample_trajectory(traj, time_name, min_dt_sec=30):
    # Forked from: https://stackoverflow.com/a/56904899
    sumlm = np.frompyfunc(lambda a,b: a+b if a < min_dt_sec else b, 2, 1)

    traj_dt = traj[time_name].diff()
    traj_dt_sumlm = sumlm.accumulate(traj_dt, dtype=int)

    return traj.drop(traj_dt_sumlm.loc[traj_dt_sumlm < min_dt_sec].index)


def discretize_values(value, lookup, unk_token, max_ais_code=99):
    if 10 <= value <= max_ais_code:
        return lookup.loc[value, 'description']
    elif value > max_ais_code:
        return 'other' 
    return unk_token


# In[9]:
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Centralized ("Share-All") VRF Worker')
    parser.add_argument('--data', help='Select Dataset', choices=['brest', 'norway', 'piraeus', 'mt'], type=str, required=True)
    parser.add_argument('--min_dt', help='Minimum $\Delta t$ threshold (default:10 sec.)', default=10, type=int, required=False)
    parser.add_argument('--max_dt', help='Maximum $\Delta t$ threshold (default:1800 sec.)', default=1800, type=int, required=False)
    parser.add_argument('--min_speed', help='Minimum speed threshold (stationaries; default:1 knot)', default=1, type=int, required=False)
    parser.add_argument('--max_speed', help='Maximum speed threshold (outliers; default:50 knots)', default=50, type=int, required=False)
    parser.add_argument('--min_pts', help='Minimum points threshold for constructing a trajectory (default:20 points)', default=20, type=int, required=False)
    parser.add_argument('--shiptype', help='Include shiptype', action='store_true', required=False)
    parser.add_argument('--njobs', help='#CPUs (default:200)', default=200, type=int, required=False)
    args = parser.parse_args()


    # Drop invalid MMSIs
    mmsi_mid = pd.read_pickle('./mmsi_mid.pickle')


    if args.data == 'brest':
        # BREST; CRS EPSG:2154
        CFG_ROOT = '~/data/brest-dataset'

        df = pd.read_csv(os.path.join(CFG_ROOT, 'nari_dynamic.csv'))
        print(f'[Raw Dataset] Dataset AIS Positions: {len(df)}')
        VESSEL_ID_NAME, SPEED_NAME, COURSE_NAME, TIMESTAMP_NAME, TIMESTAMP_UNIT, COORDS_NAMES, COORDS_CRS = 'sourcemmsi', 'speedoverground', 'courseoverground', 't', 's', ['lon', 'lat'], 4326

        # Drop Invalid MMSIs (based on their MID)
        time_invalidmmsis = time.time()
        df = df.loc[
            df[VESSEL_ID_NAME].astype(str).str.zfill(9).str[:3].isin(
                mmsi_mid.MID.astype(str)
            )
        ].copy()
        print(f'[Invalid MMSIs] Dataset AIS Positions: {len(df)}; Time elapsed: {time.time() - time_invalidmmsis}')

        df.loc[:, f'timestamp_datetime'] = pd.to_datetime(df.loc[:, TIMESTAMP_NAME], unit=TIMESTAMP_UNIT)
        df.loc[:, f'timestamp_sec'] = df.timestamp_datetime.astype(int) // 10**9

        # Fetch ```Static``` data
        df_static = pd.read_csv(os.path.join(CFG_ROOT, 'static', 'nari_static.csv'))
        df_static = df_static.sort_values(
            TIMESTAMP_NAME
        ).dropna(
            subset=['shiptype']
        ).drop_duplicates(
            subset=[VESSEL_ID_NAME], keep='last'
        )[[VESSEL_ID_NAME, 'shiptype']]


    elif args.data == 'piraeus':
        # PIRAEUS; CRS EPSG:2100
        CFG_ROOT = '~/data/piraeus-dataset'

        df = pd.read_csv(os.path.join(CFG_ROOT, 'saronic_jan_mar_2019.csv'))
        print(f'[Raw Dataset] Dataset AIS Positions: {len(df)}')
        VESSEL_ID_NAME, SPEED_NAME, COURSE_NAME, TIMESTAMP_NAME, TIMESTAMP_UNIT, COORDS_NAMES, COORDS_CRS = 'vessel_id', 'speed', 'course', 't', None, ['lon', 'lat'], 4326

        df.loc[:, f'timestamp_datetime'] = pd.to_datetime(df.loc[:, TIMESTAMP_NAME], unit=TIMESTAMP_UNIT)
        df.loc[:, f'timestamp_sec'] = df.timestamp_datetime.astype(int) // 10**9

        # Fetch ```Static``` data
        df_static = pd.read_csv(os.path.join(CFG_ROOT, 'ais_static', 'unipi_ais_static_anon.csv'))
        df_static = df_static.dropna(
            subset=['shiptype']
        )[[VESSEL_ID_NAME, 'shiptype']]


    elif args.data == 'mt':
        # MARINETRAFFIC; CRS EPSG:2100
        CFG_ROOT = '~/data/mt-dataset'
        VESSEL_ID_NAME, SPEED_NAME, COURSE_NAME, TIMESTAMP_NAME, TIMESTAMP_UNIT, COORDS_NAMES, COORDS_CRS = 'mmsi', 'speed', 'course', 'timestamp', None, ['lon', 'lat'], 4326

        df = pd.read_csv(os.path.join(CFG_ROOT, 'ais_kinematic_aegean_2018nov_distinct_types.csv'), sep=';', parse_dates=[TIMESTAMP_NAME])
        print(f'[Raw Dataset] Dataset AIS Positions: {len(df)}')

        # Crop dataset to Aegean Sea
        df = df.loc[
            (df[COORDS_NAMES[0]].between(24.4556, 26.5869, inclusive='both')) & (df[COORDS_NAMES[1]].between(36.0846, 39.4871, inclusive='both'))
        ].copy()
        print(f'[Area Restriction] Dataset AIS Positions: {len(df)}')

        # Drop Invalid MMSIs (based on their MID)
        time_invalidmmsis = time.time()
        df = df.loc[
            df[VESSEL_ID_NAME].astype(str).str.zfill(9).str[:3].isin(
                mmsi_mid.MID.astype(str)
            )
        ].copy()
        print(f'[Invalid MMSIs] Dataset AIS Positions: {len(df)}; Time elapsed: {time.time() - time_invalidmmsis}')

        df.rename({TIMESTAMP_NAME:'timestamp_datetime'}, axis=1, inplace=True)
        df.loc[:, f'timestamp_sec'] = df.timestamp_datetime.astype(int) // 10**9

        # Fetch ```Static``` data
        df_static = df.sort_values('timestamp_datetime')[[VESSEL_ID_NAME, 'shiptype']].drop_duplicates(subset=[VESSEL_ID_NAME])
        df.drop(['shiptype'], axis=1, inplace=True)


    # Subsample Trajectories \w $\Delta t_{min}$
    time_subsample = time.time()
    tqdm.pandas()
    df_clean = df.sort_values('timestamp_sec', kind='mergesort').groupby(VESSEL_ID_NAME).progress_apply(
        lambda l: subsample_trajectory(
            l.copy(), 'timestamp_sec', args.min_dt
        )
    ).reset_index(level=0, drop=True)
    print(f'[Subsampling] Dataset AIS Positions: {len(df_clean)}; Time elapsed: {time.time() - time_subsample}')
    print(f'{df_clean.sort_values("timestamp_sec").groupby(VESSEL_ID_NAME)["timestamp_sec"].diff().describe().astype(str)=}')


    # Drop trajectories \w less than $Points_{min}$ locations
    time_trajprune = time.time()
    vessels_points = df_clean[VESSEL_ID_NAME].value_counts() 

    df_clean = df_clean.loc[
        df_clean[VESSEL_ID_NAME].isin(
            vessels_points.loc[vessels_points > args.min_pts].index
        )
    ].copy()
    print(f'[Trajectory Pruning] Dataset AIS Positions: {len(df_clean)}; Time elapsed: {time.time() - time_trajprune}')


    # Re-calculate Speed and Course over Ground
    time_speedcourse = time.time()
    df_clean = gpd.GeoDataFrame(df_clean, geometry=gpd.points_from_xy(df_clean[COORDS_NAMES[0]], df_clean[COORDS_NAMES[1]]), crs=COORDS_CRS)
    df_clean.sort_index(inplace=True)

    # ## Speed
    df_clean = stt.add_speed(
        df_clean, o_id=[VESSEL_ID_NAME,], ts='timestamp_sec', speed=SPEED_NAME, geometry=df_clean.geometry.name, n_jobs=args.njobs
    )
    # ## Course
    df_clean = stt.add_course(
        df_clean, o_id=[VESSEL_ID_NAME,], ts='timestamp_sec', course=COURSE_NAME, geometry=df_clean.geometry.name, n_jobs=args.njobs
    )


    # Drop Speed Outliers (points \w speed outside the range [$Speed_{min}$, $Speed_{max}$])
    df_clean.drop(
        df_clean.loc[~df_clean[SPEED_NAME].between(args.min_speed, args.max_speed, inclusive='both')].index,
        axis=0,
        inplace=True
    )
    print(f'[Speed Outliers] Dataset AIS Positions: {len(df_clean)}; Time elapsed: {time.time() - time_speedcourse}')
    print(f'{df_clean[SPEED_NAME].describe().round(5).astype(str)=}')

    df_clean_coords_meters = stt.applyParallel(
        df_clean.sort_values('timestamp_sec').groupby(VESSEL_ID_NAME),
        lambda traj: traj.to_crs(3857)[traj.geometry.name].apply(
            lambda point: pd.Series(stt.shapely_coords_numpy(point), index=['lon_3857', 'lat_3857'])
        ),
        n_jobs=args.njobs,
        dynamic_ncols=True
    ).reset_index(level=0, drop=True)
    df_clean.loc[:, ['lon_3857', 'lat_3857']] = df_clean_coords_meters


    # Temporal Segmentation
    time_tempseg = time.time()
    kwargs = dict(
        col_name = 'timestamp_sec',
        threshold = args.max_dt + CFG_EPS,
        min_pts = args.min_pts,
        output_name = 'temp_traj_nr'
    )

    # Cut-off at 30 mins (1800 sec.)
    tqdm.pandas()
    # gdf_seg = df_clean_sigma.sort_values('timestamp_sec').groupby([VESSEL_ID_NAME,], group_keys=False).progress_apply(
    gdf_seg = df_clean.sort_values('timestamp_sec').groupby([VESSEL_ID_NAME,], group_keys=False).progress_apply(
        lambda l: stt.temporal_segmentation(
            l.copy(), **kwargs
        )
    )
    print(f'[Temporal Segmentation] Dataset AIS Positions: {len(gdf_seg)}; Time elapsed: {time.time() - time_tempseg}')

    gdf_seg.loc[:, 'traj_nr'] = gdf_seg.groupby([VESSEL_ID_NAME, 'temp_traj_nr']).ngroup()
    gdf_seg.groupby([VESSEL_ID_NAME, 'traj_nr']).apply(len).sort_values()

    # Save Results
    gdf_seg.drop(
        ['geometry'], axis=1
    ).rename(
        {'timestamp_datetime':'timestamp', 'traj_nr':'id'}, axis=1
    ).sort_values('timestamp_sec').to_csv(
        os.path.join(CFG_ROOT, 'dataset_trajectories_preprocessed.fixed.csv'),
        index=True, 
        header=True
    )


    if args.shiptype:
        # Fetch ```AIS Codes``` and prepare lookup table
        ais_codes = pd.read_csv('./data/ais_codes_types_extended_final.csv')
        ais_codes.drop(ais_codes.loc[ais_codes.shiptype_max > 99].index, inplace=True)      # Preprocessing: Keep only AIS-derived codes

        ais_codes[
            ['type_name', 'description', 'detailed_description']
        ] = ais_codes[
            ['type_name', 'description', 'detailed_description']
        ].apply(
            lambda l: l.str.lower()
        )     # Preprocessing: Convert all strings to "lowercase"; Eliminate inconsistencies

        ais_codes.replace({'unspecified':UNSPECIFIED_SHIPTYPE_TOKEN}, inplace=True)     # Preprocessing: Replace "Unspecified" with the "<UNK>" token
        ais_codes.loc[:, 'shiptype_max_open'] = ais_codes.shiptype_max + 1      # Preprocessing: Add +1 to max shiptype nr.; Use left-closed intervals

        ais_codes.loc[:, 'shiptype_cat'] = ais_codes.apply(lambda l: pd.Interval(l.shiptype_min, l.shiptype_max_open, closed='left'), axis=1)
        ais_codes.set_index('shiptype_cat', inplace=True)

        df_static.loc[:, 'shiptype_discrete'] = df_static.shiptype.apply(lambda l: discretize_values(l, ais_codes, UNSPECIFIED_SHIPTYPE_TOKEN))       # V3
        ais_static = df_static[[VESSEL_ID_NAME, 'shiptype_discrete']].rename({'shiptype_discrete':'shiptype'}, axis=1)
        
        gdf_seg_w_type = gdf_seg.set_index(VESSEL_ID_NAME).join(ais_static.set_index(VESSEL_ID_NAME)[['shiptype']])
        gdf_seg_w_type.shiptype.fillna(UNSPECIFIED_SHIPTYPE_TOKEN, inplace=True)

        print(f'[Shiptype Embeddings] Dataset AIS Positions: {len(gdf_seg_w_type)}')

        gdf_seg_w_type.drop(
            ['geometry'], axis=1
        ).rename(
            {'timestamp_datetime':'timestamp', 'traj_nr':'id'}, axis=1
        ).sort_values('timestamp_sec').to_csv(
            os.path.join(CFG_ROOT, 'dataset_trajectories_preprocessed_with_type.fixed.csv'),
            index=True, 
            header=True
        )
