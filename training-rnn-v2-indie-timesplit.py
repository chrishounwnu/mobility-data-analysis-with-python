#!/usr/bin/env python
# coding: utf-8

# In[1]:
# import pdb
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import os.path
import sys

import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn.preprocessing import StandardScaler

# In[3]:

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Subset
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

torch.manual_seed(10)
torch.autograd.set_detect_anomaly(True)

# In[4]:
import dataset as ds
import models as ml
import train as tr
import helper as hl


# %%
#   * #### Create delta series
def create_delta_dataset(segment, time_name, speed_name, course_name, crs=3857, min_pts=22):
    if len(segment) < min_pts:
        return None

    segment.sort_values(time_name, inplace=True)
    
    delta_curr = segment.to_crs(crs)[segment.geometry.name].apply(lambda l: pd.Series(hl.shapely_coords_numpy(l), index=['dlon', 'dlat'])).diff()
    delta_curr_feats = segment[[speed_name, course_name]].diff().rename({speed_name:'dspeed_curr', course_name:'dcourse_curr'}, axis=1)
    delta_next = delta_curr.shift(-1)
    delta_tau  = pd.merge(
        segment[time_name].diff().rename('dt_curr'),
        segment[time_name].diff().shift(-1).rename('dt_next'),
        right_index=True, 
        left_index=True
    )
    
    return delta_curr.join(delta_curr_feats).join(delta_tau).join(delta_next, lsuffix='_curr', rsuffix='_next').dropna(subset=['dt_curr', 'dt_next']).fillna(method='bfill')


# %%
#   * #### Create constant-length windows for ML model training
def traj_windowing(
    segment, 
    length_max=1024, 
    length_min=20,
    stride=512, 
    input_feats=['dlon_curr', 'dlat_curr', 'dt_curr', 'dt_next'], 
    output_feats=['dlon_next', 'dlat_next'], 
):
    traj_inputs, traj_labels = [], []
    
    # input_feats_idx = [segment.columns.get_loc(input_feat) for input_feat in input_feats]
    output_feats_idx = [segment.columns.get_loc(output_feat) for output_feat in output_feats]
        
    for ptr_curr in range(0, len(segment), stride):
        segment_window = segment.iloc[ptr_curr:ptr_curr+length_max].copy()     

        if len(segment_window) < length_min:
            break

        traj_inputs.append(segment_window[input_feats].values)
        traj_labels.append(segment_window.iloc[-1, output_feats_idx].values)
    
    return pd.Series([traj_inputs, traj_labels], index=['samples', 'labels'])


# In[9]:
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Centralized ("Share-All") VRF Worker')
    parser.add_argument('--data', help='Select Dataset', choices=['brest_1800', 'brest_3600', 'piraeus_1800', 'piraeus_3600', 'mt_1800', 'mt_3600'], type=str, required=True)
    parser.add_argument('--gpuid', help='GPU ID', default=0, type=int, required=False)
    parser.add_argument('--njobs', help='#CPUs', default=-1, type=int, required=False)
    parser.add_argument('--crs', help='Dataset CRS (default: 3857)', default=3857, type=int, required=False)
    parser.add_argument('--bi', help='Use Bidirectional LSTM', action='store_true')
    parser.add_argument('--dspeed', help='Use Rate of Speed', action='store_true')
    parser.add_argument('--dcourse', help='Use Rate of Course', action='store_true')
    parser.add_argument('--shiptype', help='Use AIS Shiptype', action='store_true')
    parser.add_argument('--bs', help='Batch Size', default=1, type=int, required=False)
    parser.add_argument('--length', help='Rolling Window Length (default: 1024)', default=1024, type=int, required=False)
    parser.add_argument('--stride', help='Rolling Window Stride (default: 1024)', default=1024, type=int, required=False)
    parser.add_argument('--patience', help='Patience (#Epochs) for Early Stopping (default: 10)', type=int)
    parser.add_argument('--max_dt', help='Maximum $\Delta t$ threshold (default:1800 sec.)', default=1800, type=int, required=False)
    parser.add_argument('--skip_train', help='Skip training; Evaluate best model @ Test Set', action='store_true')
    args = parser.parse_args()


    #   * #### Drop invalid MMSIs
    mmsi_mid = pd.read_pickle('mmsi_mid.pickle')


    if args.data == 'brest_1800' or args.data == 'brest_3600':
        # BREST
        trajectories = pd.read_csv(f'./data/brest-dataset/10_sec__{args.max_dt}_sec/dataset_trajectories_preprocessed_with_type.fixed.csv', parse_dates=['timestamp'])
        print(f'[Loaded] Dataset AIS Positions: {len(trajectories)}')
        VESSEL_NAME, SPEED_NAME, COURSE_NAME, TIME_NAME, TYPE_NAME = 'sourcemmsi', 'speedoverground', 'courseoverground', 't', 'shiptype'

        trajectories_mmsis = trajectories[VESSEL_NAME].unique()
        valid_mmsis = [mmsi for mmsi in trajectories_mmsis if mmsi//10**6 in mmsi_mid.MID.values]
        trajectories = trajectories.loc[trajectories[VESSEL_NAME].isin(valid_mmsis)].copy()
        print(f'[Invalid MIDs] Dataset AIS Positions: {len(trajectories)}')
    
    elif args.data == 'piraeus_1800' or args.data == 'piraeus_3600':
        # PIRAEUS
        trajectories = pd.read_csv(f'./data/piraeus-dataset/10_sec__{args.max_dt}_sec/dataset_trajectories_preprocessed_with_type.fixed.csv', parse_dates=['timestamp'])
        print(f'[Loaded] Dataset AIS Positions: {len(trajectories)}')
        VESSEL_NAME, SPEED_NAME, COURSE_NAME, TIME_NAME, TYPE_NAME = 'vessel_id', 'speed', 'course', 'timestamp_sec', 'shiptype'
        # pdb.set_trace()

    elif args.data == 'mt_1800' or args.data == 'mt_3600':
        # MARINETRAFFIC
        trajectories = pd.read_csv(f'./data/mt-dataset/10_sec__{args.max_dt}_sec/dataset_trajectories_preprocessed_with_type.fixed.csv', parse_dates=['timestamp'])
        print(f'[Loaded] Dataset AIS Positions: {len(trajectories)}')
        VESSEL_NAME, SPEED_NAME, COURSE_NAME, TIME_NAME, TYPE_NAME = 'mmsi', 'speed', 'course', 'timestamp_sec', 'shiptype'

        trajectories_mmsis = trajectories[VESSEL_NAME].unique()
        valid_mmsis = [mmsi for mmsi in trajectories_mmsis if mmsi//10**6 in mmsi_mid.MID.values]
        trajectories = trajectories.loc[trajectories.mmsi.isin(valid_mmsis)].copy()
        print(f'[Invalid MIDs] Dataset AIS Positions: {len(trajectories)}')


    if args.shiptype:
        vessel_shiptypes = trajectories.groupby(VESSEL_NAME)[TYPE_NAME].unique().apply(lambda l: l[0])
        vessel_shiptypes.to_pickle(f'./data/pkl/exp_study/{args.data}_vessel_shiptypes_v3.pkl')   # Saving vessel shiptypes for future reference
        shiptype_token_lookup = pd.read_pickle(f'./data/pkl/shiptype_token_lookup_v3.pkl')
        shiptype_embeddings = nn.Embedding(len(shiptype_token_lookup), len(shiptype_token_lookup)//2) 


    trajectories = gpd.GeoDataFrame(trajectories, crs=4326, geometry=gpd.points_from_xy(trajectories['lon'], trajectories['lat']))

    # Temporal train/dev/test split; 50/25/25 (e.g., 3mos, ~1.5mos will be used for train and ~0.75mos will be used for validation and testing, respectively)
    trajectories_dates = trajectories.timestamp.dt.date.sort_values().unique()
    train_dates, dev_dates, test_dates = ds.timeseries_train_test_split(trajectories_dates, dev_size=0.25, test_size=0.25, shuffle=False)
    

    print(
        f'Train @{(min(trajectories_dates[train_dates]), max(trajectories_dates[train_dates]))=};'+\
        f'\nDev @{(min(trajectories_dates[dev_dates]), max(trajectories_dates[dev_dates]))=};'+\
        f'\nTest @{(min(trajectories_dates[test_dates]), max(trajectories_dates[test_dates]))=}'
    )


    trajectories.loc[trajectories['timestamp'].dt.date.isin(trajectories_dates[train_dates]), 'dataset_tr1_val2_test3'] = 1
    trajectories.loc[trajectories['timestamp'].dt.date.isin(trajectories_dates[dev_dates]), 'dataset_tr1_val2_test3'] = 2
    trajectories.loc[trajectories['timestamp'].dt.date.isin(trajectories_dates[test_dates]), 'dataset_tr1_val2_test3'] = 3

    params = dict(
        time_name=TIME_NAME, 
        speed_name=SPEED_NAME, 
        course_name=COURSE_NAME, 
        crs=args.crs, 
        min_pts=20
    )

    print(f"Sanity Check #1;\n\t{trajectories.groupby([VESSEL_NAME, 'id', 'dataset_tr1_val2_test3'])['timestamp'].is_monotonic_increasing.all()=}")

    # # Create VRF training dataset
    traj_delta = hl.applyParallel(
        trajectories.groupby([VESSEL_NAME, 'id', 'dataset_tr1_val2_test3'], group_keys=True), 
        lambda l: create_delta_dataset(l, **params),
        n_jobs=args.njobs
    )

    windowing_params = dict(
        length_min=18, 
        length_max=args.length, 
        stride=args.stride,
        input_feats=[
            'dlon_curr', 
            'dlat_curr', 
            *(['dspeed_curr',] if args.dspeed else []),
            *(['dcourse_curr',] if args.dcourse else []),
            'dt_curr',
            'dt_next'
        ]
    )
    print(f'{windowing_params["input_feats"]=}')

    traj_delta_windows = hl.applyParallel(
        traj_delta.reset_index().groupby([VESSEL_NAME, 'id', 'dataset_tr1_val2_test3']),
        lambda l: traj_windowing(l, **windowing_params),
        n_jobs=args.njobs
    ).reset_index(level=-1)\
        .pivot(columns=['level_3'])\
        .rename_axis([None, None], axis=1)\
        .sort_index(axis=1, ascending=False)

    # pdb.set_trace()
    traj_delta_windows.columns = traj_delta_windows.columns.droplevel(0)
    traj_delta_windows = traj_delta_windows.explode(['samples', 'labels'])

    traj_delta_windows.to_pickle(
        f'./data/pkl/exp_study/{args.data}_dataset_'+\
        f'window_{windowing_params["length_max"]}_stride_{windowing_params["stride"]}_crs_{args.crs}_'+\
        f'{"dspeed" if args.dspeed else ""}_'+\
        f'{"dcourse" if args.dcourse else ""}.traj_delta_windows.pickle'
    )

    #   * #### Split trajectories train/dev/test sets
    bins = np.arange(0, args.max_dt+1, 300)
    look_discrete = pd.cut(traj_delta_windows.samples.apply(lambda l: l[-1, -1]), bins=bins).rename('labels')

    #   * #### Visualize Train/Dev/Test Distribution
    fig, ax = plt.subplots(1,3, figsize=(20, 7))
    # pdb.set_trace()
    look_discrete.xs(1, level=2).value_counts(sort=False).plot.bar(ax=ax[0], color='tab:blue')
    look_discrete.xs(2, level=2).value_counts(sort=False).plot.bar(ax=ax[1], color='tab:orange')
    look_discrete.xs(3, level=2).value_counts(sort=False).plot.bar(ax=ax[2], color='tab:green')

    [ax_i.set_yscale('log') for ax_i in ax];
    [ax_i.bar_label(ax_i.containers[0]) for ax_i in ax];
    plt.savefig(
        f'./data/fig/exp_study/delta_series_timeseries_split_lookahead_distribution_{args.data}_'+\
        f'window_{windowing_params["length_max"]}_stride_{windowing_params["stride"]}_crs_{args.crs}_'+\
        f'{"dspeed" if args.dspeed else ""}_'+\
        f'{"dcourse" if args.dcourse else ""}_'+\
        f'.pdf',
        dpi=300, 
        bbox_inches='tight'
    )

    #   * #### Create unified train/dev/test dataset(s)
    train_delta_windows = traj_delta_windows.xs(1, level=2).copy()
    dev_delta_windows = traj_delta_windows.xs(2, level=2).copy()
    test_delta_windows = traj_delta_windows.xs(3, level=2).copy()

    # # Create kinematic features' temporal sequence (i.e. training dataset)
    if args.shiptype:
        train_dataset = ds.VRFDataset_LE(vessel_shiptypes, shiptype_token_lookup, train_delta_windows)
        dev_dataset, test_dataset = ds.VRFDataset_LE(vessel_shiptypes, shiptype_token_lookup, dev_delta_windows, scaler=train_dataset.scaler),\
                                    ds.VRFDataset_LE(vessel_shiptypes, shiptype_token_lookup, test_delta_windows, scaler=train_dataset.scaler)
    else:
        train_dataset = ds.VRFDataset(train_delta_windows)
        dev_dataset, test_dataset = ds.VRFDataset(dev_delta_windows, scaler=train_dataset.scaler),\
                                    ds.VRFDataset(test_delta_windows, scaler=train_dataset.scaler)

    train_loader, dev_loader, test_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, collate_fn=train_dataset.pad_collate),\
                                            DataLoader(dev_dataset,   batch_size=args.bs, shuffle=False, collate_fn=dev_dataset.pad_collate),\
                                            DataLoader(test_dataset,  batch_size=args.bs, shuffle=False, collate_fn=test_dataset.pad_collate)


    # In[12]:
    device = torch.device(f'cuda:{args.gpuid}') if torch.cuda.is_available() else torch.device('cpu')

    model_params = dict(
        **dict(embedding=shiptype_embeddings) if args.shiptype else {},
        input_size=len(windowing_params['input_feats']),
        scale=dict(
            sigma=torch.Tensor(train_dataset.scaler.scale_[:2]), 
            mu=torch.Tensor(train_dataset.scaler.mean_[:2])
        ),
        bidirectional=args.bi,
        num_layers=1,
        hidden_size=350,
        fc_layers=[150,]
    )

    model = ml.ShipTypeVRF(**model_params) if args.shiptype else ml.VesselRouteForecasting(**model_params)
    model.to(device)

    print(model)
    print(f'{device=}')

    criterion = tr.RMSELoss(eps=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model_name_base = f'{"bi-" if model_params["bidirectional"] else ""}'+\
                f'lstm_{model_params["num_layers"]}_'+\
                f'{model_params["hidden_size"]}_fc_{"_".join(map(str, model_params["fc_layers"]))}_{"share_all"}_'+\
                f'window_{windowing_params["length_max"]}_stride_{windowing_params["stride"]}_crs_{args.crs}_'+\
                f'{"dspeed" if args.dspeed else ""}_'+\
                f'{"dcourse" if args.dcourse else ""}_'+\
                f'{"shiptype" if args.shiptype else ""}_'+\
                f'batchsize_{args.bs}_'+\
                f'patience_{args.patience}__'+\
                f'{args.data}_dataset__timeseries_split_.'+\
                'dropout_after_cat.sn_cml.epoch{0}.pth'
    
    save_path_epoch = os.path.join('.', 'data', 'pth', f'{args.data}', model_name_base)
    save_path_best = os.path.join('.', 'data', 'pth', f'{args.data}', model_name_base.format('best'))
    print(save_path_epoch)

    evaluate_fun_params = dict(
        bins=np.arange(0, args.max_dt+1, 300)
    )

    early_stop_params = dict(
        patience=args.patience,
        save_best=True,
        path=save_path_best
    )

    save_current_params = dict(
        path=save_path_epoch
    )

    if not args.skip_train:
        tr.train_model(
            model, device, criterion, optimizer, 100, 
            train_loader, dev_loader, early_stop=True, save_current=True, 
            evaluate_fun=tr.vrf_evaluate_model_singlehead, evaluate_fun_params=evaluate_fun_params,
            early_stop_params=early_stop_params, save_current_params=save_current_params
        )

    # %%
    ## Load best model and test its accuracy on the test set
    checkpoint = torch.load(save_path_best)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    tr.vrf_evaluate_model_singlehead(model, device, criterion, test_loader, desc='ADE @ Test Set...', **evaluate_fun_params)
