import time
from collections import OrderedDict

import os

import pandas as pd
import numpy as np

import dataset as ds
import models as ml
import train as tr
import argparse

import torch
import torch.utils.data

torch.manual_seed(10)
torch.autograd.set_detect_anomaly(True)


def load_data(params):
    print(params)    
    data_path = f'./data/pkl/exp_study/{params["data"]}_dataset_'+\
                f'window_{params["length_max"]}_stride_{params["stride"]}_crs_{params["crs"]}_'+\
                f'{"dspeed" if params["dspeed"] else ""}_'+\
                f'{"dcourse" if params["dcourse"] else ""}.traj_delta_windows.pickle'

    ## Parse Train/Dev/Test sequences (from centralized script)
    traj_delta_windows, shiptype_embeddings = pd.read_pickle(data_path), None
    print(f'Loaded Trajectories from {os.path.basename(data_path)}')

    if params['shiptype']:
        vessel_shiptypes = pd.read_pickle(f'./data/pkl/exp_study/{params["data"]}_vessel_shiptypes_v3.pkl')
        shiptype_token_lookup = pd.read_pickle(f'./data/pkl/shiptype_token_lookup_v3.pkl')
        shiptype_embeddings = torch.nn.Embedding(len(shiptype_token_lookup), len(shiptype_token_lookup)//2) 

    train_delta_windows = traj_delta_windows.xs(1, level=2).copy()
    dev_delta_windows = traj_delta_windows.xs(2, level=2).copy()
    test_delta_windows = traj_delta_windows.xs(3, level=2).copy()

    ## Create Train/Dev/Test PyTorch Dataset
    if args.shiptype:
        train_dataset = ds.VRFDataset_LE(vessel_shiptypes, shiptype_token_lookup, train_delta_windows)
        dev_dataset, test_dataset = ds.VRFDataset_LE(vessel_shiptypes, shiptype_token_lookup, dev_delta_windows, scaler=train_dataset.scaler),\
                                    ds.VRFDataset_LE(vessel_shiptypes, shiptype_token_lookup, test_delta_windows, scaler=train_dataset.scaler)
    else:
        train_dataset = ds.VRFDataset(train_delta_windows)
        dev_dataset, test_dataset = ds.VRFDataset(dev_delta_windows, scaler=train_dataset.scaler),\
                                    ds.VRFDataset(test_delta_windows, scaler=train_dataset.scaler)

    ## Create Train/Dev/Test PyTorch DataLoader
    train_loader, dev_loader, test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['bs'], shuffle=True, collate_fn=train_dataset.pad_collate),\
                                            torch.utils.data.DataLoader(dev_dataset,   batch_size=params['bs'], shuffle=False, collate_fn=dev_dataset.pad_collate),\
                                            torch.utils.data.DataLoader(test_dataset,  batch_size=params['bs'], shuffle=False, collate_fn=test_dataset.pad_collate)


    print(train_dataset.scaler.mean_.round(5).astype(str), train_dataset.scaler.scale_.round(5).astype(str), sep='\t')

    return data_path, train_loader, dev_loader, test_loader, shiptype_embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Evaluate (Fed)Nautilus Models')
    # Arguments from CML script
    parser.add_argument('--data', help='Select Dataset', choices=['brest_1800', 'brest_3600', 'piraeus_1800', 'piraeus_3600', 'mt_1800', 'mt_3600'], type=str, required=True)
    parser.add_argument('--gpuid', help='GPU ID', default=0, type=int, required=False)
    parser.add_argument('--crs', help='Dataset CRS (default: 3857)', default=3857, type=int, required=False)
    parser.add_argument('--bi', help='Use Bidirectional LSTM', action='store_true')
    parser.add_argument('--dspeed', help='Use Rate of Speed', action='store_true')
    parser.add_argument('--dcourse', help='Use Rate of Course', action='store_true')
    parser.add_argument('--shiptype', help='Use AIS Shiptype', action='store_true')
    parser.add_argument('--bs', help='Batch Size', default=1, type=int, required=False)
    parser.add_argument('--length', help='Rolling Window Length (default: 1024)', default=1024, type=int, required=False)
    parser.add_argument('--stride', help='Rolling Window Stride (default: 1024)', default=1024, type=int, required=False)
    parser.add_argument('--max_dt', help='Maximum $\Delta t$ threshold (default:1800 sec.)', default=1800, type=int, required=False)
    parser.add_argument('--patience', help='Patience (#Epochs) for Early Stopping (default: 10)', type=int, default=10)
    # Arguments related to FL connection/aggregation
    parser.add_argument('--silos', help='#Data Silos', default=3, type=int, required=False)
    parser.add_argument('--fraction_fit', help='#clients to train per round (%%)', default=1.0, type=float, required=False)
    parser.add_argument('--fraction_eval', help='#clients to evaluate per round (%%)', default=1.0, type=float, required=False)
    # Parameters related to model evaluation
    parser.add_argument('--cml', action='store_true')
    parser.add_argument('--fl', action='store_true')
    parser.add_argument('--perfl', action='store_true')
    parser.add_argument('--global_ver', help='Verson of global model to load', default=70, type=int, required=False)
    parser.add_argument('--mu', help='Proximal $\mu$', default='0.0001,0.001,0.01,0.1,1.0', type=str, required=False)


    args = parser.parse_args()
    params = dict(
        # **vars(args),
        data=args.data,
        crs=args.crs,
        bs=args.bs,
        length_min=18, 
        length_max=args.length, 
        stride=args.stride,
        dspeed=args.dspeed,
        shiptype=args.shiptype,
        dcourse=args.dcourse,
        input_feats=[
            'dlon_curr', 
            'dlat_curr', 
            *(['dspeed_curr',] if args.dspeed else []),
            *(['dcourse_curr',] if args.dcourse else []),
            'dt_curr', 
            'dt_next'
        ]
    )

    data_path, train_loader, dev_loader, test_loader, shiptype_embeddings = load_data(params)
    device = torch.device(f'cuda:{args.gpuid}') if torch.cuda.is_available() else torch.device('cpu')

    model_params = dict(
        **dict(embedding=shiptype_embeddings) if args.shiptype else {},
        input_size=len(params['input_feats']),
        scale=dict(
            sigma=torch.Tensor(train_loader.dataset.scaler.scale_[:2]), 
            mu=torch.Tensor(train_loader.dataset.scaler.mean_[:2])
        ),
        bidirectional=args.bi,
        num_layers=1,
        hidden_size=350,
        fc_layers=[150,]
    )

    evaluate_fun_params = dict(
        bins=np.arange(0, args.max_dt+1, 300)
    )
    
    
    if args.cml:
        print('\t\tEvaluating Centralized VRF Model')
        model_name_base = f'{"bi-" if model_params["bidirectional"] else ""}'+\
                          f'lstm_{model_params["num_layers"]}_'+\
                          f'{model_params["hidden_size"]}_fc_{"_".join(map(str, model_params["fc_layers"]))}_{"share_all"}_'+\
                          f'window_{params["length_max"]}_stride_{params["stride"]}_crs_{args.crs}_'+\
                          f'{"dspeed" if args.dspeed else ""}_'+\
                          f'{"dcourse" if args.dcourse else ""}_'+\
                          f'{"shiptype" if args.shiptype else ""}_'+\
                          f'batchsize_{args.bs}_'+\
                          f'patience_{args.patience}__'+\
                          f'{args.data}_dataset__timeseries_split_'+\
                          '.dropout_after_cat.sn_cml.epoch{0}.pth'
        
        cml_model_dict = torch.load(os.path.join(f'./data/pth/{args.data}', model_name_base.format(f'best')), map_location=torch.device('cpu'))

        cml_model = ml.ShipTypeVRF(**model_params) if args.shiptype else ml.VesselRouteForecasting(**model_params)
        cml_model.load_state_dict(cml_model_dict['model_state_dict'])
        cml_model.to(device)

        cml_model.eval()
        tr.vrf_evaluate_model_singlehead(cml_model, device=device, criterion=tr.RMSELoss(eps=1e-4), test_loader=test_loader, **evaluate_fun_params)

    
    if args.fl:
        for mu in [float(x) for x in args.mu.split(',')]:
            global_model_dir = os.path.join(
                '.', 'data', 'pth', 
                f'fl_experiments_v202402__maxdt={args.max_dt}_silos={args.silos}_fraction_fit={args.fraction_fit}_fraction_eval={args.fraction_eval}_proximal_mu={mu}'
            )

            print(f'\t\tEvaluating Global FedVRF Model (\mu = {mu})')
            global_model_name = f'{"bi-" if model_params["bidirectional"] else ""}'+\
                                f'lstm_{model_params["num_layers"]}_'+\
                                f'{model_params["hidden_size"]}_fc_{"_".join(map(str, model_params["fc_layers"]))}_{"share_all"}_'+\
                                f'window_{params["length_max"]}_stride_{params["stride"]}_'+\
                                f'{"dspeed" if args.dspeed else ""}_'+\
                                f'{"dcourse" if args.dcourse else ""}_'+\
                                f'{"shiptype" if args.shiptype else ""}_'+\
                                f'_timeseries_split_'+\
                                '.dropout_after_cat.flwr_global.epoch{0}.pth'
            
            fl_model_dict = torch.load(os.path.join(global_model_dir, global_model_name.format(args.global_ver)), map_location=torch.device('cpu'))

            global_fl_model = ml.ShipTypeVRF(**model_params) if args.shiptype else ml.VesselRouteForecasting(**model_params)
            global_fl_model.load_state_dict(fl_model_dict['model_state_dict'])
            global_fl_model.to(device)

            global_fl_model.eval()
            tr.vrf_evaluate_model_singlehead(global_fl_model, device=device, criterion=tr.RMSELoss(eps=1e-4), test_loader=test_loader, **evaluate_fun_params)


    if args.perfl:
        for mu in [float(x) for x in args.mu.split(',')]:
            global_model_dir = os.path.join(
                '.', 'data', 'pth', 
                f'fl_experiments_v202402__maxdt={args.max_dt}_silos={args.silos}_fraction_fit={args.fraction_fit}_fraction_eval={args.fraction_eval}_proximal_mu={mu}'
            )

            print(f'\t\tEvaluating Personalized FedVRF Model (\mu = {mu})')
            local_model_name = f'{"bi-" if model_params["bidirectional"] else ""}'+\
                               f'lstm_{model_params["num_layers"]}_'+\
                               f'{model_params["hidden_size"]}_fc_{"_".join(map(str, model_params["fc_layers"]))}_{"share_all"}_'+\
                               f'window_{params["length_max"]}_stride_{params["stride"]}_crs_{args.crs}_'+\
                               f'{"dspeed" if args.dspeed else ""}_'+\
                               f'{"dcourse" if args.dcourse else ""}_'+\
                               f'{"shiptype" if args.shiptype else ""}_'+\
                               f'batchsize_{args.bs}_'+\
                               f'{args.data}_dataset__timeseries_split_'+\
                               '.dropout_after_cat.flwr_local.epoch{0}.pth'
        
            perfl_model_dict = torch.load(os.path.join(global_model_dir, local_model_name.format(f'best.personalized')), map_location=torch.device('cpu'))

            local_perfl_model = ml.ShipTypeVRF(**model_params) if args.shiptype else ml.VesselRouteForecasting(**model_params)
            local_perfl_model.load_state_dict(perfl_model_dict['model_state_dict'])
            local_perfl_model.to(device)

            local_perfl_model.eval()
            tr.vrf_evaluate_model_singlehead(local_perfl_model, device=device, criterion=tr.RMSELoss(eps=1e-4), test_loader=test_loader, **evaluate_fun_params)
