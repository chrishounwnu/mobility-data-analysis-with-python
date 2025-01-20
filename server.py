"""Flower server example."""
import os
import argparse
from copy import deepcopy

import torch
import flwr as fl
import pandas as pd 

import models as ml
import train as tr
import strategy as st
import helper as hl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='FedVRF Aggregation Server')
    # Arguments from CML script
    parser.add_argument('--bi', help='Use Bidirectional LSTM', action='store_true')
    parser.add_argument('--dspeed', help='Use Rate of Speed', action='store_true')
    parser.add_argument('--dcourse', help='Use Rate of Course', action='store_true')
    parser.add_argument('--shiptype', help='Use AIS Shiptype', action='store_true')
    parser.add_argument('--length', help='Rolling Window Length (default: 1024)', default=1024, type=int, required=False)
    parser.add_argument('--stride', help='Rolling Window Stride (default: 1024)', default=1024, type=int, required=False)
    parser.add_argument('--max_dt', help='Maximum $\Delta t$ threshold (default:1800 sec.)', default=1800, type=int, required=False)
    # Arguments related to FL connection/aggregation
    parser.add_argument('--silos', help='#Data Silos (default: 3)', default=3, type=int, required=False)
    parser.add_argument('--fraction_fit', help='#clients to train per round (%%)', default=1.0, type=float, required=False)
    parser.add_argument('--fraction_eval', help='#clients to evaluate per round (%%)', default=1.0, type=float, required=False)
    parser.add_argument('--num_rounds', help='#FL Rounds (default: 170)', default=70, type=int, required=False)
    parser.add_argument('--load_check', help='Continue from Latest Epoch', action="store_true")
    parser.add_argument('--port', help='Server Port', default=8080, type=int, required=False)
    parser.add_argument('--mu', help='Proximal $\mu$', default=0.01, type=float, required=False)
    args = parser.parse_args()
    
    params = dict(
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

    if args.shiptype:
        shiptype_token_lookup = pd.read_pickle(f'./data/pkl/shiptype_token_lookup_v3.pkl')
        shiptype_embeddings = torch.nn.Embedding(len(shiptype_token_lookup), len(shiptype_token_lookup)//2) 
    
    model_params = dict(
        **dict(embedding=shiptype_embeddings) if args.shiptype else {},
        input_size=len(params['input_feats']),
        scale=None,
        bidirectional=args.bi,
        num_layers=1,
        hidden_size=350,
        fc_layers=[150,]
    )

    model = ml.ShipTypeVRF(**model_params) if args.shiptype else ml.VesselRouteForecasting(**model_params)
    
    model_name = f'{"bi-" if model_params["bidirectional"] else ""}'+\
                 f'lstm_{model_params["num_layers"]}_'+\
                 f'{model_params["hidden_size"]}_fc_{"_".join(map(str, model_params["fc_layers"]))}_{"share_all"}_'+\
                 f'window_{params["length_max"]}_stride_{params["stride"]}_'+\
                 f'{"dspeed" if args.dspeed else ""}_'+\
                 f'{"dcourse" if args.dcourse else ""}_'+\
                 f'{"shiptype" if args.shiptype else ""}_'+\
                 f'_timeseries_split_'+\
                 '.dropout_after_cat.flwr_global.epoch{0}.pth'
    
    save_path = os.path.join('.', 'data', 'pth', f'fl_experiments_v202402__maxdt={args.max_dt}_silos={args.silos}_fraction_fit={args.fraction_fit}_fraction_eval={args.fraction_eval}_proximal_mu={args.mu}', model_name)
    print(save_path)

    if args.load_check:
        print('Loading Latest Checkpoint...')
        model_params = torch.load(save_path.format(args.num_rounds))
        model.load_state_dict(model_params['model_state_dict'])

    print('Using ```FedProx``` Aggregation Strategy')
    strategy = st.FedProxVRF(
        model=deepcopy(model),
        save_path=save_path,
        num_rounds=args.num_rounds,
        load_check=args.load_check,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_eval,
        min_fit_clients=int(args.silos * args.fraction_fit),
        min_evaluate_clients=args.silos,
        min_available_clients=args.silos,
        initial_parameters=fl.common.ndarrays_to_parameters(hl.get_parameters(model)),
        proximal_mu=args.mu
    )

    fl.server.start_server(
        server_address=f"[::]:{args.port}", 
        config=fl.server.ServerConfig(num_rounds=args.num_rounds), 
        strategy=strategy
    )
