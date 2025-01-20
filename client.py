import time
from collections import OrderedDict

import os
import flwr as fl

import pandas as pd
import numpy as np

import helper as hl
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

    return data_path, train_loader, dev_loader, test_loader, shiptype_embeddings


class VRFClient(fl.client.NumPyClient):
    def __init__(self, model_params, device, save_path, train_loader, dev_loader, test_loader, load_check=False, evaluate_fun=tr.vrf_evaluate_model_singlehead, evaluate_fun_params={}):
        self.model = ml.ShipTypeVRF(**model_params) if args.shiptype else ml.VesselRouteForecasting(**model_params)
        self.device = device
        
        self.model.to(self.device)
        self.criterion = tr.RMSELoss(eps=1e-4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.evaluate_fun, self.evaluate_fun_params = evaluate_fun, evaluate_fun_params
        print(f'{self.device=}\n', self.model)

        self.train_loader, self.dev_loader, self.test_loader = train_loader, dev_loader, test_loader
        self.train_losses, self.dev_losses, self.num_examples = [], [], dict(
            training_set=len(self.train_loader.dataset),
            dev_set=len(self.dev_loader.dataset),
            test_set=len(self.test_loader.dataset)
        )
        self.save_path, self.fl_round = save_path, 0

        if load_check:
            print('Loading Latest Checkpoint...')
            model_params = torch.load(self.save_path, map_location=self.device)
            self.model.load_state_dict(model_params['model_state_dict'])
            self.optimizer.load_state_dict(model_params['optimizer_state_dict'])
            self.model.mu, self.model.sigma = torch.Tensor(model_params['scaler'].mean_[:2]),\
                                              torch.Tensor(model_params['scaler'].scale_[:2])
            self.train_losses = model_params['loss']
            self.dev_losses = model_params['dev_loss']
            self.fl_round = model_params['epoch']

    def get_parameters(self, **kwargs):
        return hl.get_parameters(self.model)

    def set_parameters(self, parameters):
        return hl.set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        # print(f'CHECK THIS OUT --> {config}')
        self.set_parameters(parameters)

        train_loss, dev_loss = tr.train_model(
            self.model, self.device, self.criterion, self.optimizer, n_epochs=1,
            train_loader=self.train_loader, dev_loader=self.dev_loader,
            evaluate_cycle=-1, early_stop=False, save_current=False,
            evaluate_fun=self.evaluate_fun, evaluate_fun_params=self.evaluate_fun_params
        )
        fit_tldr = dict(
            train_loss=float(train_loss[-1]),
            dev_loss=float(dev_loss[-1]),
        )
        self.train_losses.append(fit_tldr['train_loss'])
        self.dev_losses.append(fit_tldr['dev_loss'])

        # Save Current Client
        kwargs = dict({
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.train_losses,
            'dev_loss': self.dev_losses,
            'scaler': train_loader.dataset.scaler,
            'epoch': self.fl_round,
        })
        tr.save_model(self.model, self.save_path.format(self.fl_round), **kwargs)

        self.fl_round += 1
        return self.get_parameters(), self.num_examples['training_set'], fit_tldr

    def evaluate(self, parameters, config):
        # print(f'CHECK THIS OUT --> {config}')
        self.set_parameters(parameters)
        dev_loss, dev_acc = self.evaluate_fun(
            self.model, self.device, self.criterion, self.dev_loader,
            desc='[Flwr-Local] ADE @ Dev Set...', **self.evaluate_fun_params
        )
        eval_tldr = dict(
            dev_loss=float(dev_loss),
            dev_acc=float(np.mean(dev_acc)),
        )
        return np.float64(dev_loss), self.num_examples['dev_set'], eval_tldr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='FedVRF Local Worker')
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
    # Arguments related to FL connection/aggregation
    parser.add_argument('--load_check', help='Continue from Latest Epoch', action="store_true")
    parser.add_argument('--port', help='Server Port', default=8080, type=int, required=False)
    parser.add_argument('--silos', help='#Data Silos', default=4, type=int, required=False)
    parser.add_argument('--mu', help='Proximal $\mu$', default=0.01, type=float, required=False)
    parser.add_argument('--fraction_fit', help='#clients to train per round (%%)', default=1.0, type=float, required=False)
    parser.add_argument('--fraction_eval', help='#clients to evaluate per round (%%)', default=1.0, type=float, required=False)
    # Parameters related to FL personalization
    parser.add_argument('--personalize', help='Fine-tune the global model to the local clients\' data', action='store_true')
    parser.add_argument('--global_ver', help='Version of global model to load', default=70, type=int, required=False)
    parser.add_argument('--num_rounds', help='Number of epochs for fine-tuning', default=10, type=int, required=False)
    

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
        # fc_layers=[128,16,]
        fc_layers=[150,]
    )

    evaluate_fun_params = dict(
        bins=np.arange(0, args.max_dt+1, 300)
    )
    
    model_name = f'{"bi-" if model_params["bidirectional"] else ""}'+\
                f'lstm_{model_params["num_layers"]}_'+\
                f'{model_params["hidden_size"]}_fc_{"_".join(map(str, model_params["fc_layers"]))}_{"share_all"}_'+\
                f'window_{params["length_max"]}_stride_{params["stride"]}_crs_{args.crs}_'+\
                f'{"dspeed" if args.dspeed else ""}_'+\
                f'{"dcourse" if args.dcourse else ""}_'+\
                f'{"shiptype" if args.shiptype else ""}_'+\
                f'batchsize_{args.bs}_'+\
                f'{args.data}_dataset__timeseries_split_'+\
                '.dropout_after_cat.flwr_local.epoch{0}.pth'
    
    if not args.personalize:
        save_path = os.path.join('.', 'data', 'pth', f'fl_experiments_v202402__maxdt={args.max_dt}_silos={args.silos}_fraction_fit={args.fraction_fit}_fraction_eval={args.fraction_eval}_proximal_mu={args.mu}', model_name)
        print(save_path)

        client = VRFClient(
            model_params, device, save_path,
            train_loader, dev_loader, test_loader,
            args.load_check, evaluate_fun_params=evaluate_fun_params
        )
        print(f"[::]:{args.port}")
        fl.client.start_numpy_client(server_address=f"[::]:{args.port}", client=client)
    else:
        save_path_dir = os.path.join(
            '.', 'data', 'pth', 
            f'fl_experiments_v202402__maxdt={args.max_dt}_silos={args.silos}_fraction_fit={args.fraction_fit}_fraction_eval={args.fraction_eval}_proximal_mu={args.mu}'
        )

        global_model_name = f'{"bi-" if model_params["bidirectional"] else ""}'+\
                            f'lstm_{model_params["num_layers"]}_'+\
                            f'{model_params["hidden_size"]}_fc_{"_".join(map(str, model_params["fc_layers"]))}_{"share_all"}_'+\
                            f'window_{params["length_max"]}_stride_{params["stride"]}_'+\
                            f'{"dspeed" if args.dspeed else ""}_'+\
                            f'{"dcourse" if args.dcourse else ""}_'+\
                            f'{"shiptype" if args.shiptype else ""}_'+\
                            f'_timeseries_split_'+\
                            '.dropout_after_cat.flwr_global.epoch{0}.pth'
            
        fedvrf_global = torch.load(
            os.path.join(
                save_path_dir,
                global_model_name.format(args.global_ver)
            ),
            torch.device('cpu')
        )

        persn_model = ml.ShipTypeVRF(**model_params) if args.shiptype else ml.VesselRouteForecasting(**model_params)
        persn_model.load_state_dict(fedvrf_global['model_state_dict'])
        persn_model.to(device)

        print(f'Personalizing FedVRF on {args.data}...')
        criterion = tr.RMSELoss(eps=1e-4)
        optimizer = torch.optim.Adam(persn_model.parameters(), lr=1e-4)

        early_stop_params=dict(
            patience=3,
            save_best=True,
            path=os.path.join(save_path_dir, model_name.format('best.personalized')),
            # min_loss=torch.tensor(fedvrf_global['loss'][-1])
        )

        tr.train_model(
            persn_model, device, criterion, optimizer, n_epochs=args.num_rounds,
            train_loader=train_loader, dev_loader=dev_loader,
            evaluate_cycle=-1, early_stop=True, save_current=True,
            save_current_params=dict(
                path=os.path.join(save_path_dir, model_name.format('{0}.personalized'))
            ),
            early_stop_params=early_stop_params,
            evaluate_fun_params=evaluate_fun_params
        )

        ## Load best model and test its accuracy on the test set
        checkpoint = torch.load(early_stop_params['path'])

        persn_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        tr.vrf_evaluate_model_singlehead(persn_model, device=device, criterion=criterion, test_loader=test_loader, **evaluate_fun_params)
