import flwr.server.strategy
import torch
import flwr as fl
import pandas as pd
import numpy as np

from flwr.common import FitRes, Parameters, Scalar, EvaluateRes, logger
from flwr.server.strategy import FedProx
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate

import os
from logging import WARNING
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import helper as hl
import train as tr


def model_checkpoint(obj, rnd, results, aggregated_weights):
    hl.set_parameters(obj.model, fl.common.parameters_to_ndarrays(aggregated_weights[0]))

    obj.train_loss_aggregated.append(hl.weighted_sum(results, 'train_loss'))
    obj.dev_loss_aggregated.append(hl.weighted_sum(results, 'dev_loss'))

    kwargs = dict({
        'epoch': rnd,
        'loss': obj.train_loss_aggregated,
        'dev_loss': obj.dev_loss_aggregated
    })
    # tr.save_model(self.model, self.save_path, **kwargs)
    tr.save_model(obj.model, obj.save_path.format(rnd), **kwargs)
    return aggregated_weights


class FedProxVRF(FedProx):
    def __init__(self, model, save_path, num_rounds, load_check, ndigits=10, **kwargs):
        self.model = model
        self.ndigits = ndigits
        self.save_path = save_path
        self.train_loss_aggregated = []
        self.dev_loss_aggregated = []
        
        if load_check:
            model_params = torch.load(self.save_path.format(num_rounds))
            self.train_loss_aggregated = model_params['loss']
            self.dev_loss_aggregated = model_params['dev_loss']

        super().__init__(**kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_weights = super().aggregate_fit(server_round, results, failures)

        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {server_round} aggregated_weights...")
            model_checkpoint(self, server_round, results, aggregated_weights)

        return aggregated_weights

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None
        
        # Weigh accuracy of each client by number of examples used
        accuracy_aggregated = hl.weighted_sum(results, 'dev_acc')
        # print(f"Round {server_round} accuracy aggregated from client results: {accuracy_aggregated}")

        # Call aggregate_evaluate from base class (FedAvg)
        loss_aggregated, metrics = super().aggregate_evaluate(server_round, results, failures)

        return round(loss_aggregated, ndigits=self.ndigits), \
               {**metrics, 'accuracy': round(accuracy_aggregated, ndigits=self.ndigits)}
