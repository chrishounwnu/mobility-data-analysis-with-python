import rdp
import torch
import numpy as np
import pandas as pd

from shapely.geometry import Point
from collections import OrderedDict

from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm


# Get the coordinates of a Shapely Geometry (e.g. Point, Polygon, etc.) as NumPy array
shapely_coords_numpy = lambda l: np.array(*list(l.coords))


def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def weighted_sum(results, metric):
    # Weigh accuracy of each client by number of examples used
    metric_aggregated = [r.metrics[metric] * r.num_examples for _, r in results]
    examples = [r.num_examples for _, r in results]

    # Aggregate and print custom metric
    return sum(metric_aggregated) / sum(examples)


def applyParallel(df_grouped, fun, n_jobs=-1, **kwargs):
    '''
    Forked from: https://stackoverflow.com/a/27027632
    '''
    n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    print(f'Scaling {fun} to {n_jobs} CPUs')

    df_grouped_names = df_grouped.grouper.names
    _fun = lambda name, group: (fun(group.drop(df_grouped_names, axis=1)), name)

    result, keys = zip(*Parallel(n_jobs=n_jobs)(
        delayed(_fun)(name, group) for name, group in tqdm(df_grouped, **kwargs)
    ))
    return pd.concat(result, keys=keys, names=df_grouped_names)
