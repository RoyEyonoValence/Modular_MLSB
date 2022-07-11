import os
import json
import types
import six
import torch
import hashlib
import functools
import fsspec
import numpy as np
from scipy import sparse
from copy import deepcopy
from joblib import Parallel, delayed
from collections.abc import MutableMapping
import pdb


def is_callable(func):
    FUNCTYPES = (types.FunctionType, types.MethodType, functools.partial)
    return func and (isinstance(func, FUNCTYPES) or callable(func))


def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def params_to_hash(all_params):
    all_params = deepcopy(all_params)
    all_params["dataset_params"].pop("nb_query_samples", 0)
    uid = hashlib.sha1(json.dumps(all_params, sort_keys=True).encode()).hexdigest()
    return uid


def parent_at_depth(filename, depth=1):
    f = os.path.abspath(filename)
    n = len(f.split("/"))
    assert depth < n, "The maximum depth is exceeded"
    for _ in range(depth):
        f = os.path.dirname(f)
    return f


def batch_run(f, iterable, *args, **kwargs):
    return [f(x, *args, **kwargs) for x in iterable]


def parallel_run(f, iterable, *args, batch_size=256, n_jobs=-1, verbose=1, **kwargs):
    k = batch_size
    res = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(batch_run)(f, iterable[i : i + k], *args, **kwargs)
        for i in range(0, len(iterable), k)
    )
    res = sum(res, [])
    return res


def map_reduce(f_map, f_reduce, iterable, *args, **kwargs):
    return f_reduce([f_map(x, *args, **kwargs) for x in iterable])


def parallel_map_reduce(
    f_map, f_reduce, iterable, *args, batch_size=256, n_jobs=-1, verbose=1, **kwargs
):
    k = batch_size
    res = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(map_reduce)(f_map, f_reduce, iterable[i : i + k], *args, **kwargs)
        for i in range(0, len(iterable), k)
    )
    res = f_reduce(res)
    return res


def path_exists(path):
    if path is None:
        return False
    fs, url = fsspec.core.url_to_fs(path)
    return fs.exists(url)

# TORCH UTILS

ACTIVATIONS = {
    'ReLU' : torch.nn.ReLU(), 
    'Sigmoid' : torch.nn.Sigmoid(), 
    'Tanh' : torch.nn.Tanh(), 
    'ELU' : torch.nn.ELU(), 
    'SELU' : torch.nn.SELU(), 
    'GLU' : torch.nn.GLU(), 
    'LeakyReLU' : torch.nn.LeakyReLU(), 
    'Softplus' : torch.nn.Softplus(), 
    'None': None}


OPTIMIZERS = {
    'adadelta': torch.optim.Adadelta,
    'adagrad': torch.optim.Adagrad,
    'adam': torch.optim.Adam,
    'sparseadam': torch.optim.SparseAdam,
    'asgd': torch.optim.ASGD,
    'sgd': torch.optim.SGD,
    'rprop': torch.optim.Rprop,
    'rmsprop': torch.optim.RMSprop,
    'optimizer': torch.optim.Optimizer,
    'lbfgs': torch.optim.LBFGS
}


def to_tensor(*x, device=None, dtype=None):
    r"""
    Convert a numpy array to tensor. The tensor type will be
    the same as the original array, unless specify otherwise

    Arguments
    ----------
        x: tuple(numpy.ndarray or scipy.sparse.spmatrix or torch.tensor)
            Numpy array to convert to tensor type
        device: torch.device, optional
        dtype: torch.dtype, optional
            Enforces new data type for the output

    Returns
    -------
        New torch.Tensor

    """
    out = []
    for elem in x:
        if isinstance(elem, torch.Tensor):
            pass
        elif isinstance(elem, np.ndarray):
            elem = torch.from_numpy(elem)
        elif isinstance(elem, sparse.spmatrix):
            elem = torch.from_numpy(elem.todense())
        else:
            elem = torch.tensor(elem)

        elem = elem.to(dtype=elem.dtype if dtype is None else dtype,
                       device=elem.device if device is None else device)
        out.append(elem)

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)


def to_numpy(*x, dtype=None, densify=False):
    r"""
    Convert a scipy or tensor to numpy. The array type will be
    the same as the original array, unless specify otherwise

    Arguments
    ----------
        x: tuple(numpy.ndarray or scipy.sparse.spmatrix or torch.tensor)
            Numpy array to convert to tensor type
        dtype: torch.dtype, optional
            Enforces new data type for the output

    Returns
    -------
        New np.ndarray

    """
    out = []
    for elem in x:
        if isinstance(elem, torch.Tensor):
            elem = elem.cpu().detach().numpy()
        elif isinstance(elem, np.ndarray):
            elem = elem.astype(dtype)
        elif isinstance(elem, sparse.spmatrix):
            if densify:
                elem = elem.todense()
        else:
            elem = np.array(elem)

        elem = elem.astype(dtype=dtype)
        out.append(elem)

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)


def is_device_cuda(device, ignore_errors=False):
    r"""Check wheter the given device is a cuda device.

    Arguments
    ----------
        device: str, torch.device
            object to check for cuda
        ignore_errors: bool, Optional
            Whether to ignore the error if the device is not recognized.
            Otherwise, ``False`` is returned in case of errors.
            (Default=False)
    Returns
    -------
        is_cuda: bool
    """

    if ignore_errors:
        is_cuda = False
        try:
            is_cuda = torch.device(device).type == 'cuda'
        except:
            pass
    else:
        is_cuda = torch.device(device).type == 'cuda'
    return is_cuda


def get_activation(activation):
    if is_callable(activation):
        return activation
    if activation is None:
        return None
    activation = [
        ACTIVATIONS[x] for x in ACTIVATIONS if activation.lower() == x.lower()]
    assert len(activation) > 0, \
        'Unhandled activation function'
    return activation[0]


def get_optimizer(optimizer, *args, **kwargs):
    r"""
    Get an optimizer by name. cUstom optimizer, need to be subclasses of :class:`torch.optim.Optimizer`.

    Arguments
    ----------
        optimizer: :class:`torch.optim.Optimizer` or str
            A class (not an object) or a valid pytorch Optimizer name

    Returns
    -------
        optm `torch.optim.Optimizer`
            Class that should be initialized to get an optimizer.s
    """
    OPTIMIZERS = {k.lower(): v for k, v in vars(torch.optim).items()
                  if not k.startswith('__')}
    if not isinstance(optimizer, six.string_types) and issubclass(optimizer.__class__, torch.optim.Optimizer):
        return optimizer
    return OPTIMIZERS[optimizer.lower()](*args, **kwargs)



def sigmoid_cosine_distance_p(x, y, p=1):
    sig = torch.nn.Sigmoid()
    cosine_sim = torch.nn.CosineSimilarity()
    return (1 - sig(cosine_sim(x, y))) ** p
