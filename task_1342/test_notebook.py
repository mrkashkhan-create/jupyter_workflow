import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor
from types import FunctionType
import inspect

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for testing

notebook_file = 'final_notebook.ipynb'


@pytest.mark.parametrize("notebook_file", [notebook_file])
def test_notebook_exec(notebook_file):
    """Test that the Jupyter notebook runs without errors."""
    with open(notebook_file) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=1200, kernel_name='python3')

    try:
        assert ep.preprocess(nb) is not None, f"Notebook {notebook_file} did not execute properly."
    except Exception as e:
        assert False, f"Notebook execution failed: {e}"


def get_notebook_namespace(notebook_file):
    """Extract the namespace (variables and functions) from the executed notebook."""
    with open(notebook_file) as f:
        nb = nbformat.read(f, as_version=4)

    namespace = {}
    for cell in nb.cells:
        if cell.cell_type == 'code':
            exec(cell.source, namespace)
    return namespace


def _check_variable_existence_and_type(variable_name, type_, type_str, namespace):
    v = "function" if type_ == FunctionType else "variable"
    assert variable_name in namespace, f"{v} '{variable_name}' is missing in the notebook."
    assert isinstance(namespace[variable_name], type_), f"{v} '{variable_name}' is not of type {type_str}."


namespace = get_notebook_namespace(notebook_file)


# tests on data and features

def test_core_data_and_features():
    """Validate df_model X, y and basic feature engineering."""
    _check_variable_existence_and_type('df_model', pd.DataFrame, 'DataFrame', namespace)

    df_model = namespace['df_model']
    


    # core arrays
    assert "X" in namespace and "y" in namespace, "Variables 'X' and 'y' must exist."
    X = namespace['X']
    y = namespace['y']

    assert isinstance(X, (np.ndarray,)) or torch.is_tensor(X), "X should be a numpy array or torch Tensor."
    assert isinstance(y, (np.ndarray,)) or torch.is_tensor(y), "y should be a numpy array or torch Tensor."

    X_arr = X.cpu().numpy() if torch.is_tensor(X) else X
    y_arr = y.cpu().numpy() if torch.is_tensor(y) else y

    assert X_arr.ndim == 2, "X should be a 2D array."
    assert y_arr.ndim in (1, 2), "y should be 1D or 2D."
    if y_arr.ndim == 2:
        assert y_arr.shape[1] == 1, "y should have a single target column."

    assert X_arr.shape[0] == y_arr.shape[0], "X and y should have the same number of rows."
    assert not np.isnan(X_arr).any(), "Feature matrix X contains NaN values."
    assert not np.isnan(y_arr).any(), "Target vector y contains NaN values."


def test_train_val_test_splits_and_scaling():
    """Check splits exist, shapes match and training data is scaled."""
    required_vars = ["X_train_scaled", "X_val_scaled", "X_test_scaled", "y_train", "y_val", "y_test"]
    for name in required_vars:
        assert name in namespace, f"Variable '{name}' is missing in the notebook."

    def to_numpy(x):
        return x.cpu().numpy() if torch.is_tensor(x) else x

    X_train = to_numpy(namespace['X_train_scaled'])
    X_val = to_numpy(namespace['X_val_scaled'])
    X_test = to_numpy(namespace['X_test_scaled'])
    y_train = to_numpy(namespace['y_train'])
    y_val = to_numpy(namespace['y_val'])
    y_test = to_numpy(namespace['y_test'])

    for arr, name in [(X_train, "X_train_scaled"), (X_val, "X_val_scaled"), (X_test, "X_test_scaled")]:
        assert arr.ndim == 2, f"{name} should be 2D."
        assert arr.shape[0] > 0, f"{name} should not be empty."

    for arr, name in [(y_train, "y_train"), (y_val, "y_val"), (y_test, "y_test")]:
        assert arr.ndim in (1, 2), f"{name} should be 1D or 2D with one column."
        if arr.ndim == 2:
            assert arr.shape[1] == 1, f"{name} should have one target column."

    # consistent feature counts
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], "Feature count must match across splits."

    # scaling on train set (StandardScaler)
    mean_close = np.allclose(X_train.mean(axis=0), 0, atol=0.15)
    train_stds = X_train.std(axis=0)
    non_zero = train_stds > 1e-6
    std_close = np.allclose(train_stds[non_zero], 1, atol=0.15)
    # allow zero variance columns (std == 0), otherwise expect ~1
    assert mean_close and std_close, "X_train_scaled does not appear standardised (mean≈0, std≈1) on non-constant features."

    # no NaNs
    for arr, name in [(X_train, 'X_train_scaled'), (X_val, 'X_val_scaled'), (X_test, 'X_test_scaled'),
                      (y_train, 'y_train'), (y_val, 'y_val'), (y_test, 'y_test')]:
        assert np.isfinite(arr).all(), f"{name} contains non-finite values."


def _check_loader(loader, batch_size, name):
    """Helper to validate a DataLoader's batch structure and size."""
    batch = next(iter(loader))
    assert isinstance(batch, (list, tuple)) and len(batch) == 2, f"{name} batch should be (features, targets)."
    X_batch, y_batch = batch
    assert isinstance(X_batch, torch.Tensor), f"Features from {name} should be torch Tensors."
    assert isinstance(y_batch, torch.Tensor), f"Targets from {name} should be torch Tensors."
    assert X_batch.ndim == 2, f"Features from {name} should be 2D tensors."
    assert y_batch.ndim in (1, 2), f"Targets from {name} should be 1D or 2D tensors."
    if y_batch.ndim == 2:
        assert y_batch.shape[1] == 1, f"Targets from {name} should have one column."
    assert 0 < X_batch.shape[0] <= batch_size, f"{name} batch size should be <= BATCH_SIZE."
    assert X_batch.shape[0] == y_batch.shape[0], f"{name} features and targets should align."


def test_datasets_and_dataloaders():
    """Check HousePriceDataset, loaders and batch structure."""
    _check_variable_existence_and_type('HousePriceDataset', type, 'class', namespace)
    assert issubclass(namespace['HousePriceDataset'], Dataset), "HousePriceDataset should inherit from torch.utils.data.Dataset."
    assert "BATCH_SIZE" in namespace and isinstance(namespace["BATCH_SIZE"], int), "Variable 'BATCH_SIZE' is missing or not an int."

    for name in ["train_loader", "val_loader", "test_loader"]:
        _check_variable_existence_and_type(name, DataLoader, 'DataLoader', namespace)

    batch_size = namespace['BATCH_SIZE']
    _check_loader(namespace['train_loader'], batch_size, "train_loader")
    _check_loader(namespace['val_loader'], batch_size, "val_loader")
    _check_loader(namespace['test_loader'], batch_size, "test_loader")


def test_model_structure_and_forward():
    """Ensure model is an nn.Module and forward outputs correct shape."""
    assert "PricePredictorNN" in namespace, "Class 'PricePredictorNN' is missing."
    PricePredictorNN = namespace['PricePredictorNN']
    assert issubclass(PricePredictorNN, nn.Module), "PricePredictorNN should inherit from nn.Module."
    assert 'model' in namespace, "Variable 'model' is missing in the notebook."
    model = namespace['model']
    assert isinstance(model, nn.Module), "Model is not an instance of torch.nn.Module."

    train_loader = namespace['train_loader']
    X_batch, _ = next(iter(train_loader))
    device = namespace.get('device', torch.device('cpu'))
    model = model.to(device)
    X_batch = X_batch.to(device)

    with torch.no_grad():
        outputs = model(X_batch)

    assert outputs.ndim == 2, "Model output should be a 2D tensor."
    assert outputs.shape[1] == 1, "Model output should have one column for regression."


def test_train_and_eval_functions():
    """Check training/eval functions signatures and basic run."""
    _check_variable_existence_and_type("train_epoch", FunctionType, 'function', namespace)
    _check_variable_existence_and_type("eval_model", FunctionType, 'function', namespace)
    sig_train = inspect.signature(namespace['train_epoch'])
    sig_eval = inspect.signature(namespace['eval_model'])
    expected_train = ['model', 'dataloader', "optimiser", "mse_loss_fn", "mae_loss_fn", "device"]
    expected_eval = ['model', 'dataloader', "mse_loss_fn", "mae_loss_fn", "device"]
    assert list(sig_train.parameters.keys()) == expected_train, f"train_epoch parameters should be {expected_train}"
    assert list(sig_eval.parameters.keys()) == expected_eval, f"eval_model parameters should be {expected_eval}"

    # required training objects
    for var in ["model", "train_loader", "val_loader", "mse_loss_fn", "mae_loss_fn", "optimiser"]:
        assert var in namespace, f"Variable '{var}' is missing in the notebook."

    model = namespace['model']
    val_loader = namespace['val_loader']
    mse_loss_fn = namespace['mse_loss_fn']
    mae_loss_fn = namespace['mae_loss_fn']
    device = namespace.get('device', torch.device('cpu'))

    # eval_model should return finite floats
    val_mse, val_mae = namespace['eval_model'](model, val_loader, mse_loss_fn, mae_loss_fn, device)
    assert isinstance(val_mse, float) and np.isfinite(val_mse), "Returned val_mse is not a finite float."
    assert isinstance(val_mae, float) and np.isfinite(val_mae), "Returned val_mae is not a finite float."

    # train_epoch should run without errors
    train_loader = namespace['train_loader']
    optimiser = namespace['optimiser']
    result = namespace['train_epoch'](model, train_loader, optimiser, mse_loss_fn, mae_loss_fn, device)
    assert result is None or isinstance(result, (tuple, list)), "train_epoch should return None or a tuple/list."
