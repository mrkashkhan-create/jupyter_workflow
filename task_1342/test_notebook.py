import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor
from types import FunctionType
import inspect

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for testing

notebook_file='initial_notebook.ipynb'


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

def _check_variable_existence_and_type(variable_name, type_,type_str,namespace):
    v="function" if type_==FunctionType else "variable"
    assert variable_name in namespace, f"{v} '{variable_name}' is missing in the notebook."
    assert isinstance(namespace[variable_name], type_), f"{v} '{variable_name}' is not of type {type_str}."

namespace = get_notebook_namespace(notebook_file)


# tests on data and features

def test_core_data_and_features():
    """check df_model, X, y exist and are clean"""
    _check_variable_existence_and_type('df_model', pd.DataFrame,'DataFrame',namespace)
    df_model=namespace['df_model']

    assert "log_price" in df_model.columns, "Column 'log_price' is missing in df_model."
    
    
    _check_variable_existence_and_type('X', np.ndarray,'ndarray',namespace)
    _check_variable_existence_and_type('y', np.ndarray,'ndarray',namespace)
    
    X=namespace['X']
    y=namespace['y']
    
    # check for NaN    
    assert not np.isnan(X).any(), "Feature matrix X contains NaN values."
    assert not np.isnan(y).any(), "Target vector y contains NaN values."

def test_traim_val_test_splits_and_scaling():
    """check train, val, test splits and scalers exist and that the have the correct shapes"""
    for name in ["X_train_scaled", "X_val_scaled", "X_test_scaled", "y_train", "y_val", "y_test"]:
        assert name in namespace, f"Variable '{name}' is missing in the notebook."


    X_train_scaled=namespace['X_train_scaled']
    X_val_scaled=namespace['X_val_scaled']
    X_test_scaled=namespace['X_test_scaled']
    y_train=namespace['y_train']
    y_val=namespace['y_val']
    y_test=namespace['y_test']

    assert X_train_scaled.ndim == 2, "X_train_scaled should be a 2D array."
    assert X_val_scaled.ndim == 2, "X_val_scaled should be a 2D array."
    assert X_test_scaled.ndim == 2, "X_test_scaled should be a 2D array."
    
    assert y_train.ndim == 2 and y_train.shape[1] == 1, "y_train should be a 2D array with one column."
    assert y_val.ndim == 2 and y_val.shape[1] == 1, "y_val should be a 2D array with one column."
    assert y_test.ndim == 2 and y_test.shape[1] == 1, "y_test should be a 2D array with one column."

    # no NaN values after scaling
    for arr, name in [(X_train_scaled,'X_train_scaled'), (X_val_scaled,'X_val_scaled'), (X_test_scaled,'X_test_scaled'),
                  (y_train,'y_train'), (y_val,'y_val'), (y_test,'y_test')]:
        assert not np.isnan(arr).any(), f"Scaled data '{name}' contains NaN values."


# dataset and dataloader tests

def test_datasets_and_dataloaders():
    """checl HousePriceDataset and DataLoaders existence and functionality"""
    _check_variable_existence_and_type('HousePriceDataset', type,'class',namespace)
    HousePriceDataset=namespace['HousePriceDataset']
    
    assert "BATCH_SIZE" in namespace, "Variable 'BATCH_SIZE' is missing in the notebook."
    _check_variable_existence_and_type('train_loader', DataLoader,'DataLoader',namespace)
    _check_variable_existence_and_type('val_loader', DataLoader,'DataLoader',namespace)
    _check_variable_existence_and_type('test_loader', DataLoader,'DataLoader',namespace)

    train_loader=namespace['train_loader']

    # check one batch
    bathch = next(iter(train_loader))
    X_batch, y_batch = bathch
    assert isinstance(X_batch, torch.Tensor), "Features in batch are not torch Tensors."
    assert isinstance(y_batch, torch.Tensor), "Targets in batch are not torch Tensors."
    
    assert X_batch.ndim == 2 or X_batch.ndim==4, "Features batch should be 2D or 4D tensor."
    assert y_batch.ndim == 2 and y_batch.shape[1] == 1, "Targets batch should be 2D tensor with one column."


# model, training and evaluation tests

def test_model_structure_and_forward():
    """ENsure model is an instance of nn.Module and output shape is correct"""
    assert 'model' in namespace, "Variable 'model' is missing in the notebook."
    model=namespace['model']
    assert isinstance(model, torch.nn.Module), "Model is not an instance of torch.nn.Module."

    train_loader=namespace['train_loader']
    batch = next(iter(train_loader))
    X_batch, _ = batch

    device = namespace.get('device', torch.device('cpu'))
    model=model.to(device)
    X_batch = X_batch.to(device)

    with torch.no_grad():
        outputs = model(X_batch)

    assert outputs.ndim == 2, "Model output should be a 2D tensor."
    assert outputs.shape[1] == 1, "Model output should have one column. [regression output]"

# eval function test

def test_train_epoch_signature():
    """ensure train_epoch has correct signature and parameters"""
    _check_variable_existence_and_type("train_epoch", FunctionType,'function',namespace)
    sig=inspect.signature(namespace['train_epoch'])
    params=list(sig.parameters.keys())
    expected=['model', 'dataloader', "optimiser","mse_loss_fn","mae_loss_fn","device"]
    assert params==expected, f"train_epoch function should have parameters: {expected}"


def test_eval_model_signature_and_run():
    """check eval_model function existence, signature and run on val_loader"""
    _check_variable_existence_and_type("eval_model", FunctionType,'function',namespace)
    sig=inspect.signature(namespace['eval_model'])
    params=list(sig.parameters.keys())
    expected=['model', 'dataloader', "mse_loss_fn","mae_loss_fn","device"]
    assert params==expected, f"eval_model function should have parameters: {expected}"

    model=namespace['model']
    val_loader=namespace['val_loader']
    mse_loss_fn=namespace['mse_loss_fn']
    mae_loss_fn=namespace['mae_loss_fn']
    device=namespace.get('device', torch.device('cpu'))

    # run to ensure no eexceptions and numeric outputs
    val_mse, val_mae = namespace['eval_model'](model, val_loader, mse_loss_fn, mae_loss_fn, device)
    assert isinstance(val_mse, float), "Returned val_mse is not a float."
    assert isinstance(val_mae, float), "Returned val_mae is not a float."
    assert np.isfinite(val_mse), "Returned val_mse is not finite."
    assert np.isfinite(val_mae), "Returned val_mae is not finite."
