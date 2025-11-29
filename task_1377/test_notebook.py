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

notebook_file='final_notebook.ipynb'

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
    """check if df_model, feature_cols_cls, X_cls, y_cls exist and have correct types and values"""

    _check_variable_existence_and_type('df_model', pd.DataFrame,'DataFrame',namespace)
    _check_variable_existence_and_type('feature_cols_cls', list,'list',namespace)

    assert "X_cls" in namespace, "Variable 'X_cls' is missing in the notebook."
    assert "y_cls" in namespace, "Variable 'y_cls' is missing in the notebook."
    X_cls=namespace['X_cls']
    y_cls=namespace['y_cls']
    feature_cols_cls=namespace['feature_cols_cls']

    assert isinstance(X_cls, (np.ndarray,)) or torch.is_tensor(X_cls), "X_cls should be a numpy array or torch Tensor."
    assert isinstance(y_cls, (np.ndarray,)) or torch.is_tensor(y_cls), "y_cls should be a numpy array or torch Tensor."

    # convert to numpy for shape checks
    if torch.is_tensor(X_cls):
        X_arr = X_cls.numpy()
    else:
        X_arr = np.array(X_cls)
    
    if torch.is_tensor(y_cls):
        y_arr = y_cls.numpy()
    else:
        y_arr = np.array(y_cls)

    assert X_arr.ndim == 2, "X_cls should be a 2D array."
    assert y_arr.ndim == 1 or (y_cls.ndim == 2 and y_cls.shape[1] == 1), "y_cls should be a 1D array."

    # number of features should match feature_cols_cls
    assert X_arr.shape[1] == len(feature_cols_cls), "Number of features in X_cls does not match length of feature_cols_cls."
    assert X_arr.shape[0] == y_arr.shape[0], "Number of samples in X_cls and y_cls do not match."


def test_traim_val_test_splits_and_scaling():
    """check train, val, test splits and scalers exist and that the have the correct shapes"""
    for name in ["X_train_scaled", "X_val_scaled", "X_test_scaled", "y_train", "y_val", "y_test"]:
        assert name in namespace, f"Variable '{name}' is missing in the notebook."

    X_train_scaled = namespace['X_train_scaled']
    X_val_scaled = namespace['X_val_scaled']
    X_test_scaled = namespace['X_test_scaled']
    y_train = namespace['y_train']
    y_val = namespace['y_val']
    y_test = namespace['y_test']    

    # check they are numpy arrays or torch tensors
    def to_numpy(x):
        if torch.is_tensor(x):
            return x.cpu().numpy()
        return x
    
    X_train_arr = to_numpy(X_train_scaled)
    X_val_arr = to_numpy(X_val_scaled)
    X_test_arr = to_numpy(X_test_scaled)
    y_train_arr = to_numpy(y_train)
    y_val_arr = to_numpy(y_val)
    y_test_arr = to_numpy(y_test)

    # check shapes
    assert X_train_arr.ndim == 2, "X_train_scaled should be a 2D array."
    assert y_train_arr.ndim == y_val_arr.ndim == y_test_arr.ndim == 1,  "y_train, y_val, y_test should be 1D arrays."

    # same number of features
    assert X_train_arr.shape[1] == X_val_arr.shape[1] == X_test_arr.shape[1], "Number of features should be the same across train, val and test sets."

    # non epmty splits
    assert X_train_arr.shape[0] > 0 and X_val_arr.shape[0]>0 and X_test_arr.shape[0]>0, "Train, val and test sets should not be empty."

    # num_classes and class_names
    assert "num_classes" in namespace, "Variable 'num_classes' is missing in the notebook."

    num_classes = namespace['num_classes']
    assert isinstance(num_classes, int) and num_classes > 1, "'num_classes' should be an integer greater than 1."

    assert "class_names" in namespace, "Variable 'class_names' is missing in the notebook."
    assert len(namespace['class_names']) == num_classes, "'class_names' length should match 'num_classes'."


    # dataloaders
    for name in ["train_loader", "val_loader", "test_loader"]:
        _check_variable_existence_and_type(name, DataLoader,'DataLoader',namespace)

    train_loader = namespace['train_loader']

    # check dataloader contents
    batch = next(iter(train_loader))
    assert isinstance(batch, (list, tuple)) and len(batch) == 2, "Each batch from train_loader should be a tuple (inputs, targets)."
    Xbatch, ybatch = batch
    assert isinstance(Xbatch, torch.Tensor), "Inputs from train_loader should be torch Tensors."
    assert isinstance(ybatch, torch.Tensor), "Targets from train_loader should be torch Tensors."
    assert Xbatch.ndim == 2, "Inputs from train_loader should be 2D tensors."
    assert ybatch.ndim == 1, "Targets from train_loader should be 1D tensors."


# Model tests
def test_model_structure_and_forwards():
    """check model exists, is nn.Module and has correct input/output shapes"""
    assert "model" in namespace, "Variable 'model' is missing in the notebook."
    model = namespace['model']
    num_classes = namespace['num_classes']

    assert isinstance(model, torch.nn.Module), "'model' should be an instance of torch.nn.Module."

    # check final layer output size
    train_loader = namespace['train_loader']
    batch = next(iter(train_loader))
    Xbatch, _ = batch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=model.to(device)
    Xbatch = Xbatch.to(device)
    with torch.no_grad():
        outputs = model(Xbatch)

    assert outputs.ndim == 2, "Model outputs should be 2D tensors."
    assert outputs.shape[1] == num_classes, f"Model outputs should have shape = to num_classes ({num_classes})."

# training functions
def test_train_function_sig():
    """check train function exists and has correct parameters"""
    _check_variable_existence_and_type('train', FunctionType,'function',namespace)

    sig=inspect.signature(namespace['train'])
    params=list(sig.parameters.keys())
    expected=['model', 'loader', 'criterion', 'optimiser', 'epoch']
    assert params==expected, f"'train' function should have parameters {expected}, but has {params}."    


def test_evaluate_function_sig():
    """check evaluate function exists and has correct parameters"""
    _check_variable_existence_and_type('evaluate', FunctionType,'function',namespace)

    sig=inspect.signature(namespace['evaluate'])
    params=list(sig.parameters.keys())
    expected=['model', 'loader']
    assert params==expected, f"'evaluate' function should have parameters {expected}, but has {params}."

def test_train_eval_functions_run():
    """check train and evaluate functions run without errors for one epoch"""
    
    _check_variable_existence_and_type('train_eval', FunctionType,'function',namespace)
    sig=inspect.signature(namespace['train_eval'])
    assert "epochs" in sig.parameters.keys(), "'train_eval' function should have 'epochs' parameter."

    # run for 1 epoch
    train_eval=namespace['train_eval']
    try:
        result = train_eval()
    except TypeError:
        result = train_eval(1)

    # allow tuple,list or None
    assert result is None or isinstance(result, (tuple, list)), "'train_eval' should return None or a tuple/list."
