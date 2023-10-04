#from tests import _PATH_DATA
import pandas as pd
import yaml
from src.models.train_model import load_data
from src.models.train_model import normalize_data
from tests import _PROJECT_ROOT
import os  
import os.path
import pytest

def get_cfg():
    with open(os.path.join(_PROJECT_ROOT, "src/configs/config.yaml"), "r") as yaml_file:
        cfg = yaml.safe_load(yaml_file)
 
    return cfg  

@pytest.mark.skipif(not os.path.exists(get_cfg()["paths"]["training_data_path"]), reason="Data files not found")
def test_data_shape():
    cfg = get_cfg()
    data = load_data(cfg["paths"]["training_data_path"]) 
    assert len(data) == 66595, "Dataset did not have the correct number of samples"
    assert data.shape[1] == cfg["hyperparameters"]["input_size"] + 1, "Dataset did not have the correct number of columns"
   
@pytest.mark.skipif(not os.path.exists(get_cfg()["paths"]["training_data_path"]), reason="Data files not found") 
def test_x_y_split():
    cfg = get_cfg()
    data = load_data(cfg["paths"]["training_data_path"])
    x,y = normalize_data(data)
    assert x.shape[1] == cfg["hyperparameters"]["input_size"], "Features do not have the correct shape"
    assert len(y.shape) == cfg["hyperparameters"]["output_size"], "Targets do not have the correct shape"
    assert y.min() >= -1437 and y.max() <= 156, "Targets not in expected range"

@pytest.mark.skipif(not os.path.exists(get_cfg()["paths"]["training_data_path"]), reason="Data files not found")
def test_normalization(): 
    cfg = get_cfg()
    data = load_data(cfg["paths"]["training_data_path"]) 
    x,_ = normalize_data(data)
    
    # for every column in the input values, apply a min max normalization
    for i in range(x.shape[1]):
        x[:,i] = (x[:,i] - x[:,i].min()) / (x[:,i].max() - x[:,i].min())
        
    x_vals = []
    for x_i in x:
      x_vals.extend(x_i.tolist())
    assert all((pd.isna(x_val) or (x_val >= 0 and x_val <= 1)) for x_val in x_vals) == True, "Features are not normalized correctly"