from typing import Any, List, Dict, Tuple, Callable, Optional, Union
from typing_extensions import Literal

from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat

import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

from tqdm.auto import tqdm

class Stats:
  def __init__(self, model, return_nodes:Dict[str, str], device:str='cuda:0'):
    model.eval()
    model.to(device)
    self.model = create_feature_extractor(model, return_nodes)
    self.device = device
    self.features = {v: [] for v in return_nodes.values()}
    self.target = []

  def run(self, dataloader:DataLoader, prog_bar:bool=True):
    if prog_bar:
      pbar = tqdm(dataloader)
    else:
      pbar = dataloader
    with torch.no_grad():
      for imgs, labels in pbar:
        imgs = imgs.to(self.device)
        features = self.model(imgs)
        for k, v in features.items():
          self.features[k].append(v.cpu())
        self.target.append(labels.cpu())

  def compute(self, target:int=None, std:bool=False):
    features = {
      k: torch.concat(v) if target is None else torch.concat(v)[torch.concat(self.target) == target]
      for k, v in self.features.items()
    }
    ret_avg = {
      k: v.mean(dim=(0,2,3)) if v.dim()==4 else v.mean(dim=0)
      for k, v in features.items()
    }
    if std:
      ret_std = {
        k: v.std(dim=(0,2,3)) if v.dim()==4 else v.std(dim=0)
        for k, v in features.items()
      }
      return ret_avg, ret_std
    return ret_avg
  

class Recorder(Metric):
  def __init__(
      self,
      state_names:List[str],
      **kwargs: Any,
  ) -> None:
    super().__init__(**kwargs)
    self.state_names = state_names
    for name in state_names:
      self.add_state(name, default=[], dist_reduce_fx="cat")

  def update(self, state: Dict[str,torch.Tensor]) -> None:
    for name in self.state_names:
      getattr(self, name).append(state.get(name, torch.Tensor([0.])))

  def _final_state(self) -> Dict[str, torch.Tensor]:
    return {name: dim_zero_cat(getattr(self, name)) for name in self.state_names}

  def compute(self) -> Dict[str,torch.Tensor]:
    state = self._final_state()
    return {k: v.mean(dim=0) for k, v in state.items()}
  
  def stats(
      self, idx:Tuple[bool]=None, std:bool=False,
    ) -> Union[Dict[str,torch.Tensor], Tuple[Dict[str,torch.Tensor], Dict[str,torch.Tensor]]]:
    state = self._final_state()
    if idx is None:
      avg = {
        k: v.mean(dim=(0,2,3)) if v.dim()==4 else v.mean(dim=0)
        for k, v in state.items()
      }
    else:
      avg = {
        k: v[idx].mean(dim=(0,2,3)) if v.dim()==4 else v[idx].mean(dim=0)
        for k, v in state.items()
      }
    
    if std:
      if idx is None:
        return avg, {
          k: v.std(dim=(0,2,3)) if v.dim()==4 else v.std(dim=0)
          for k, v in state.items()
        }
      return avg, {
          k: v[idx].std(dim=(0,2,3)) if v.dim()==4 else v[idx].std(dim=0)
          for k, v in state.items()
        }
      
    return avg