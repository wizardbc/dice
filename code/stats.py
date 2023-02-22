from typing import Dict, Optional, Tuple, Callable

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

  def compute(self, target:int=None):
    features = {
      k: torch.concat(v) if target is None else torch.concat(v)[torch.concat(self.target) == target]
      for k, v in self.features.items()
    }
    mean = {
      k: v.mean(dim=(0,2,3)) if v.dim()==4 else v.mean(dim=0)
      for k, v in features.items()
    }
    std = {
      k: v.std(dim=(0,2,3)) if v.dim()==4 else v.std(dim=0)
      for k, v in features.items()
    }
    return mean, std