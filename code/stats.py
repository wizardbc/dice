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

  def run(self, dataloader:DataLoader, prog_bar:bool=True):
    if prog_bar:
      pbar = tqdm(dataloader)
    else:
      pbar = dataloader
    with torch.no_grad():
      for imgs, _ in pbar:
        imgs = imgs.to(self.device)
        features = self.model(imgs)
        for k, v in features.items():
          self.features[k].append(v)

  def compute(self):
    features = {}
    for k, v in self.features.items():
      features[k] = torch.concat(v).mean(dim=0)
    return {
      k: v.cpu().numpy() if len(v.shape)==1 else v.flatten(start_dim=1).mean(dim=1).cpu().numpy()
      for k, v in features.items()
    }