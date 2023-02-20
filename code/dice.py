import numpy as np

import torch
import torch.nn.functional as F
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from tqdm.auto import tqdm

class DICE:
  def __init__(self, model, device='cuda:0'):
    model.eval()
    model.to(device)
    self.model = model
    self.device = device
    self._saved_weight = None
    self.contrib = None
    self.mode = 'energe'

  def avg_features(self, id_loader, prog_bar:bool=True):
    # _, eval_nodes = get_graph_node_names(self.model)
    nodes = {'view': 'feature'}
    model_feat = create_feature_extractor(self.model, nodes)

    features = [[] for _ in id_loader.dataset.classes]
    pbar = tqdm(id_loader) if prog_bar else id_loader
    with torch.no_grad():
      for x, y in pbar:
        x = x.to(self.device)
        logit = model_feat(x)['feature']
        for feature, label in zip(logit, y):
          features[label.item()].append(feature)
      features = [torch.stack(lst) for lst in features]
      return torch.cat(features).mean(dim=0, keepdims=True)

  def set_dice_(self, id_loader, p=90, prog_bar:bool=True):
    if self._saved_weight is None:
      self._saved_weight = self.model.fc.weight.data
    weight = self._saved_weight
    if self.contrib is None:
      self.contrib = self.avg_features(id_loader, prog_bar) * weight
    contrib = self.contrib

    thresh = np.percentile(contrib.cpu().numpy(), p)
    mask = contrib > thresh
    masked_weight = weight * mask
    self.model.fc.weight.data = masked_weight

  def forward(self, x):
    with torch.no_grad():
      if self.mode == 'energe':
        return torch.logsumexp(self.model(x), -1)/1000.0  # to make the return values in [0,1] interval.
      if self.mode == 'msp':
        return F.softmax(self.model(x), -1).max(dim=-1)[0]
      return self.model(x)