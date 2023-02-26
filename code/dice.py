import numpy as np

import torch
import torch.nn.functional as F
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from tqdm.auto import tqdm

class DICE:
  def __init__(self, model, device='cuda:0', mode='energy'):
    model.eval()
    model.to(device)
    self.model = model
    self.device = device
    self._saved_weight = None
    self.mode = mode

  def set_dice_(self, avg_features, p=90):
    if self._saved_weight is None:
      self._saved_weight = self.model.fc.weight.data
    weight = self._saved_weight
    contrib = avg_features * weight
    thresh = np.percentile(contrib.cpu().numpy(), p)
    mask = contrib > thresh
    masked_weight = weight * mask
    self.model.fc.weight.data = masked_weight

  def forward(self, x):
    with torch.no_grad():
      if self.mode == 'energy':
        return torch.logsumexp(self.model(x), -1)/1000.0  # to make the return values in [0,1] interval.
      if self.mode == 'msp':
        return F.softmax(self.model(x), -1).max(dim=-1)[0]
      return self.model(x)

  __call__ = forward