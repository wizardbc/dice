from typing import Dict, Optional, Tuple, Callable

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import Tensor

from torchmetrics.classification.precision_recall_curve import BinaryPrecisionRecallCurve
from torchmetrics.functional.classification.roc import _binary_roc_compute
from torchmetrics.functional.classification.precision_recall_curve import _binary_precision_recall_curve_compute
from torchmetrics.utilities.data import dim_zero_cat

from tqdm.auto import tqdm
import matplotlib.pyplot as plt


class BinaryCurves(BinaryPrecisionRecallCurve):
  """Curves obtained by confusion matrix.
  """
  is_differentiable: bool = False
  higher_is_better: Optional[bool] = None
  full_state_update: bool = False

  def compute_prc_in(self) -> Tuple[Tensor, Tensor, Tensor]:
    if self.thresholds is None:
      state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)]
    else:
      state = self.confmat
    return _binary_precision_recall_curve_compute(state, self.thresholds)

  def compute_prc_out(self) -> Tuple[Tensor, Tensor, Tensor]:
    if self.thresholds is None:
      state = [1-dim_zero_cat(self.preds), 1-dim_zero_cat(self.target)]
    else:
      state = self.confmat.flip((1,2))
    return _binary_precision_recall_curve_compute(state, self.thresholds)

  def compute_roc(self) -> Tuple[Tensor, Tensor, Tensor]:
    if self.thresholds is None:
      state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)]
    else:
      state = self.confmat
    return _binary_roc_compute(state, self.thresholds)

  def compute(self) -> Dict[str, Tuple[Tensor, Tensor, Tensor]]:
    """Compute ROC, PRC_In, PRC_Out.
    """
    return {
      'ROC': self.compute_roc(),
      'PRC_In': self.compute_prc_in(),
      'PRC_Out': self.compute_prc_out(),
    }


class BinaryMetrics(BinaryCurves):
  """Metrics used in ODIN paper.
  https://github.com/facebookresearch/odin/blob/main/code/calMetric.py
  """
  def fpr_at_95(self) -> float:
    fpr, tpr, _ = self.compute_roc()
    return fpr[tpr > 0.95].min().item()

  def detection_err(self) -> float:
    fpr, tpr, _ = self.compute_roc()
    return ((1-tpr+fpr)/2).min().item()

  def auroc(self, plot:bool=False, **plot_kwargs) -> float:
    fpr, tpr, _ = self.compute_roc()
    if plot:
      plt.plot(fpr.cpu(), tpr.cpu(), **plot_kwargs)
    return torch.trapz(tpr, fpr).item()

  def aupr_in(self, plot:bool=False, **plot_kwargs) -> float:
    p, r, _ = self.compute_prc_in()
    if plot:
      plt.plot(r.cpu(), p.cpu(), **plot_kwargs)
    return -torch.trapz(p, r).item()

  def aupr_out(self, plot:bool=False, **plot_kwargs) -> float:
    p, r, _ = self.compute_prc_out()
    if plot:
      plt.plot(r.cpu(), p.cpu(), **plot_kwargs)
    return -torch.trapz(p, r).item()

  def compute(self, metrics=False) -> Dict[str, Tuple[Tensor, Tensor, Tensor]]|Dict[str, float]:
    res = super().compute()
    if not metrics:
      return res

    fpr, tpr, _ = res['ROC']
    p_in, r_in, _ = res['PRC_In']
    p_out, r_out, _ = res['PRC_Out']
    
    metrics = ({
      'FPR@95': fpr[tpr > 0.95].min().item(),
      'DTErr': ((1-tpr+fpr)/2).min().item(),
      'AUROC': torch.trapz(tpr, fpr).item(),
      'AUPR_In': -torch.trapz(p_in, r_in).item(),
      'AUPR_Out': -torch.trapz(p_out, r_out).item(),
    })

    return metrics


class Runner:
  def __init__(self, ftn:Callable, metrics:BinaryMetrics, id_loader:DataLoader=None, device='cuda:0', prog_bar:bool=True):
    self.ftn = ftn
    self.metrics = metrics.to(device)
    self.device = device
    if id_loader is not None:
      metrics.reset()
      self.run_over_dl(id_loader, ood=False, prog_bar=prog_bar, postfix=False)
    self._save_metrics()

  def run_over_dl(self, dataloader, ood:bool=False, prog_bar:bool=True, postfix:bool=False):
    if prog_bar:
      pbar = tqdm(dataloader, desc='Out-of-dist' if ood else 'In-dist')
    else:
      pbar = dataloader
    with torch.no_grad():
      for imgs, _ in pbar:
        imgs = imgs.to(self.device)
        preds = self.ftn(imgs)
        targets = torch.tensor([int(not ood)]*preds.shape[0], device=self.device)
        self.metrics(preds, targets)
        if prog_bar and postfix:
          pbar.set_postfix(self.metrics.compute(metrics=True))

  def _save_metrics(self):
    if self.metrics.thresholds is None:
      self._id_preds = self.metrics.preds
      self._id_target = self.metrics.target
    else:
      self._id_confmat = self.metrics.confmat
    self._id_update_count = self.metrics._update_count

  def _reset_metrics(self):
    if self.metrics.thresholds is None:
      self.metrics.preds = self._id_preds
      self.metrics.target = self._id_target
    else:
      self.metrics.confmat = self._id_confmat
    self.metrics._update_count = self._id_update_count


  def run(self, ood_loader, prog_bar:bool=True, postfix:bool=False):
    self._reset_metrics()
    self.run_over_dl(ood_loader, ood=True, prog_bar=prog_bar, postfix=postfix)
    return self.metrics.compute(metrics=True)

def avg_metrics(dicts_metrics:Tuple[Dict]):
  res = {}
  for d in dicts_metrics:
    for k, v in d.items():
      if k in res.keys():
        res[k].append(v)
      else:
        res[k] = [v]
  for k, v in res.items():
    res[k] = np.mean(v)
  return res