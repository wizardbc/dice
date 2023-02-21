from typing import Dict, Optional, Tuple, Callable
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

class Stats:
  def __init__(self, model, return_nodes:Dict[str, str]):
    model.eval()
    self.model = create_feature_extractor(model, return_nodes)
    self.features = {v: [] for v in return_nodes.values()}
