"""Loads the LGBM model and fuses it with the rule engine."""
import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import lightgbm as lgb

from src.cv.features import features_to_vector
from src.cv.rule_engine import rule_probs, CLASSES


class PostureClassifier:
    def __init__(self,
                 model_path: str = "models/posture_lgbm_v3.txt",
                 feature_order_path: str = "models/feature_order.json",
                 rule_weight: float = 0.30):
        self.model = lgb.Booster(model_file=model_path)
        with open(feature_order_path) as f:
            self.feature_order = json.load(f)
        self.rule_weight = rule_weight

    def predict(self, features: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        x = features_to_vector(features).reshape(1, -1)
        ml = self.model.predict(x)[0]
        rl = rule_probs(features)
        fused = (1.0 - self.rule_weight) * ml + self.rule_weight * rl
        idx = int(np.argmax(fused))
        return (CLASSES[idx],
                float(fused[idx]),
                dict(zip(CLASSES, fused.tolist())))
