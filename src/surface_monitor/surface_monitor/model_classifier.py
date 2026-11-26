"""Load a trained surface model (from surface_train) and classify frames with it."""

import joblib
import numpy as np

from .features import FEATURE_VERSION, feature_vector


class ModelClassifier:
    def __init__(self, path: str, min_conf: float = 0.6):
        bundle = joblib.load(path)
        if bundle.get("feature_version") != FEATURE_VERSION:
            raise ValueError(f"model was trained with feature_version {bundle.get('feature_version')}, "
                             f"this code uses {FEATURE_VERSION}: retrain with surface_train")
        self.model = bundle["model"]
        self.labels = list(bundle["labels"])
        self.roi_frac = float(bundle.get("roi_frac", 0.6))
        self.min_conf = min_conf
        self.info = {k: v for k, v in bundle.items() if k != "model"}

    def predict(self, bgr):
        """Return (label, confidence, {label: probability}). label is 'unclear' below min_conf."""
        f = feature_vector(bgr, self.roi_frac)[None, :]
        probs = self.model.predict_proba(f)[0]
        classes = list(self.model.classes_)
        p = {classes[i]: float(probs[i]) for i in range(len(classes))}
        best = max(p, key=p.get)
        conf = p[best]
        return (best if conf >= self.min_conf else "unclear"), conf, p
