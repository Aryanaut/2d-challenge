"""Train the surface classifier from labelled frames.

Data layout (made by surface_collect / surface_extract, or by hand):
    data/
      wood/<session>/*.jpg     e.g. wood/flat_run1/, wood/rubble_run1/, wood/steps_run2/
      metal/<session>/*.jpg
      other/<session>/*.jpg
    (images directly in data/<label>/ are fine too; they count as one session)

  ros2 run surface_monitor surface_train data/ --out ~/surface_model.joblib

What it does:
  1. features for every frame + N augmented copies (lighting, blur, rotation, zoom)
  2. cross-validation that keeps whole SESSIONS out of training, so the score says how it does
     on a run it has never seen (neighbouring frames of one run are near-identical; mixing them
     into train and test would give a fake ~100%)
  3. compares an SVM and a random forest, keeps the better one, prints a confusion matrix
  4. refits on everything and saves model + metadata with joblib
"""

import argparse
import glob
import os
import sys
import time
from collections import Counter

import cv2
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import FEATURE_VERSION, augment, feature_vector

EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def load_dataset(root):
    items = []          # (path, label, session)
    for label in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        ldir = os.path.join(root, label)
        for ext in EXTS:
            items += [(p, label, "_root") for p in glob.glob(os.path.join(ldir, ext))]
        for sess in sorted(d for d in os.listdir(ldir) if os.path.isdir(os.path.join(ldir, d))):
            for ext in EXTS:
                items += [(p, label, sess) for p in glob.glob(os.path.join(ldir, sess, ext))]
    return items


def build_features(items, n_aug, roi_frac, seed=0):
    rng = np.random.default_rng(seed)
    X, y, groups, is_orig = [], [], [], []
    t0 = time.time()
    for i, (path, label, sess) in enumerate(items):
        img = cv2.imread(path)
        if img is None:
            print(f"  skipping unreadable {path}")
            continue
        group = f"{label}/{sess}"
        X.append(feature_vector(img, roi_frac)); y.append(label); groups.append(group); is_orig.append(True)
        for _ in range(n_aug):
            X.append(feature_vector(augment(img, rng), roi_frac)); y.append(label)
            groups.append(group); is_orig.append(False)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(items)} images ({time.time() - t0:.0f} s)")
    return np.array(X), np.array(y), np.array(groups), np.array(is_orig)


def candidate_models(seed=0):
    return {
        # SVM wrapped for probabilities (works on old and new scikit-learn)
        "svm_rbf": make_pipeline(StandardScaler(), CalibratedClassifierCV(
            SVC(C=10, gamma="scale", class_weight="balanced"), cv=3)),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                                class_weight="balanced", random_state=seed),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="folder with one sub-folder per label")
    ap.add_argument("--out", default=os.path.expanduser("~/surface_model.joblib"))
    ap.add_argument("--augment", type=int, default=3, help="augmented copies per training frame")
    ap.add_argument("--roi", type=float, default=0.6, help="centre crop fraction (match the node's th_roi_frac)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    items = load_dataset(args.data)
    if not items:
        sys.exit(f"no images found under {args.data}/<label>/")
    per_label = Counter(l for _, l, _ in items)
    sessions = {l: sorted({s for _, ll, s in items if ll == l}) for l in per_label}
    print("Images per label:", dict(per_label))
    print("Sessions per label:", {l: len(s) for l, s in sessions.items()})
    if len(per_label) < 2:
        sys.exit("need at least two labels")

    print(f"Extracting features (+{args.augment} augmented copies each)...")
    X, y, groups, is_orig = build_features(items, args.augment, args.roi, args.seed)

    # Cross-validation by session; fall back to per-image groups if a label has one session
    n_groups = len(set(groups))
    if min(len(s) for s in sessions.values()) < 2:
        print("\nWARNING: some label has only one session, so the score below mixes frames of the same "
              "run into train and test and will look better than reality. Record 2+ runs per surface.")
        groups = np.array([f"{g}#{i // (args.augment + 1)}" for i, g in enumerate(groups)])
        n_groups = len(set(groups))
    if n_groups <= 20:
        cv, cv_desc = LeaveOneGroupOut(), f"{n_groups} folds, one session held out at a time"
    else:
        folds = max(2, min(args.folds, n_groups))
        cv, cv_desc = GroupKFold(n_splits=folds), f"{folds} folds, whole sessions held out"

    print(f"\nCross-validating ({cv_desc})...")
    best_name, best_score, best_pred = None, -1.0, None
    for name, model in candidate_models(args.seed).items():
        pred = np.empty(len(y), dtype=object)
        for tr, te in cv.split(X, y, groups):
            m = sklearn.base.clone(model).fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        # score only on real (non-augmented) frames
        score = f1_score(y[is_orig], pred[is_orig], average="macro")
        print(f"  {name:14s} macro-F1 on held-out real frames = {score:.3f}")
        if score > best_score:
            best_name, best_score, best_pred = name, score, pred

    labels = sorted(per_label)
    print(f"\nBest: {best_name}. Held-out results on real frames:")
    print(classification_report(y[is_orig], best_pred[is_orig], labels=labels, digits=3, zero_division=0))
    cm = confusion_matrix(y[is_orig], best_pred[is_orig], labels=labels)
    print("Confusion matrix (rows = true, columns = predicted):")
    print("          " + "".join(f"{l:>8s}" for l in labels))
    for l, row in zip(labels, cm):
        print(f"{l:>10s}" + "".join(f"{v:8d}" for v in row))

    print("\nPer session (each scored by a model that never saw it):")
    sess_of = np.array([g.split("#")[0] for g in groups])
    for sess in sorted(set(sess_of[is_orig])):
        m = is_orig & (sess_of == sess)
        acc = float((best_pred[m] == y[m]).mean())
        wrong = Counter(best_pred[m][best_pred[m] != y[m]])
        flag = "   <-- check this run" if acc < 0.8 else ""
        print(f"  {sess:28s} {acc:6.1%}  wrong as: {dict(wrong) if wrong else '-'}{flag}")

    print("\nFitting the final model on all data...")
    final = candidate_models(args.seed)[best_name].fit(X, y)
    if isinstance(final, RandomForestClassifier):
        final.set_params(n_jobs=1)          # one frame at a time: threads only add overhead
    t0 = time.perf_counter()
    for _ in range(20):
        final.predict_proba(X[:1])
    pred_ms = (time.perf_counter() - t0) / 20 * 1000

    bundle = {
        "model": final, "labels": labels, "roi_frac": args.roi, "feature_version": FEATURE_VERSION,
        "model_type": best_name, "cv_macro_f1": round(best_score, 4),
        "images_per_label": dict(per_label), "sessions_per_label": {l: len(s) for l, s in sessions.items()},
        "sklearn_version": sklearn.__version__, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    joblib.dump(bundle, args.out)
    print(f"Saved {args.out}  (model {pred_ms:.1f} ms + features ~10-15 ms per frame)")


if __name__ == "__main__":
    main()
