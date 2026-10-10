import argparse, glob, json, os
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from torch.utils.data import DataLoader

from train import CachedDataset, build_model, forward


@torch.no_grad()
def predict(model, loader, kind, dev, zero=None):
    model.eval()
    Y, P = [], []
    for b in loader:
        b = dict(b)
        if zero:
            b[zero] = torch.zeros_like(b[zero])
        P.append(torch.softmax(forward(model, b, kind, dev), 1)[:, 1].float().cpu()); Y.append(b["y"])
    return torch.cat(Y).numpy(), torch.cat(P).numpy()


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", pos_label=1, zero_division=0)
    auc = roc_auc_score(y, p) if len(set(y.tolist())) == 2 else None
    return dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
                precision_fake=pr, recall_fake=rc, f1_fake=f1, roc_auc=auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "proposed"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cache", default="processed_real")
    ap.add_argument("--splits_dir", default="splits")
    a = ap.parse_args()
    if a.tag.startswith("dummy"):
        print("#" * 60 + "\n# DUMMY RUN: software test only. Do NOT report these numbers.\n" + "#" * 60)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    use_emo = a.model == "proposed"
    ds = CachedDataset(f"{a.splits_dir}/test.csv", a.cache, use_emo)
    loader = DataLoader(ds, batch_size=16, num_workers=2)
    model = build_model(a.model).to(dev)
    model.load_state_dict(torch.load(f"best_{a.model}_{a.tag}.pth", map_location=dev))
    y, p = predict(model, loader, a.model, dev)
    m = metrics(y, p)
    cm = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1]).tolist()
    ds.df["correct"] = ((p >= 0.5).astype(int) == y)
    percat = {c: dict(n=int(len(g)), correct_rate=float(g.correct.mean())) for c, g in ds.df.groupby("category")}
    res = dict(model=a.model, tag=a.tag, n_test=int(len(y)), class_counts=dict(Real=int((y == 0).sum()), Fake=int((y == 1).sum())),
               metrics=m, confusion_matrix=cm, per_category=percat)
    zeroable = ["a"] + (["ie", "ae", "te"] if use_emo else [])
    res["zero_replacement"] = {k: metrics(*predict(model, loader, a.model, dev, zero=k)) for k in zeroable}
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(f"results/eval_{a.model}_{a.tag}.json", "w"), indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))
    files = sorted(glob.glob(f"results/eval_*_{a.tag}.json"))
    if len(files) > 1:
        print("\nCOMPARISON (test set)")
        for f in files:
            r = json.load(open(f)); mm = r["metrics"]
            auc = "n/a" if mm["roc_auc"] is None else f"{mm['roc_auc']:.3f}"
            print(f"{r['model']:9s} bal_acc {mm['balanced_accuracy']:.3f} | F1(fake) {mm['f1_fake']:.3f} | AUC {auc}")


if __name__ == "__main__":
    main()
