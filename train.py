import argparse, json, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset

from models import ProposedMultimodalFramework, ResNet18_LSTM_MM


class CachedDataset(Dataset):
    def __init__(self, csv, root, use_emo):
        df = pd.read_csv(csv)
        ok = [os.path.exists(f"{root}/{v}.va.npz") and (not use_emo or os.path.exists(f"{root}/{v}.emo.npz")) for v in df.video_id]
        self.dropped = int(len(df) - sum(ok))
        self.df = df[ok].reset_index(drop=True)
        if len(self.df) == 0:
            raise SystemExit(f"No cached files for {csv} in {root}. Run preprocess_dataset.py first.")
        self.root, self.use_emo = root, use_emo
        self.labels = self.df.label.values.astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        vid = self.df.video_id[i]
        z = np.load(f"{self.root}/{vid}.va.npz")
        item = dict(v=torch.from_numpy(z["v"]).permute(0, 3, 1, 2).float() / 255.0,
                    a=torch.from_numpy(z["a"].astype(np.float32)).unsqueeze(0), y=int(self.labels[i]))
        if self.use_emo:
            e = np.load(f"{self.root}/{vid}.emo.npz")
            item.update(ie=torch.from_numpy(e["ie"]), ae=torch.from_numpy(e["ae"]), te=torch.from_numpy(e["te"]))
        return item


def build_model(kind, freeze=True):
    return ResNet18_LSTM_MM(freeze) if kind == "baseline" else ProposedMultimodalFramework(freeze)


def forward(model, b, kind, dev):
    v, a = b["v"].to(dev), b["a"].to(dev)
    if kind == "baseline":
        return model(v, a)
    return model(v, a, b["ie"].to(dev), b["ae"].to(dev), b["te"].to(dev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "proposed"], required=True)
    ap.add_argument("--tag", required=True, help="e.g. real or dummy")
    ap.add_argument("--cache", default="processed_real")
    ap.add_argument("--splits_dir", default="splits")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--unfreeze", action="store_true", help="fine-tune ResNet18 (slow on free Colab)")
    a = ap.parse_args()
    torch.manual_seed(42); np.random.seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    use_emo = a.model == "proposed"
    tr = CachedDataset(f"{a.splits_dir}/train.csv", a.cache, use_emo)
    va = CachedDataset(f"{a.splits_dir}/val.csv", a.cache, use_emo)
    counts = np.bincount(tr.labels, minlength=2)
    print(f"train {len(tr)} (dropped {tr.dropped}) | val {len(va)} (dropped {va.dropped}) | train [Real,Fake] = {counts.tolist()}")
    w = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32).to(dev)
    model = build_model(a.model, not a.unfreeze).to(dev)
    print("trainable params:", sum(q.numel() for q in model.parameters() if q.requires_grad))
    opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=a.lr)
    crit = nn.CrossEntropyLoss(weight=w)
    amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    tl = DataLoader(tr, batch_size=a.batch_size, shuffle=True, num_workers=2, drop_last=len(tr) > a.batch_size)
    vl = DataLoader(va, batch_size=a.batch_size, num_workers=2)
    os.makedirs("results", exist_ok=True)
    wpath = f"best_{a.model}_{a.tag}.pth"
    best, bad, hist = -1.0, 0, []
    for ep in range(1, a.epochs + 1):
        out = {}
        for phase, loader in (("train", tl), ("val", vl)):
            model.train(phase == "train")
            tot, ys, ps = 0.0, [], []
            for b in loader:
                y = b["y"].to(dev)
                with torch.set_grad_enabled(phase == "train"), torch.autocast("cuda", enabled=amp):
                    logits = forward(model, b, a.model, dev)
                    loss = crit(logits.float(), y)
                if phase == "train":
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(opt); scaler.update()
                tot += loss.item() * len(y); ys.append(y.cpu()); ps.append(logits.argmax(1).cpu())
            y_, p_ = torch.cat(ys).numpy(), torch.cat(ps).numpy()
            out[phase] = (tot / len(y_), float((y_ == p_).mean()), balanced_accuracy_score(y_, p_))
        hist.append(dict(epoch=ep, **{f"{k}_{n}": v for k, vals in out.items() for n, v in zip(("loss", "acc", "bal_acc"), vals)}))
        flag = ""
        if out["val"][2] > best:
            best, bad, flag = out["val"][2], 0, "  <- best, saved"
            torch.save(model.state_dict(), wpath)
        else:
            bad += 1
        print(f"ep {ep:02d} | train loss {out['train'][0]:.4f} bal {out['train'][2]:.3f} | val loss {out['val'][0]:.4f} bal {out['val'][2]:.3f}{flag}")
        if bad >= a.patience:
            print("early stop"); break
    pd.DataFrame(hist).to_csv(f"results/history_{a.model}_{a.tag}.csv", index=False)
    json.dump(dict(model=a.model, tag=a.tag, best_val_balanced_accuracy=best, weights=wpath, train_counts=counts.tolist()),
              open(f"results/train_{a.model}_{a.tag}.json", "w"), indent=2)
    print(f"\nbest val balanced accuracy {best:.4f} -> {wpath}. Test split NOT used.")


if __name__ == "__main__":
    main()
