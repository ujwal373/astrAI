# training/train_uv.py
import json, pickle, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import savgol_filter
import optuna


# -------------------------
# Paths
# -------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "real"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

UV_PKLS = {
    "JUPITER": DATA_DIR / "jupiter_uv.pkl",
    "SATURN":  DATA_DIR / "saturn_uv.pkl",
    "URANUS":  DATA_DIR / "uranus_uv.pkl",
}


# -------------------------
# Config
# -------------------------
SEED = 7
N_RESAMPLE = 1024
BATCH = 128

N_TRIALS = 25
N_SYNTH_PER_PLANET = 1200   # per trial; total = 3 * this
EPOCHS_TUNE = 12
PATIENCE = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)


# -------------------------
# Species + bands (UV)
# NOTE: You can start with 8 labels if you want (recommended).
# -------------------------
SPECIES = [
    "CH4", "NH3", "C2H2", "C2H6",
    "PH3", "C6H6", "C4H2",
    "CO",
    # Uncomment later (Phase 3)
    # "C2H4", "C3H4", "GeH4", "AsH3",
]

BANDS = {
    "CH4":  [(2350, 260), (2750, 320)],
    "C2H2": [(2700, 140), (2810, 130), (2200, 150)],
    "C2H6": [(2400, 170), (2550, 150)],
    "NH3":  [(2050, 140), (2150, 140), (2250, 100)],
    "PH3":  [(2100, 200), (2300, 150)],
    "C6H6": [(2550, 80), (2600, 60)],
    "C4H2": [(2200, 50), (2320, 50), (2440, 50)],
    "CO":   [(1990, 40), (2060, 40)],
    # Phase 3 (commented)
    # "C2H4": [(1700, 150), (1850, 100)],
    # "C3H4": [(1900, 150), (2050, 100)],
    # "GeH4": [(1950, 200), (2100, 100)],
    # "AsH3": [(2000, 200), (2150, 100)],
}


# Simple priors (keep labels learnable)
PRIORS = {
    "CH4": 0.85,
    "NH3": 0.25,
    "C2H2": 0.18,
    "C2H6": 0.18,
    "PH3": 0.12,
    "C6H6": 0.08,
    "C4H2": 0.08,
    "CO":  0.06,
    # Phase 3 priors (if enabled)
    # "C2H4": 0.06,
    # "C3H4": 0.06,
    # "GeH4": 0.05,
    # "AsH3": 0.05,
}


# -------------------------
# IO helpers (handles different PKL schemas)
# -------------------------
def load_pkl_spectrum(path: Path):
    with open(path, "rb") as f:
        d = pickle.load(f)

    # handle keys: wavelength vs wave
    w = d["wavelength"] if "wavelength" in d else d["wave"]
    f_ = d["flux"]

    w = np.asarray(w, dtype=float)
    f_ = np.asarray(f_, dtype=float)

    m = np.isfinite(w) & np.isfinite(f_)
    w, f_ = w[m], f_[m]

    idx = np.argsort(w)
    w, f_ = w[idx], f_[idx]

    target = d.get("target", d.get("target_name", "unknown"))
    return target, w, f_


def resample(w, f, n=N_RESAMPLE):
    w_new = np.linspace(w.min(), w.max(), n)
    f_new = np.interp(w_new, w, f)
    return w_new.astype(np.float32), f_new.astype(np.float32)


# -------------------------
# Preprocess: baseline + channels
# -------------------------
def compute_baseline(flux, win):
    win = int(win)
    if win % 2 == 0:
        win += 1
    win = max(11, min(win, len(flux) - 1))
    b = savgol_filter(flux.astype(float), win, 3)
    b = np.clip(b, np.percentile(b, 1), np.percentile(b, 99))
    return b.astype(np.float32) + 1e-12


def make_channels(w, f, win):
    base = compute_baseline(f, win)
    r = (f / base) - 1.0
    r = (r - np.median(r)) / (np.std(r) + 1e-8)
    d1 = np.gradient(r, w)
    d2 = np.gradient(d1, w)
    return np.stack([r, d1, d2], axis=0).astype(np.float32)


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# -------------------------
# Synthetic generator anchored to EACH planet
# -------------------------
def build_synth_for_planet(w, f, win, n, seed):
    rng = np.random.default_rng(seed)
    base = compute_baseline(f, win)

    wmin, wmax = float(w.min()), float(w.max())
    bands_in_range = {}
    for sp, blist in BANDS.items():
        keep = [(c, w0) for (c, w0) in blist if (wmin <= c <= wmax)]
        if keep:
            bands_in_range[sp] = keep

    X_list, Y_list = [], []

    for _ in range(n):
        y = {sp: (1 if rng.random() < PRIORS.get(sp, 0.05) else 0) for sp in SPECIES}
        if sum(y.values()) == 0:
            y["CH4"] = 1

        spec = base.copy()

        # mild drift so we don't memorize continuum
        x = (w - w.min()) / (w.max() - w.min())
        spec *= 1.0 + rng.normal(0, 0.01) + rng.normal(0, 0.01) * (x - 0.5)

        # absorption dips
        for sp, present in y.items():
            if not present:
                continue
            for c, w0 in bands_in_range.get(sp, []):
                depth = rng.uniform(0.03, 0.25)
                width = w0 * rng.uniform(0.8, 1.3)
                c_jit = c + rng.normal(0, 15)
                spec *= (1.0 - depth * gaussian(w, c_jit, width))

        # noise (NumPy 2.0-safe)
        sigma = 0.01 * np.ptp(spec)
        noise = rng.normal(0, sigma, size=len(spec))
        noise = np.convolve(noise, np.ones(7) / 7, mode="same")
        spec = spec + noise

        X_list.append(make_channels(w, spec.astype(np.float32), win))
        Y_list.append([y[s] for s in SPECIES])

    return np.stack(X_list), np.asarray(Y_list, dtype=np.float32)


# -------------------------
# Model
# -------------------------
class MLP(nn.Module):
    def __init__(self, h1, h2, drop1, drop2, k):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * N_RESAMPLE, h1),
            nn.ReLU(),
            nn.Dropout(drop1),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(drop2),
            nn.Linear(h2, k),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Objective
# -------------------------
def objective(trial):
    h1 = trial.suggest_categorical("h1", [256, 512, 768])
    h2 = trial.suggest_categorical("h2", [128, 256, 384])
    drop1 = trial.suggest_float("drop1", 0.10, 0.30)
    drop2 = trial.suggest_float("drop2", 0.05, 0.25)
    lr = trial.suggest_float("lr", 1e-4, 5e-4, log=True)
    wd = trial.suggest_float("wd", 1e-6, 1e-4, log=True)
    win = trial.suggest_categorical("baseline_win", [101, 151, 201])

    # build multi-planet synthetic dataset for this trial
    Xs, Ys = [], []
    for i, (planet, pkl_path) in enumerate(UV_PKLS.items()):
        _, w_raw, f_raw = load_pkl_spectrum(pkl_path)
        w, f = resample(w_raw, f_raw)

        Xp, Yp = build_synth_for_planet(
            w, f, win=win, n=N_SYNTH_PER_PLANET, seed=1000 + trial.number * 10 + i
        )
        Xs.append(Xp)
        Ys.append(Yp)

    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)

    perm = np.random.default_rng(999 + trial.number).permutation(len(X))
    X, Y = X[perm], Y[perm]
    n_train = int(0.85 * len(X))

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X[:n_train]), torch.tensor(Y[:n_train])),
        batch_size=BATCH,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X[n_train:]), torch.tensor(Y[n_train:])),
        batch_size=BATCH,
        shuffle=False,
    )

    k = len(SPECIES)
    model = MLP(h1, h2, drop1, drop2, k).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()

    best_loss = 1e9
    bad = 0

    for _ in range(EPOCHS_TUNE):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # val
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                tot += loss_fn(model(xb), yb).item() * xb.size(0)
        vloss = tot / len(val_loader.dataset)

        if vloss < best_loss - 1e-4:
            best_loss = vloss
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    return float(best_loss)


# -------------------------
# Run Optuna + save config
# -------------------------
if __name__ == "__main__":
    print("UV training | device:", DEVICE)
    for p, fp in UV_PKLS.items():
        t, w, f = load_pkl_spectrum(fp)
        print(f" - {p}: {fp.name} | target={t} | w=[{w.min():.2f},{w.max():.2f}] | n={len(w)}")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=25)

    best = study.best_params
    print("\nBest UV params:", best)

    (MODEL_DIR / "uv_config.json").write_text(json.dumps({
        "species": SPECIES,
        "bands": BANDS,
        "priors": PRIORS,
        "best_params": best,
        "n_resample": N_RESAMPLE,
        "seed": SEED,
    }, indent=2))

    print("\nSaved:", MODEL_DIR / "uv_config.json")
