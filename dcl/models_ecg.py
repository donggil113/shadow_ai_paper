"""ECG waveform encoder for the MIMIC-IV-ECG x MIMIC-IV-ECHO arm (T1).

Self-contained (torch + numpy + sklearn only), CPU-runnable on tiny inputs.

* :class:`ECGEncoder` -- 1-D ResNet-18-style network over ``(12, 5000)``
  (12 leads, 10 s at 500 Hz): stem conv -> 4 stages of 2 basic blocks each
  (widths ``base_width * (1, 2, 4, 8)``) -> global average pooling -> 256-d
  embedding -> linear head.  ``forward`` returns logits, ``embed`` the 256-d
  embedding.
* :class:`ECGDataset` -- reads one record per ``file_name`` of
  ``record_list.csv``.  WFDB records (``<path>.hea`` + ``.dat``) are read with
  the ``wfdb`` package when it is installed; a ``<path>.npy`` array (used by the
  pytest fixture, which must not depend on ``wfdb``) is the documented fallback.
  A missing record exits with code 2 and a precise message (CLAUDE.md rule 1);
  nothing is ever simulated.
* :func:`embed_records` -- batched, gradient-free embeddings as an ndarray.
* :func:`train_ecg_classifier` -- trains one encoder per seed with early
  stopping on a held-out **subject-level** split (records of one subject are
  never on both sides).
"""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "N_LEADS", "N_SAMPLES", "SAMPLING_HZ", "EMBEDDING_DIM", "EXIT_MISSING_DATA",
    "BasicBlock1d", "ECGEncoder", "ECGDataset", "read_waveform", "embed_records",
    "subject_split", "TrainResult", "train_ecg_classifier", "predict_proba",
]

N_LEADS = 12
N_SAMPLES = 5000
SAMPLING_HZ = 500
EMBEDDING_DIM = 256
EXIT_MISSING_DATA = 2
_WFDB_URL = "https://physionet.org/content/mimic-iv-ecg/1.0/"


# ================================================================ network
class BasicBlock1d(nn.Module):
    """ResNet basic block with two ``kernel_size`` convolutions (1-D)."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False),
                                          nn.BatchNorm1d(c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ECGEncoder(nn.Module):
    """1-D ResNet-18-style encoder: ``(N, 12, L) -> logits (N,)`` / ``(N, n_outputs)``.

    Parameters
    ----------
    in_channels : int            number of leads (12)
    base_width : int             channels of the first stage (64 = ResNet-18;
                                 use 8 for CPU tests)
    layers : tuple of 4 ints     blocks per stage (2, 2, 2, 2) = ResNet-18
    embedding_dim : int          size of the embedding after global pooling (256)
    n_outputs : int              head width; with 1 the logits are squeezed to (N,)
    kernel_size : int            convolution width inside the blocks
    dropout : float              dropout on the embedding before the head
    """

    def __init__(self, in_channels: int = N_LEADS, base_width: int = 64,
                 layers: Sequence[int] = (2, 2, 2, 2), embedding_dim: int = EMBEDDING_DIM,
                 n_outputs: int = 1, kernel_size: int = 7, dropout: float = 0.0):
        super().__init__()
        if len(layers) != 4:
            raise ValueError("layers must have 4 entries (ResNet-18 style)")
        self.in_channels = int(in_channels)
        self.embedding_dim = int(embedding_dim)
        self.n_outputs = int(n_outputs)
        w = int(base_width)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, w, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(w), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        stages = []
        c_in = w
        for i, (n_blocks, mult) in enumerate(zip(layers, (1, 2, 4, 8))):
            c_out = w * mult
            stride = 1 if i == 0 else 2
            blocks = [BasicBlock1d(c_in, c_out, stride, kernel_size)]
            blocks += [BasicBlock1d(c_out, c_out, 1, kernel_size) for _ in range(n_blocks - 1)]
            stages.append(nn.Sequential(*blocks))
            c_in = c_out
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(nn.Linear(c_in, self.embedding_dim), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(self.embedding_dim, self.n_outputs)

    def _check(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected input of shape (N, {self.in_channels}, L), got {tuple(x.shape)}")
        return x.float()

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, in_channels, L) -> (N, embedding_dim)``."""
        x = self._check(x)
        h = self.stages(self.stem(x))
        return self.proj(self.pool(h).flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(self.dropout(self.embed(x)))
        return out.squeeze(1) if self.n_outputs == 1 else out


# ================================================================= data
def _wfdb_module():
    try:
        import wfdb  # type: ignore
    except ImportError:
        return None
    return wfdb


def read_waveform(path: str, n_leads: int = N_LEADS, n_samples: Optional[int] = N_SAMPLES,
                  ) -> np.ndarray:
    """Read one record as a float32 array of shape ``(n_leads, n_samples)``.

    ``path`` is the record path without extension (``record_list.file_name``
    joined to the dataset root).  Resolution order:

    1. ``<path>.hea`` exists -> WFDB via ``wfdb.rdrecord`` (physical units,
       ``p_signal`` of shape (samples, leads)).  If ``wfdb`` is not installed an
       ``ImportError`` says so.
    2. ``<path>.npy`` exists -> ``numpy.load``; either orientation
       ``(leads, samples)`` or ``(samples, leads)`` is accepted.  This is the
       documented WFDB-free fallback used by the pytest fixture.
    3. neither -> ``FileNotFoundError`` naming both candidates.

    NaN samples are set to 0.  When ``n_samples`` is given the length must match
    exactly (``ValueError`` otherwise; the MIMIC-IV-ECG records are 10 s at
    500 Hz).
    """
    hea, npy = path + ".hea", path + ".npy"
    if os.path.isfile(hea):
        wfdb = _wfdb_module()
        if wfdb is None:
            raise ImportError(f"{hea} is a WFDB record but the 'wfdb' package is not installed "
                              "(pip install wfdb)")
        rec = wfdb.rdrecord(path)
        sig = np.asarray(rec.p_signal, dtype=np.float32)          # (samples, leads)
        if int(getattr(rec, "fs", SAMPLING_HZ)) != SAMPLING_HZ:
            raise ValueError(f"{path}: sampling rate {rec.fs} Hz, expected {SAMPLING_HZ}")
    elif os.path.isfile(npy):
        sig = np.asarray(np.load(npy), dtype=np.float32)
    else:
        raise FileNotFoundError(f"no waveform for record {path!r}: looked for {hea} "
                                f"(+ .dat, WFDB) and {npy} (numpy fallback)")
    if sig.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D array, got shape {sig.shape}")
    if sig.shape[0] != n_leads and sig.shape[1] == n_leads:
        sig = sig.T
    if sig.shape[0] != n_leads:
        raise ValueError(f"{path}: expected {n_leads} leads, got shape {sig.shape}")
    if n_samples is not None and sig.shape[1] != n_samples:
        raise ValueError(f"{path}: expected {n_samples} samples per lead, got {sig.shape[1]}")
    return np.nan_to_num(np.ascontiguousarray(sig), nan=0.0, posinf=0.0, neginf=0.0)


class ECGDataset(Dataset):
    """Waveforms of a ``record_list`` as ``(12, 5000)`` float32 tensors.

    Parameters
    ----------
    file_names : sequence of str
        ``record_list.file_name`` values (relative record paths without
        extension).
    root : str
        Directory the paths are relative to (``<data-root>/mimic-iv-ecg/1.0``
        on the GPU server, the fixture directory in tests).
    y : array-like, optional
        Labels; when given ``__getitem__`` returns ``(x, y)``.
    normalize : bool
        Per-record, per-lead standardisation (zero mean, unit sd; constant
        leads stay 0).
    on_missing : {"exit", "raise"}
        Existence of every record is checked up front.  ``"exit"`` prints the
        missing paths and ``sys.exit(2)`` (CLAUDE.md rule 1); ``"raise"`` raises
        ``FileNotFoundError`` (tests).
    """

    def __init__(self, file_names: Sequence[str], root: str, y: Optional[Sequence[float]] = None,
                 normalize: bool = True, on_missing: str = "exit",
                 n_leads: int = N_LEADS, n_samples: Optional[int] = N_SAMPLES):
        self.root = os.path.abspath(os.path.expanduser(str(root)))
        self.paths = [os.path.join(self.root, str(f)) for f in file_names]
        self.y = None if y is None else np.asarray(y, dtype=np.float32).reshape(-1)
        if self.y is not None and len(self.y) != len(self.paths):
            raise ValueError("y must have one entry per record")
        self.normalize = bool(normalize)
        self.n_leads, self.n_samples = int(n_leads), n_samples
        if on_missing not in ("exit", "raise"):
            raise ValueError("on_missing must be 'exit' or 'raise'")
        missing = self.missing_files()
        if missing:
            msg = "\n".join(
                ["[dcl.models_ecg] REQUIRED ECG WAVEFORM FILES ARE MISSING",
                 f"  root         : {self.root}",
                 f"  n missing    : {len(missing)} of {len(self.paths)}",
                 "  first missing: " + "; ".join(f"{p}.hea/.dat (or .npy)" for p in missing[:5]),
                 f"  obtain from  : {_WFDB_URL} (mimic-iv-ecg v1.0, credentialed PhysioNet resource)",
                 "  No simulator is substituted (CLAUDE.md rule 1). Exiting with code 2."])
            if on_missing == "raise":
                raise FileNotFoundError(msg)
            print(msg, file=sys.stderr, flush=True)
            sys.exit(EXIT_MISSING_DATA)
        self.uses_wfdb = any(os.path.isfile(p + ".hea") for p in self.paths)
        if self.uses_wfdb and _wfdb_module() is None:
            raise ImportError("WFDB records present but the 'wfdb' package is not installed "
                              "(pip install wfdb)")

    def missing_files(self) -> List[str]:
        return [p for p in self.paths if not (os.path.isfile(p + ".hea") or os.path.isfile(p + ".npy"))]

    def __len__(self) -> int:
        return len(self.paths)

    def waveform(self, i: int) -> np.ndarray:
        sig = read_waveform(self.paths[i], self.n_leads, self.n_samples)
        if self.normalize:
            mu = sig.mean(axis=1, keepdims=True)
            sd = sig.std(axis=1, keepdims=True)
            sig = (sig - mu) / np.where(sd > 0, sd, 1.0)
        return sig.astype(np.float32)

    def __getitem__(self, i: int):
        x = torch.from_numpy(self.waveform(i))
        if self.y is None:
            return x
        return x, torch.tensor(self.y[i], dtype=torch.float32)


class _ArrayDataset(Dataset):
    """In-memory ``(N, C, L)`` array as a Dataset (optionally with labels)."""

    def __init__(self, X: Union[np.ndarray, torch.Tensor], y: Optional[Sequence[float]] = None):
        self.X = torch.as_tensor(np.asarray(X) if not isinstance(X, torch.Tensor) else X).float()
        if self.X.dim() != 3:
            raise ValueError(f"waveform array must have shape (N, C, L), got {tuple(self.X.shape)}")
        self.y = None if y is None else torch.as_tensor(np.asarray(y, dtype=np.float32)).reshape(-1)
        if self.y is not None and len(self.y) != len(self.X):
            raise ValueError("y must have one entry per record")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, i: int):
        return self.X[i] if self.y is None else (self.X[i], self.y[i])


class _Indexed(Dataset):
    """Subset of a waveform dataset (labels supplied separately)."""

    def __init__(self, base: Dataset, idx: np.ndarray, y: np.ndarray):
        self.base, self.idx, self.y = base, np.asarray(idx, dtype=int), np.asarray(y, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, k: int):
        item = self.base[int(self.idx[k])]
        x = item[0] if isinstance(item, (tuple, list)) else item
        return x, torch.tensor(self.y[k], dtype=torch.float32)


def _as_dataset(X_wave: Union[np.ndarray, torch.Tensor, Dataset]) -> Dataset:
    return X_wave if isinstance(X_wave, Dataset) else _ArrayDataset(X_wave)


def _device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================ embeddings
@torch.no_grad()
def embed_records(encoder: ECGEncoder, records: Union[np.ndarray, torch.Tensor, Dataset],
                  batch_size: int = 64, device: Optional[Union[str, torch.device]] = None,
                  num_workers: int = 0) -> np.ndarray:
    """Embeddings ``(N, embedding_dim)`` (float32 ndarray) of every record.

    ``records`` is an :class:`ECGDataset`, any Dataset yielding ``(12, L)``
    tensors (or ``(x, y)`` pairs), or an array ``(N, 12, L)``.  The encoder is
    put in eval mode and left there.
    """
    dev = _device(device)
    encoder = encoder.to(dev).eval()
    ds = _as_dataset(records)
    out: List[np.ndarray] = []
    for batch in DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=num_workers):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        out.append(encoder.embed(x.to(dev)).cpu().numpy().astype(np.float32))
    if not out:
        return np.zeros((0, encoder.embedding_dim), dtype=np.float32)
    return np.concatenate(out, axis=0)


# ============================================================== training
def subject_split(subject_ids: Sequence[Any], val_frac: float = 0.2, seed: int = 0,
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Record indices ``(train_idx, val_idx)`` from a **subject-level** split.

    Unique subjects are shuffled with ``seed`` and ``ceil(val_frac * n_subjects)``
    of them (at least one, at most n - 1) form the validation set; every record
    of a subject goes to the side its subject was assigned to, so the two index
    sets never share a subject.
    """
    sid = np.asarray(subject_ids)
    if sid.ndim != 1 or len(sid) == 0:
        raise ValueError("subject_ids must be a non-empty 1-D array")
    uniq = np.unique(sid)
    if len(uniq) < 2:
        raise ValueError("a subject-level split needs at least 2 distinct subjects")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(uniq)
    n_val = int(np.ceil(float(val_frac) * len(uniq)))
    n_val = min(max(n_val, 1), len(uniq) - 1)
    val_subjects = set(perm[:n_val].tolist())
    is_val = np.array([s in val_subjects for s in sid.tolist()], dtype=bool)
    train_idx, val_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    assert not (set(sid[train_idx].tolist()) & set(sid[val_idx].tolist()))
    return train_idx, val_idx


@dataclass
class TrainResult:
    """One trained :class:`ECGEncoder` per seed plus the early-stopping record."""

    models: List[ECGEncoder] = field(default_factory=list)
    seeds: List[int] = field(default_factory=list)
    val_auc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    best_epoch: List[int] = field(default_factory=list)
    history: List[List[Dict[str, float]]] = field(default_factory=list)
    train_subjects: List[np.ndarray] = field(default_factory=list)
    val_subjects: List[np.ndarray] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, X_wave, batch_size: int = 64, device=None) -> np.ndarray:
        """Seed-averaged ``P(y = 1 | x)`` for every record of ``X_wave``."""
        return predict_proba(self.models, X_wave, batch_size=batch_size, device=device)

    def summary(self) -> Dict[str, Any]:
        return {
            "seeds": list(self.seeds), "val_auc": list(self.val_auc),
            "val_loss": list(self.val_loss), "best_epoch": list(self.best_epoch),
            "n_train_subjects": [int(len(s)) for s in self.train_subjects],
            "n_val_subjects": [int(len(s)) for s in self.val_subjects],
            "config": dict(self.config),
        }


@torch.no_grad()
def predict_proba(models: Sequence[ECGEncoder], X_wave, batch_size: int = 64, device=None,
                  ) -> np.ndarray:
    """Mean sigmoid(logit) over ``models`` for every record (float64 ndarray)."""
    dev = _device(device)
    ds = _as_dataset(X_wave)
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False)
    probs = []
    for m in models:
        m = m.to(dev).eval()
        chunks = []
        for batch in loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            chunks.append(torch.sigmoid(m(x.to(dev))).reshape(-1).cpu().numpy())
        probs.append(np.concatenate(chunks) if chunks else np.zeros(0))
    return np.mean(np.stack(probs, axis=0), axis=0).astype(float)


@torch.no_grad()
def _eval_loss(model: ECGEncoder, loader: DataLoader, dev: torch.device
               ) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    lossf = nn.BCEWithLogitsLoss(reduction="sum")
    tot, n, logits, ys = 0.0, 0, [], []
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        out = model(x).reshape(-1)
        tot += float(lossf(out, y))
        n += int(y.numel())
        logits.append(out.cpu().numpy())
        ys.append(y.cpu().numpy())
    if n == 0:
        return float("nan"), np.zeros(0), np.zeros(0)
    return tot / n, np.concatenate(logits), np.concatenate(ys)


def _safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def train_ecg_classifier(X_wave: Union[np.ndarray, torch.Tensor, Dataset], y: Sequence[float],
                         subject_ids: Sequence[Any], seeds: Sequence[int] = (0, 1, 2, 3, 4),
                         *, val_frac: float = 0.2, epochs: int = 30, patience: int = 5,
                         batch_size: int = 64, lr: float = 1e-3, weight_decay: float = 1e-4,
                         device: Optional[Union[str, torch.device]] = None,
                         num_workers: int = 0, verbose: bool = False,
                         **encoder_kwargs: Any) -> TrainResult:
    """Train one :class:`ECGEncoder` per seed with subject-level early stopping.

    Parameters
    ----------
    X_wave : array ``(N, 12, L)`` or a Dataset yielding ``(12, L)`` tensors
        (an :class:`ECGDataset`; labels inside the dataset are ignored, ``y``
        is authoritative).
    y : array ``(N,)`` of 0/1 labels.
    subject_ids : array ``(N,)``
        Used for the held-out split: **the split is by subject, never by
        record**, so no subject contributes to both the training and the
        early-stopping set (``subject_split``).
    seeds : sequence of int
        One model per seed; the seed drives the subject split, the
        initialisation and the batch order.  Five seeds by default (CLAUDE.md
        rule 4).
    epochs, patience : early stopping on the validation BCE loss; the state
        with the lowest validation loss is restored.
    **encoder_kwargs : forwarded to :class:`ECGEncoder` (e.g. ``base_width=8``
        for CPU tests).
    """
    ds = _as_dataset(X_wave)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    sid = np.asarray(subject_ids)
    if not (len(ds) == len(y) == len(sid)):
        raise ValueError(f"X_wave ({len(ds)}), y ({len(y)}) and subject_ids ({len(sid)}) "
                         "must have the same length")
    if not set(np.unique(y).tolist()) <= {0.0, 1.0}:
        raise ValueError("y must be binary 0/1")
    dev = _device(device)
    res = TrainResult(config={"val_frac": val_frac, "epochs": epochs, "patience": patience,
                              "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
                              "encoder_kwargs": dict(encoder_kwargs), "device": str(dev),
                              "split": "subject-level (subject_split)"})
    for seed in seeds:
        seed = int(seed)
        tr_idx, va_idx = subject_split(sid, val_frac=val_frac, seed=seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = ECGEncoder(n_outputs=1, **encoder_kwargs).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        lossf = nn.BCEWithLogitsLoss()
        g = torch.Generator().manual_seed(seed)
        tr_loader = DataLoader(_Indexed(ds, tr_idx, y[tr_idx]), batch_size=int(batch_size),
                               shuffle=True, generator=g, num_workers=num_workers, drop_last=False)
        va_loader = DataLoader(_Indexed(ds, va_idx, y[va_idx]), batch_size=int(batch_size),
                               shuffle=False, num_workers=num_workers)
        best_loss, best_state, best_ep, bad, hist = float("inf"), None, -1, 0, []
        for ep in range(int(epochs)):
            model.train()
            tot, n = 0.0, 0
            for x, yb in tr_loader:
                x, yb = x.to(dev), yb.to(dev)
                opt.zero_grad(set_to_none=True)
                loss = lossf(model(x).reshape(-1), yb)
                loss.backward()
                opt.step()
                tot += float(loss) * int(yb.numel())
                n += int(yb.numel())
            va_loss, va_logit, va_y = _eval_loss(model, va_loader, dev)
            va_auc = _safe_auc(va_y, va_logit)
            hist.append({"epoch": ep, "train_loss": tot / max(n, 1),
                         "val_loss": va_loss, "val_auc": va_auc})
            if verbose:
                print(f"[train_ecg_classifier] seed={seed} epoch={ep} train_loss={tot / max(n, 1):.4f} "
                      f"val_loss={va_loss:.4f} val_auc={va_auc:.3f}", flush=True)
            if np.isfinite(va_loss) and va_loss < best_loss - 1e-6:
                best_loss, best_ep, bad = va_loss, ep, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if bad >= int(patience):
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        _, va_logit, va_y = _eval_loss(model, va_loader, dev)
        res.models.append(model)
        res.seeds.append(seed)
        res.val_loss.append(float(best_loss) if np.isfinite(best_loss) else float("nan"))
        res.val_auc.append(_safe_auc(va_y, va_logit))
        res.best_epoch.append(int(best_ep))
        res.history.append(hist)
        res.train_subjects.append(np.unique(sid[tr_idx]))
        res.val_subjects.append(np.unique(sid[va_idx]))
        assert not (set(res.train_subjects[-1].tolist()) & set(res.val_subjects[-1].tolist()))
    return res
