"""Data contract, fitting, checkpoint selection and evaluation for external panels.

The caller supplies an already aligned panel. No surveillance or movement data
are distributed with this module.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

from .config import FEATURE_NAMES, model_config
from .evaluation import forecast_metrics
from .loss import CoVFourLoss
from .model import ForecastBackbone, MosquitoGraphormer


ORIGIN_FIELDS = (
    "x", "history", "mobility", "population", "prior_a",
    "prior_sigma_v", "prior_mu_v", "prior_M_z", "target",
)
STATIC_FIELDS = ("distance", "geo_support")


def load_panel(path: str | Path) -> dict[str, np.ndarray]:
    """Load one externally prepared panel with split codes 0=train, 1=dev, 2=test.

    ``x`` and ``history`` cover the prior 14 observations, oldest first.
    ``mobility`` has origin rows and destination columns for those same days.
    All target horizons start strictly after the forecast origin. Callers are
    responsible for constructing and auditing the temporal split.
    """
    with np.load(path, allow_pickle=False) as source:
        required = set(ORIGIN_FIELDS) | set(STATIC_FIELDS) | {"split"}
        missing = required - set(source.files)
        if missing:
            raise ValueError(f"Missing panel arrays: {sorted(missing)}")
        panel = {key: np.asarray(source[key]) for key in required}
        if "mask" in source.files:
            panel["mask"] = np.asarray(source["mask"])
    origins, history_days, nodes, features = panel["x"].shape
    if history_days != 14 or features != len(FEATURE_NAMES):
        raise ValueError("x must have shape [origins,14,nodes,8]")
    expected = {
        "history": (origins, 14, nodes),
        "mobility": (origins, 14, nodes, nodes),
        "population": (origins, nodes),
        "prior_a": (origins, nodes),
        "prior_sigma_v": (origins, nodes),
        "prior_mu_v": (origins, nodes),
        "prior_M_z": (origins, nodes),
        "target": (origins, nodes, 7),
        "distance": (nodes, nodes),
        "geo_support": (nodes, nodes),
        "split": (origins,),
    }
    for key, shape in expected.items():
        if panel[key].shape != shape:
            raise ValueError(f"{key} must have shape {shape}, got {panel[key].shape}")
    for key in ORIGIN_FIELDS + ("distance",):
        if not np.isfinite(panel[key]).all():
            raise ValueError(f"{key} contains nonfinite values")
    if np.any(panel["target"] < 0) or np.any(panel["history"] < 0):
        raise ValueError("Case counts must be nonnegative")
    if set(np.unique(panel["split"])) != {0, 1, 2}:
        raise ValueError("Panel requires nonempty train, development and test splits")
    if "mask" not in panel:
        panel["mask"] = np.ones_like(panel["target"], dtype=bool)
    if panel["mask"].shape != panel["target"].shape:
        raise ValueError("mask must match target shape")
    panel["mask"] = panel["mask"].astype(bool)
    for code in (0, 1, 2):
        if not panel["mask"][panel["split"] == code].any():
            raise ValueError("Each split needs at least one observed target")
    panel["geo_support"] = panel["geo_support"].astype(bool)
    return panel


def _batch(panel: dict[str, np.ndarray], indices: np.ndarray, device: torch.device) -> dict:
    result = {
        key: torch.as_tensor(panel[key][indices], dtype=torch.float32, device=device)
        for key in ORIGIN_FIELDS if key != "target"
    }
    result["distance"] = torch.as_tensor(panel["distance"], dtype=torch.float32, device=device)
    result["geo_support"] = torch.as_tensor(panel["geo_support"], dtype=torch.bool, device=device)
    return result


@torch.no_grad()
def predict_split(model: MosquitoGraphormer, panel: dict, split: int,
                  batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ids = np.flatnonzero(panel["split"] == split)
    predictions = []
    for start in range(0, len(ids), batch_size):
        batch = _batch(panel, ids[start:start + batch_size], device)
        predictions.append(model(batch)["raw_prediction"].cpu().numpy())
    return np.concatenate(predictions), panel["target"][ids]


def fit_and_evaluate(panel: dict, output_dir: str | Path, seed: int,
                     *, max_epochs: int = 60, batch_size: int = 16,
                     learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                     patience: int = 15, device_name: str = "cpu") -> dict:
    """Fit one seed with CoV4 and select its checkpoint by development RMSE.

    No test target is read by the optimisation or checkpoint-selection loop.
    This general entry point does not encode study-specific split boundaries.
    """
    if min(max_epochs, batch_size, patience) < 1:
        raise ValueError("Epochs, batch size and patience must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_name)
    cfg = model_config()
    model = MosquitoGraphormer(
        ForecastBackbone(cfg, len(FEATURE_NAMES), panel["x"].shape[2], FEATURE_NAMES)
    ).to(device)
    objective = CoVFourLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_ids = np.flatnonzero(panel["split"] == 0)
    rng = np.random.default_rng(seed)
    best_rmse = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        objective.train()
        for indices in np.array_split(rng.permutation(train_ids),
                                      max(1, int(np.ceil(len(train_ids) / batch_size)))):
            batch = _batch(panel, indices, device)
            target = torch.as_tensor(panel["target"][indices], dtype=torch.float32, device=device)
            mask = torch.as_tensor(panel["mask"][indices], dtype=torch.bool, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = objective(model(batch), target, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        predicted, observed = predict_split(model, panel, 1, batch_size, device)
        dev_mask = panel["mask"][panel["split"] == 1]
        rmse = float(np.sqrt(np.mean((predicted[dev_mask] - observed[dev_mask]) ** 2)))
        if np.isfinite(rmse) and rmse < best_rmse:
            best_rmse, best_epoch = rmse, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise FloatingPointError("No finite development checkpoint")
    model.load_state_dict(best_state)
    predicted, observed = predict_split(model, panel, 2, batch_size, device)
    test_mask = panel["mask"][panel["split"] == 2]
    scores = {"pooled": forecast_metrics(observed[test_mask], predicted[test_mask])}
    for h in range(7):
        valid = test_mask[..., h]
        scores[f"h{h + 1}"] = (forecast_metrics(observed[..., h][valid], predicted[..., h][valid])
                              if valid.any() else None)
    torch.save({"model": best_state, "seed": seed, "epoch": best_epoch,
                "config": cfg}, output_dir / "checkpoint.pt")
    np.savez_compressed(output_dir / "test_predictions.npz",
                        predicted=predicted, observed=observed, mask=test_mask)
    report = {"seed": seed, "best_epoch": best_epoch,
              "development_rmse": best_rmse, "metrics": scores}
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    return report
