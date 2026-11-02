"""Train an MLP + GRU path encoder with geometry reconstruction targets.

Run from the repository root:
    python3 trajectories/train_path_encoder.py

The default encoder reads body-relative trajectory points. The prescribed
speed profile is excluded because path shape does not determine speed.
"""

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


class PathEncoder(nn.Module):
    """Encode a masked path window to one fixed-size latent vector."""

    def __init__(self, input_dim=6, point_dim=64, hidden_dim=96, latent_dim=32):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, point_dim), nn.GELU(),
            nn.Linear(point_dim, point_dim), nn.GELU(),
        )
        self.gru = nn.GRU(point_dim, hidden_dim, batch_first=True,
                          bidirectional=True)
        self.latent_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, latent_dim), nn.LayerNorm(latent_dim),
        )

    def forward(self, path_window):
        valid = path_window[..., -1]
        lengths = valid.sum(dim=1).round().clamp(min=1).long().cpu()
        points = self.point_mlp(path_window)
        packed = pack_padded_sequence(points, lengths, batch_first=True,
                                      enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.latent_head(torch.cat((hidden[-2], hidden[-1]), dim=-1))


class PathAutoencoder(nn.Module):
    """Decode geometry, path derivatives and maneuver-event scores."""

    def __init__(self, input_dim=6, point_dim=64, hidden_dim=96, latent_dim=32,
                 reconstruction_dim=5):
        super().__init__()
        self.encoder = PathEncoder(input_dim, point_dim, hidden_dim, latent_dim)
        self.decoder_gru = nn.GRU(latent_dim + 2, hidden_dim, batch_first=True)
        self.decoder_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
        )
        self.geometry_head = nn.Linear(hidden_dim // 2, reconstruction_dim)
        self.curvature_head = nn.Linear(hidden_dim // 2, 2)
        self.event_head = nn.Linear(hidden_dim // 2, 2)
        nn.init.constant_(self.event_head.bias, -2.0)

    def forward(self, path_window):
        latent = self.encoder(path_window)
        batch, steps, _ = path_window.shape
        position = torch.linspace(0.0, 1.0, steps, device=latent.device,
                                  dtype=latent.dtype).view(1, steps, 1)
        position = position.expand(batch, -1, -1)
        decoder_input = torch.cat((
            latent[:, None, :].expand(-1, steps, -1),
            position, path_window[..., -1:],
        ), dim=-1)
        decoded, _ = self.decoder_gru(decoder_input)
        decoded = self.decoder_mlp(decoded)
        return (self.geometry_head(decoded),
                self.curvature_head(decoded),
                self.event_head(decoded))


def losses(prediction, target, valid, event_positive_weight, reconstruction_dim):
    """Return total and named masked losses for valid preview points."""
    geometry, curvature, event_logits = prediction
    mask = valid[..., None]
    count = mask.sum().clamp_min(1.0)
    reconstruction = (F.smooth_l1_loss(geometry, target[..., :reconstruction_dim],
                                       beta=0.05, reduction="none") * mask
                      ).sum() / (reconstruction_dim * count)
    curvature_loss = (F.smooth_l1_loss(
        curvature, target[..., reconstruction_dim:reconstruction_dim + 2],
                                       beta=0.05, reduction="none") * mask
                      ).sum() / (2 * count)
    event_target = target[..., -2:]
    event_weights = 1.0 + event_positive_weight * event_target
    event_loss = (F.binary_cross_entropy_with_logits(
        event_logits, event_target, reduction="none") * event_weights * mask
    ).sum() / (event_weights * mask).sum().clamp_min(1.0)
    total = reconstruction + curvature_loss + 0.1 * event_loss
    return total, reconstruction, curvature_loss, event_loss


def load_dataset(corpus_dir, split, spec, legacy_target, limit, seed):
    inputs = np.load(corpus_dir / spec["input_file"].format(split=split))
    targets = np.load(corpus_dir / spec["target_file"].format(split=split))
    expected_input = tuple(spec["input_shape_per_window"])
    expected_target = tuple(spec["target_shape_per_window"])
    if inputs.shape[1:] != expected_input or targets.shape[1:] != expected_target:
        raise ValueError(f"{split}: array shapes do not match corpus metadata")
    if len(inputs) != len(targets) or not np.isfinite(inputs).all() or not np.isfinite(targets).all():
        raise ValueError(f"{split}: invalid input or target arrays")
    if legacy_target:
        # The original path-aligned targets had a validity channel between
        # reconstruction and auxiliary channels. It is already in the input.
        targets = np.concatenate((targets[..., :5], targets[..., 6:]), axis=-1)
    if limit and limit < len(inputs):
        indices = np.random.default_rng(seed).choice(len(inputs), limit,
                                                     replace=False)
        inputs, targets = inputs[indices], targets[indices]
    return TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))


def make_train_loader(dataset, batch_size, event_oversample, seed, workers):
    targets = dataset.tensors[1]
    event_window = targets[..., -2:].amax(dim=(1, 2)) > 0.2
    weights = torch.ones(len(dataset), dtype=torch.double)
    weights[event_window] = event_oversample
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, len(dataset), replacement=True,
                                    generator=generator)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      num_workers=workers, pin_memory=torch.cuda.is_available())


def run_epoch(model, loader, optimizer, device, event_positive_weight,
              reconstruction_dim):
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(4, dtype=float)
    count = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            prediction = model(inputs)
            components = losses(prediction, targets, inputs[..., -1],
                                event_positive_weight, reconstruction_dim)
            if training:
                optimizer.zero_grad(set_to_none=True)
                components[0].backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        totals += np.array([part.detach().item() for part in components]) * len(inputs)
        count += len(inputs)
    return totals / count


def event_f1(model, loader, device):
    """Compute F1 at score 0.5 for turn-entry and depth-reversal markers."""
    true_positive = np.zeros(2, dtype=np.int64)
    false_positive = np.zeros(2, dtype=np.int64)
    false_negative = np.zeros(2, dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predicted = torch.sigmoid(model(inputs)[2]) > 0.5
            expected = targets[..., -2:] > 0.5
            valid = inputs[..., -1:].bool()
            true_positive += ((predicted & expected) & valid).sum((0, 1)).cpu().numpy()
            false_positive += ((predicted & ~expected) & valid).sum((0, 1)).cpu().numpy()
            false_negative += ((~predicted & expected) & valid).sum((0, 1)).cpu().numpy()
    return (2 * true_positive / np.maximum(1, 2 * true_positive +
                                           false_positive + false_negative)).tolist()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path,
                        default=Path(__file__).with_name("path_corpus"))
    parser.add_argument("--dataset", choices=("agent_relative", "path_aligned"),
                        default="agent_relative",
                        help="agent_relative builds body-frame windows on first use")
    parser.add_argument("--variants-per-anchor", type=int, default=2,
                        help="agent states per path position: one aligned and the rest perturbed")
    parser.add_argument("--output-dir", type=Path,
                        help="new run directory; default is a timestamped directory")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--point-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--event-oversample", type=float, default=3.0)
    parser.add_argument("--event-positive-weight", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20268)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=0,
                        help="optional random subset size for a quick trial")
    parser.add_argument("--limit-val", type=int, default=0,
                        help="optional random validation subset size")
    args = parser.parse_args()
    if (min(args.epochs, args.batch_size, args.latent_dim, args.point_dim,
            args.hidden_dim, args.patience, args.threads,
            args.variants_per_anchor) <= 0 or
            args.learning_rate <= 0 or args.weight_decay < 0 or
            args.event_oversample < 1 or args.event_positive_weight < 0 or
            args.workers < 0 or args.limit_train < 0 or args.limit_val < 0):
        parser.error("invalid nonpositive training setting")
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    metadata = json.loads((args.corpus_dir / "metadata.json").read_text(encoding="utf-8"))
    if "encoder_dataset" not in metadata:
        raise ValueError("run add_encoder_targets.py on this corpus first")
    dataset = args.dataset
    dataset_key = ("agent_relative_encoder_dataset" if dataset == "agent_relative"
                   else "encoder_dataset")
    if dataset == "agent_relative":
        from trajectories.build_agent_relative_corpus import (
            INPUT_FEATURES, RECONSTRUCTION_FEATURES, build)
        spec = metadata.get(dataset_key)
        needs_build = (spec is None or
                       tuple(spec.get("input_features", ())) != INPUT_FEATURES or
                       tuple(spec.get("reconstruction_features", ())) !=
                       RECONSTRUCTION_FEATURES or
                       spec.get("variants_per_anchor") != args.variants_per_anchor or
                       spec.get("synthetic_state_seed") != args.seed or
                       any(not (args.corpus_dir / spec[key].format(split=split)).is_file()
                           for split in ("train", "val", "test")
                           for key in ("input_file", "target_file")))
        if needs_build:
            print("Preparing body-relative path windows from the trajectory corpus...",
                  flush=True)
            build(args.corpus_dir, args.variants_per_anchor, args.seed)
            metadata = json.loads((args.corpus_dir / "metadata.json").read_text(
                encoding="utf-8"))
    spec = metadata[dataset_key]
    legacy_target = dataset == "path_aligned"
    reconstruction_dim = len(spec.get("reconstruction_features",
                                      spec["input_features"][:-1]))
    train_data = load_dataset(args.corpus_dir, "train", spec, legacy_target,
                              args.limit_train, args.seed)
    val_data = load_dataset(args.corpus_dir, "val", spec, legacy_target,
                            args.limit_val, args.seed + 1)
    test_data = load_dataset(args.corpus_dir, "test", spec, legacy_target,
                             0, args.seed + 2)
    train_loader = make_train_loader(train_data, args.batch_size,
                                     args.event_oversample, args.seed,
                                     args.workers)
    val_loader = DataLoader(val_data, batch_size=args.batch_size,
                            num_workers=args.workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size,
                             num_workers=args.workers)

    encoder_config = {"input_dim": len(spec["input_features"]),
                    "point_dim": args.point_dim, "hidden_dim": args.hidden_dim,
                    "latent_dim": args.latent_dim}
    model_config = {**encoder_config, "reconstruction_dim": reconstruction_dim}
    model = PathAutoencoder(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(2, args.patience // 3))
    output_dir = args.output_dir or (
        Path("outputs/path_encoder_mlp_gru") /
        datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    history_file = output_dir / "history.csv"
    with history_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "train_total", "val_total", "val_reconstruction",
                         "val_curvature", "val_events", "learning_rate"))

    best = float("inf")
    best_epoch = 0
    stale = 0
    print(f"device={device}, dataset={dataset}, train={len(train_data)}, val={len(val_data)}, "
          f"test={len(test_data)}, output={output_dir}", flush=True)
    for epoch in range(1, args.epochs + 1):
        train_scores = run_epoch(model, train_loader, optimizer, device,
                                 args.event_positive_weight, reconstruction_dim)
        val_scores = run_epoch(model, val_loader, None, device,
                               args.event_positive_weight, reconstruction_dim)
        scheduler.step(val_scores[0])
        with history_file.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow((epoch, train_scores[0], *val_scores,
                                         optimizer.param_groups[0]["lr"]))
        if val_scores[0] < best - 1e-5:
            best = float(val_scores[0])
            best_epoch = epoch
            stale = 0
            torch.save({"state_dict": model.state_dict(), "config": model_config,
                        "epoch": epoch, "validation_loss": best,
                        "input_features": spec["input_features"],
                        "dataset": dataset},
                       output_dir / "best_autoencoder.pt")
            torch.save({"state_dict": model.encoder.state_dict(),
                        "config": encoder_config, "epoch": epoch,
                        "input_features": spec["input_features"],
                        "dataset": dataset},
                       output_dir / "best_encoder.pt")
        else:
            stale += 1
        print(f"epoch {epoch:03d}: train={train_scores[0]:.5f} "
              f"val={val_scores[0]:.5f} "
              f"(recon={val_scores[1]:.5f}, curve={val_scores[2]:.5f}, "
              f"events={val_scores[3]:.5f}) best={best:.5f}@{best_epoch}",
              flush=True)
        if stale >= args.patience:
            print(f"early stopping after {epoch} epochs", flush=True)
            break

    checkpoint = torch.load(output_dir / "best_autoencoder.pt",
                            map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test_scores = run_epoch(model, test_loader, None, device,
                            args.event_positive_weight, reconstruction_dim)
    val_f1 = event_f1(model, val_loader, device)
    test_f1 = event_f1(model, test_loader, device)
    report = {"best_epoch": best_epoch, "best_validation_loss": best,
              "test_loss": float(test_scores[0]),
              "test_reconstruction_loss": float(test_scores[1]),
              "test_curvature_loss": float(test_scores[2]),
              "test_event_loss": float(test_scores[3]),
              "validation_event_f1": val_f1, "test_event_f1": test_f1,
              "model_config": model_config, "dataset": dataset, "seed": args.seed,
              "corpus_dir": str(args.corpus_dir)}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n",
                                             encoding="utf-8")
    print(f"best epoch={best_epoch}, test={test_scores[0]:.5f}, "
          f"event F1={test_f1}, encoder={output_dir / 'best_encoder.pt'}")


if __name__ == "__main__":
    main()
