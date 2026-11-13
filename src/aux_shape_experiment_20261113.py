#!/usr/bin/env python3
"""Test auxiliary age-profile shape while reusing the frozen curve2 controls.

This separate study adds 75 unitless auxiliary profile values and five
degeneracy indicators to the existing 36 curve2 features. It trains exactly
two new models, with the same seeds/configuration/partition as the completed
transfer experiment. It never changes that experiment, its data, or its models.

Auxiliary profiles are ordered soil, lai, gpp, npp, rh, with 15 ages per
variable. For an active profile, q=(x-min(x))/(max(x)-min(x)) in float64.
An inactive profile has range <= 8 * max spacing(abs(float32(x))); its shape
is zero and its indicator is one. No physical clamp or absolute epsilon is
used in this transformation. Nonfinite inputs and float32 overflow are errors.

For positive affine transformations of an active profile, q is algebraically
invariant in exact arithmetic. The resolution-based degeneracy indicator is
NOT universally affine invariant: translation can change float32 spacing.

Actions are explicit. ``smoke`` uses synthetic inputs and CPU only. ``prepare``
freezes a new plan after the parent experiment has completed. ``run`` acquires
the existing IEEE GPU lock, performs two new fits and reports a descriptive
comparison on the already-used audit fold. No action submits to Kaggle.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import json
import logging
import os
import random
import shutil
import signal
import sys
import tempfile
from pathlib import Path
from typing import Any

for _thread_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_name] = "4"

import numpy as np


STUDY = "curve2_aux_shape_20261113_v1"
ARM = "curve2_aux_shape"
ENGINE_SHA256 = "230680807f1ec9a3ccd89c8739ac149284cb5c1b5eaa5b11707fcbd213fa2905"
AUXILIARY_NAMES = ("soil", "lai", "gpp", "npp", "rh")
SEEDS = (20261113, 20261114)
FEATURE_DIMENSION = 116
LOG = logging.getLogger(STUDY)


def load_engine(path: Path) -> Any:
    """Import the verified immutable parent engine without replacing any functions."""
    import hashlib
    if hashlib.sha256(path.read_bytes()).hexdigest() != ENGINE_SHA256:
        raise RuntimeError("The parent engine is not the frozen, verified implementation")
    specification = importlib.util.spec_from_file_location("frozen_carbon_transfer_engine", path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load parent engine: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def auxiliary_shape(auxiliary: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Return variable-major 75 shapes followed by five precision-aware flags.

    np.spacing(float32_max) is infinite. Only those entries use the backward
    representable spacing, computed by a float64 subtraction. Subnormals and
    zero use their actual positive float32 spacing, without an epsilon floor.
    """
    values = np.asarray(auxiliary)
    if values.ndim != 3 or values.shape[1:] != (15, 5):
        raise ValueError(f"Expected (sites,15,5) auxiliary profiles, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Auxiliary profile contains a nonfinite original value")
    values64 = values.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        values32 = values.astype(np.float32)
    if not np.isfinite(values32).all():
        raise ValueError("Auxiliary profile cannot be represented as finite float32")
    absolute32 = np.abs(values32)
    with np.errstate(over="ignore", invalid="ignore"):
        spacing = np.spacing(absolute32).astype(np.float64)
    overflow_spacing = np.isinf(spacing)
    if overflow_spacing.any():
        lower = np.nextafter(absolute32[overflow_spacing], np.float32(0)).astype(np.float64)
        spacing[overflow_spacing] = absolute32[overflow_spacing].astype(np.float64) - lower
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise FloatingPointError("Float32 spacing is not finite and strictly positive")
    minimum = values64.min(axis=1)
    extent = values64.max(axis=1) - minimum
    threshold = 8.0 * spacing.max(axis=1)
    degenerate = extent <= threshold
    numerator = values64 - minimum[:, None, :]
    shape = np.zeros_like(values64)
    np.divide(numerator, extent[:, None, :], out=shape, where=~degenerate[:, None, :])
    np.clip(shape, 0.0, 1.0, out=shape)
    ordered = shape.transpose(0, 2, 1).reshape(len(values), 75)
    result = np.concatenate([ordered, degenerate.astype(np.float64)], axis=1).astype(np.float32)
    diagnostics = {
        "sites": len(values), "profiles": int(degenerate.size),
        "degenerate_by_variable": {name: int(degenerate[:, index].sum())
                                   for index, name in enumerate(AUXILIARY_NAMES)},
        "backward_spacing_fallback_values": int(overflow_spacing.sum()),
    }
    return result, diagnostics


def feature_matrix(state: np.ndarray, engine: Any) -> np.ndarray:
    """Append auxiliary shape to the exact frozen curve2 features for each age."""
    state = np.asarray(state)
    if state.ndim != 3 or state.shape[1:] != (15, 7) or not np.isfinite(state).all():
        raise ValueError("Expected a finite site x 15 x 7 initial state")
    # Calculate shapes from the original values, before the parent helper's
    # unrelated sentinel handling of its height/AGB feature block.
    shape, _ = auxiliary_shape(state[:, :, 2:7])
    curve = engine.features(None, state, "curve2").reshape(len(state), 15, 36)
    repeated = np.broadcast_to(shape[:, None, :], (len(state), 15, 80))
    return np.concatenate([curve, repeated], axis=2).reshape(-1, FEATURE_DIMENSION).astype(np.float32)


def decode_initial_state(state: np.ndarray, transform: str, engine: Any) -> np.ndarray:
    """Decode target channels only, preserving every auxiliary input without clamping."""
    state = np.asarray(state)
    if state.shape[1:] != (15, 7) or not np.isfinite(state).all():
        raise ValueError("Initial states must be finite and have shape (sites,15,7)")
    result = state.copy()
    if transform == "community_affine":
        engine.verify_test_encoding(state)
        result[:, :, :2] = (result[:, :, :2] - engine.AFFINE_OFFSET) / engine.AFFINE_SLOPE
    elif transform != "raw":
        raise ValueError(f"Unknown target-channel transform: {transform}")
    return result


def copy_verified(source: Path, destination: Path, engine: Any) -> dict[str, Any]:
    """Copy a frozen control artifact, or verify an existing identical copy."""
    digest = engine.sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if engine.sha256_file(destination) != digest:
            raise RuntimeError(f"Control artifact changed or destination is corrupt: {destination}")
    else:
        temporary = destination.with_suffix(destination.suffix + ".partial")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    return {"source": str(source), "copied_path": str(destination), "sha256": digest}


def prepare(root: Path, output: Path, controls: Path, engine: Any) -> dict[str, Any]:
    """Freeze two new fits while preserving the finished parent controls verbatim."""
    import torch
    parent = engine.verify_plan(root, controls)
    summary_path = controls / "ensemble_summary.json"
    if not summary_path.is_file():
        raise RuntimeError("The parent experiment must finish before preparing this follow-up")
    if parent["seeds"] != list(SEEDS) or parent["train_folds"] != [2, 3, 4] or parent["tune_fold"] != 0 or parent["audit_fold"] != 1:
        raise RuntimeError("Parent seeds or partitions differ from the declared matched comparison")
    output.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {}
    for name in ("site_partition.npz", "climate_regimes.npz"):
        artifacts[name] = copy_verified(controls / name, output / name, engine)
    control_records = {}
    for seed in SEEDS:
        source = controls / f"curve2_seed{seed}"
        record = engine.read_json(source / "training_record.json")
        checkpoint = source / "checkpoint.pt"
        if engine.sha256_file(checkpoint) != record["checkpoint_sha256"]:
            raise RuntimeError(f"Control checkpoint failed its completed-training receipt: {seed}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload["arm"] != "curve2" or payload["seed"] != seed or payload["plan_sha256"] != engine.sha256_file(controls / "preregistered_plan.json"):
            raise RuntimeError("Control checkpoint identity is inconsistent with its frozen parent plan")
        files = {}
        for name in ("checkpoint.pt", "training_record.json", "audit_predictions.npz"):
            files[name] = copy_verified(source / name, output / "controls" / f"curve2_seed{seed}" / name, engine)
        control_records[str(seed)] = files
    configuration_keys = ("width", "depth", "dropout", "learning_rate", "weight_decay", "epochs", "patience", "batch_size")
    plan = {
        "study": STUDY, "arm": ARM, "seeds": list(SEEDS), "new_training_runs": 2,
        "code_sha256": engine.sha256_file(Path(__file__).resolve()), "engine_sha256": ENGINE_SHA256,
        "environment": engine.environment_fingerprint(),
        "parent_plan_sha256": engine.sha256_file(controls / "preregistered_plan.json"),
        "parent_summary_sha256": engine.sha256_file(summary_path),
        "configuration": {key: parent[key] for key in configuration_keys},
        "sources": parent["sources"], "site_hashes": parent["site_hashes"],
        "site_block_fold_sha256": parent["site_block_fold_sha256"], "partition_artifacts": artifacts,
        "target_temporal_audit_sha256": parent["target_temporal_audit_sha256"],
        "controls": control_records,
        "feature_schema": {
            "version": STUDY, "dimension": FEATURE_DIMENSION, "parent_curve2_dimension": 36,
            "auxiliary_variables": list(AUXILIARY_NAMES), "shape_order": "variable-major, then ascending age",
            "shape_dimension": 75, "flag_dimension": 5,
            "shape_formula": "float64 (x-min)/(max-min); no absolute epsilon or physical clamp",
            "degenerate_rule": "range <= 8 * max_age spacing(abs(float32(x)))",
            "spacing_fallback": "Only infinite spacing uses float64(absx)-float64(nextafter(absx,0))",
            "degenerate_output": "15 zero shape entries and a flag of one",
            "finite_policy": "Reject original nonfinite values and nonfinite float32 casts",
            "flag_invariance": "Not universal; affine transformations may change numerical degeneracy",
        },
        "training_policy": "Only folds2,3,4 train; fold0 chooses epoch; control weights are never retrained",
        "audit_policy": "Fold1 already used by the parent study; descriptive comparison only, no arm or parameter selection",
        "ensemble": "Equal mean of the two predeclared seeds; no fitted blend with the control",
        "test_transforms": list(engine.TRANSFORMS),
        "decoder": parent["decoder"], "bootstrap": parent["bootstrap"],
        "climate_regimes": parent["climate_regimes"], "resource_limits": parent["resource_limits"],
        "scientific_hypothesis": "Auxiliary age-profile shape adds predictive information while reducing dependence on unknown positive affine units/offsets",
        "metric": parent["metric"], "submission_policy": "No automatic Kaggle upload or final selection",
    }
    destination = output / "preregistered_plan.json"
    if destination.exists():
        if engine.read_json(destination) != plan:
            raise RuntimeError("Existing study plan differs; use a new output directory instead of overwriting")
    else:
        engine.atomic_json(destination, plan)
        engine.atomic_json(output / "prepared_receipt.json", {"prepared_utc": engine.utc_now(),
                            "plan_sha256": engine.sha256_file(destination), "new_models_trained": 0})
    return plan


def verify_study(root: Path, output: Path, engine: Any) -> dict[str, Any]:
    """Verify the frozen plan, input bytes, partition and copied controls before execution."""
    plan = engine.read_json(output / "preregistered_plan.json")
    if plan["study"] != STUDY or plan["code_sha256"] != engine.sha256_file(Path(__file__).resolve()) or plan["engine_sha256"] != ENGINE_SHA256:
        raise RuntimeError("Study implementation differs from the preregistered version")
    if plan["environment"] != engine.environment_fingerprint():
        raise RuntimeError("Software environment changed after study preparation")
    for name, source in engine.source_paths(root).items():
        if engine.sha256_file(source) != plan["sources"][name]["sha256"]:
            raise RuntimeError(f"Input changed: {name}")
    for name, record in plan["partition_artifacts"].items():
        if engine.sha256_file(output / name) != record["sha256"]:
            raise RuntimeError(f"Partition artifact changed: {name}")
    for seed in SEEDS:
        for name, record in plan["controls"][str(seed)].items():
            if engine.sha256_file(output / "controls" / f"curve2_seed{seed}" / name) != record["sha256"]:
                raise RuntimeError(f"Frozen control artifact changed: seed={seed}, {name}")
    partition = np.load(output / "site_partition.npz")
    if engine.partition_hash(partition["folds"], partition["blocks"]) != plan["site_block_fold_sha256"]:
        raise RuntimeError("Site/block/fold mapping differs from the control experiment")
    return plan


def cached_features(root: Path, output: Path, plan: dict[str, Any], engine: Any) -> np.ndarray:
    """Build the new 116-column cache in its own directory, leaving all older caches intact."""
    path, receipt_path = output / "features.npy", output / "feature_cache_receipt.json"
    plan_hash = engine.sha256_file(output / "preregistered_plan.json")
    if path.exists():
        receipt = engine.read_json(receipt_path)
        if receipt["sha256"] != engine.sha256_file(path) or receipt["plan_sha256"] != plan_hash:
            raise RuntimeError("Auxiliary feature cache failed its provenance check")
        return np.load(path, mmap_mode="r")
    state = np.load(root / "data" / "train_y0.npy", mmap_mode="r")
    temporary = output / "features.partial.npy"
    array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                    shape=(len(state) * 15, FEATURE_DIMENSION))
    counts = {name: 0 for name in AUXILIARY_NAMES}
    fallback = 0
    for start in range(0, len(state), 256):
        stop = min(start + 256, len(state))
        values = state[start:stop]
        array[start * 15:stop * 15] = feature_matrix(values, engine)
        _, diagnostics = auxiliary_shape(values[:, :, 2:])
        for name in AUXILIARY_NAMES:
            counts[name] += diagnostics["degenerate_by_variable"][name]
        fallback += diagnostics["backward_spacing_fallback_values"]
    array.flush()
    del array
    os.replace(temporary, path)
    engine.atomic_json(receipt_path, {"sha256": engine.sha256_file(path), "plan_sha256": plan_hash,
                       "shape": [len(state) * 15, FEATURE_DIMENSION], "degenerate_by_variable": counts,
                       "backward_spacing_fallback_values": fallback})
    return np.load(path, mmap_mode="r")


def load_model(path: Path, engine: Any, device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    """Load a new-model checkpoint for inference without refitting or reading predictions."""
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload["study"] != STUDY or payload["code_sha256"] != engine.sha256_file(Path(__file__).resolve())
            or payload["engine_sha256"] != ENGINE_SHA256 or payload["feature_dimension"] != FEATURE_DIMENSION):
        raise RuntimeError("Checkpoint and inference implementation do not have matching provenance")
    model = engine.make_model(torch, FEATURE_DIMENSION, payload["configuration"]).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, payload


def train_one(root: Path, output: Path, seed: int, plan: dict[str, Any], engine: Any) -> dict[str, Any]:
    """Train one auxiliary-shape model, using only the original train/tune partitions."""
    import torch
    torch.set_num_threads(4)
    directory = output / f"{ARM}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    record_path, checkpoint = directory / "training_record.json", directory / "checkpoint.pt"
    plan_hash = engine.sha256_file(output / "preregistered_plan.json")
    if record_path.exists():
        record = engine.read_json(record_path)
        if record["checkpoint_sha256"] != engine.sha256_file(checkpoint) or record["study_plan_sha256"] != plan_hash:
            raise RuntimeError("Completed auxiliary-shape run has invalid provenance")
        _, payload = load_model(checkpoint, engine)
        if payload["seed"] != seed or payload["study_plan_sha256"] != plan_hash:
            raise RuntimeError("Completed checkpoint belongs to another seed or plan")
        return record
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    sources = engine.load_sources(root)
    partition = np.load(output / "site_partition.npz")
    train_rows, tune_rows = engine.rows_for_sites(partition["train"]), engine.rows_for_sites(partition["tune"])
    cache = cached_features(root, output, plan, engine)
    mean, deviation = engine.feature_scaler(cache, train_rows)
    control = torch.load(output / "controls" / f"curve2_seed{seed}" / "checkpoint.pt", map_location="cpu", weights_only=False)
    if not (np.allclose(mean[:36], control["feature_mean"], atol=1e-6, rtol=1e-6)
            and np.allclose(deviation[:36], control["feature_std"], atol=1e-6, rtol=1e-6)):
        raise RuntimeError("The original curve2 feature scaling does not match the frozen control")
    x_train = torch.from_numpy(engine.normalized_rows(cache, train_rows, mean, deviation))
    x_tune = torch.from_numpy(engine.normalized_rows(cache, tune_rows, mean, deviation))
    del cache
    labels = {}
    anchors = {}
    for name in ("train", "tune"):
        sites = partition[name]
        labels[name] = (np.asarray(sources["train_ydec"][sites, :, 40, :2], dtype=np.float32) / engine.SCALES).reshape(-1, 2)
        anchors[name] = (engine.anchor_at_40(sources["train_y0"][sites])[:, :, :2] / engine.SCALES).reshape(-1, 2)
        if not np.isfinite(labels[name]).all():
            raise ValueError("Train/tune labels contain nonfinite values")
    y_train, a_train = torch.from_numpy(labels["train"]), torch.from_numpy(anchors["train"])
    configuration = plan["configuration"]
    model = engine.make_model(torch, FEATURE_DIMENSION, configuration).to("cuda:0")
    optimizer = torch.optim.AdamW(model.parameters(), lr=configuration["learning_rate"], weight_decay=configuration["weight_decay"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=configuration["epochs"])
    scaler = torch.amp.GradScaler("cuda")
    best, patience, best_epoch = float("inf"), 0, -1
    history = []
    for epoch in range(configuration["epochs"]):
        engine.check_runtime()
        model.train()
        loss_sum, count = 0.0, 0
        for rows in torch.randperm(len(x_train)).split(configuration["batch_size"]):
            engine.check_runtime()
            x, y, anchor = x_train[rows].to("cuda:0"), y_train[rows].to("cuda:0"), a_train[rows].to("cuda:0")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                predicted = model(x).float() + anchor
                loss = (predicted - y).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite auxiliary-shape training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.item()) * len(rows)
            count += len(rows)
        prediction = engine.predict_tensor(model, x_tune, anchors["tune"], torch, "cuda:0")
        score = engine.metric(prediction, labels["tune"])["mse"]
        if not np.isfinite(score):
            raise FloatingPointError("Nonfinite auxiliary-shape tuning score")
        schedule.step()
        history.append({"epoch": epoch, "train_mse": loss_sum / count, "tune_mse": score})
        LOG.info("arm=%s seed=%d epoch=%d train=%.8f tune=%.8f", ARM, seed, epoch, loss_sum / count, score)
        if score < best - 1e-7:
            best, patience, best_epoch = score, 0, epoch
            probe = engine.predict_tensor(model, x_tune[:128], anchors["tune"][:128], torch, "cuda:0")
            engine.save_checkpoint(checkpoint, {
                "study": STUDY, "arm": ARM, "seed": seed, "configuration": configuration,
                "code_sha256": plan["code_sha256"], "engine_sha256": ENGINE_SHA256,
                "study_plan_sha256": plan_hash, "feature_dimension": FEATURE_DIMENSION,
                "feature_schema": plan["feature_schema"], "feature_mean": mean, "feature_std": deviation,
                "site_hashes": plan["site_hashes"], "sources": plan["sources"], "decoder": plan["decoder"],
                "scales": engine.SCALES, "best_epoch": epoch, "best_tune_mse": best,
                "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "reload_probe_predictions": probe, "saved_utc": engine.utc_now(),
            }, torch)
        else:
            patience += 1
            if patience >= configuration["patience"]:
                break
    restored, payload = load_model(checkpoint, engine, "cuda:0")
    reloaded = engine.predict_tensor(restored, x_tune[:128], anchors["tune"][:128], torch, "cuda:0")
    if not np.allclose(reloaded, payload["reload_probe_predictions"], atol=1e-6, rtol=1e-6):
        raise RuntimeError("Auxiliary-shape checkpoint failed the fixed reload tolerance")
    record = {"study": STUDY, "arm": ARM, "seed": seed, "study_plan_sha256": plan_hash,
              "completed_utc": engine.utc_now(), "best_tune_mse": best, "best_epoch": best_epoch,
              "history": history, "checkpoint_sha256": engine.sha256_file(checkpoint), "audit_accessed": False,
              "reload_max_abs": float(np.max(np.abs(reloaded - payload["reload_probe_predictions"]))),
              "reload_atol": 1e-6, "reload_rtol": 1e-6}
    engine.atomic_json(record_path, record)
    del model, restored, optimizer, schedule, scaler, x_train, x_tune, y_train, a_train
    gc.collect()
    torch.cuda.empty_cache()
    return record


def infer_state(model: Any, payload: dict[str, Any], state: np.ndarray, engine: Any,
                device: str, diagnostics: dict[str, Any] | None = None) -> np.ndarray:
    """Infer from initial states only, preserving the shape transformation contract."""
    import torch
    predictions = []
    for start in range(0, len(state), 128):
        if device == "cuda:0":
            engine.check_runtime()
        values = state[start:start + 128]
        x = (feature_matrix(values, engine) - payload["feature_mean"]) / payload["feature_std"]
        if diagnostics is not None:
            diagnostics["clipped"] = diagnostics.get("clipped", 0) + int(np.count_nonzero(np.abs(x) > 8))
            diagnostics["feature_values"] = diagnostics.get("feature_values", 0) + int(x.size)
            diagnostics["clipped_fraction"] = diagnostics["clipped"] / diagnostics["feature_values"]
            _, shape_info = auxiliary_shape(values[:, :, 2:])
            counts = diagnostics.setdefault("degenerate_by_variable", {name: 0 for name in AUXILIARY_NAMES})
            for name in AUXILIARY_NAMES:
                counts[name] += shape_info["degenerate_by_variable"][name]
        x = np.clip(x, -8, 8)
        anchor = (engine.anchor_at_40(values)[:, :, :2] / engine.SCALES).reshape(-1, 2)
        predictions.append(engine.predict_tensor(model, torch.from_numpy(x), anchor, torch, device))
    return np.concatenate(predictions)


def test_predictions(root: Path, checkpoint: Path, transform: str, engine: Any,
                     device: str = "cpu") -> tuple[np.ndarray, dict[str, Any]]:
    """Reproduce all test rows from a new checkpoint without cached predictions."""
    model, payload = load_model(checkpoint, engine, device)
    outputs, diagnostics = [], {}
    for scenario in ("ssp126", "ssp585"):
        state = np.load(root / "data" / f"{scenario}_test_y0.npy", mmap_mode="r")
        if state.shape != (6171, 15, 7):
            raise ValueError(f"Unexpected test state shape: {scenario}")
        decoded = decode_initial_state(state, transform, engine)
        scenario_info: dict[str, Any] = {}
        outputs.append(infer_state(model, payload, decoded, engine, device, scenario_info))
        diagnostics[scenario] = scenario_info
    return np.concatenate(outputs), diagnostics


def finalize(root: Path, output: Path, plan: dict[str, Any], engine: Any, device: str) -> dict[str, Any]:
    """Describe the reused audit fold without selecting a model, weight or transform."""
    final_path = output / "descriptive_comparison.json"
    if final_path.exists():
        return engine.read_json(final_path)
    for seed in SEEDS:
        directory = output / f"{ARM}_seed{seed}"
        record = engine.read_json(directory / "training_record.json")
        if engine.sha256_file(directory / "checkpoint.pt") != record["checkpoint_sha256"]:
            raise RuntimeError("Both new models must complete with valid checkpoints before audit access")
    receipt_path = output / "audit_access_receipt.json"
    if not receipt_path.exists():
        engine.atomic_json(receipt_path, {"first_access_utc": engine.utc_now(),
                            "interpretation": plan["audit_policy"], "study_plan_sha256": engine.sha256_file(output / "preregistered_plan.json")})
    sources = engine.load_sources(root)
    partition = np.load(output / "site_partition.npz")
    sites = partition["audit"]
    labels = (np.asarray(sources["train_ydec"][sites, :, 40, :2], dtype=np.float32) / engine.SCALES).reshape(-1, 2)
    state = sources["train_y0"][sites]
    regimes = np.load(output / "climate_regimes.npz")["regime"][sites]
    if not np.isfinite(labels).all():
        raise ValueError("Nonfinite audit labels")
    new_predictions, controls = [], []
    test_by_transform: dict[str, list[np.ndarray]] = {name: [] for name in engine.TRANSFORMS}
    report: dict[str, Any] = {"study": STUDY, "audit_interpretation": plan["audit_policy"], "seeds": {},
                             "selection_performed": False, "kaggle_submission_performed": False}
    for seed in SEEDS:
        directory = output / f"{ARM}_seed{seed}"
        checkpoint = directory / "checkpoint.pt"
        model, payload = load_model(checkpoint, engine, device)
        predicted = infer_state(model, payload, state, engine, device)
        saved_control = np.load(output / "controls" / f"curve2_seed{seed}" / "audit_predictions.npz")
        if not np.array_equal(saved_control["site_ids"], sites):
            raise RuntimeError("Control and treatment audit site order differs")
        control = saved_control["clean"]
        if control.shape != predicted.shape or not np.isfinite(control).all():
            raise ValueError("Invalid frozen control predictions")
        np.savez_compressed(directory / "audit_predictions.npz", prediction=predicted, site_ids=sites)
        result = {"new_metric": engine.metric(predicted, labels), "control_metric": engine.metric(control, labels),
                  "stratified_new": engine.stratified_metrics(predicted, labels, regimes), "test_candidates": {}}
        new_predictions.append(predicted)
        controls.append(control)
        for transform in engine.TRANSFORMS:
            test, diagnostics = test_predictions(root, checkpoint, transform, engine, device)
            np.save(directory / f"test_predictions_{transform}.npy", test)
            result["test_candidates"][transform] = engine.write_submission(root, directory / f"submission_{transform}.csv", test)
            result["test_candidates"][transform]["input_diagnostics"] = diagnostics
            test_by_transform[transform].append(test)
        report["seeds"][str(seed)] = result
        engine.atomic_json(directory / "descriptive_record.json", result)
    averaged = np.mean(np.stack(new_predictions), axis=0, dtype=np.float64).astype(np.float32)
    control_mean = np.mean(np.stack(controls), axis=0, dtype=np.float64).astype(np.float32)
    np.save(output / "ensemble_audit.npy", averaged)
    report["equal_seed_ensemble"] = {"weights": [0.5, 0.5], "new_metric": engine.metric(averaged, labels),
                                       "frozen_control_metric": engine.metric(control_mean, labels),
                                       "stratified_new": engine.stratified_metrics(averaged, labels, regimes),
                                       "test_candidates": {}}
    bootstrap = engine.paired_bootstrap(averaged, control_mean, labels, partition["blocks"][sites], plan)
    bootstrap["comparison"] = "curve2_aux_shape minus frozen curve2; negative favors auxiliary shape"
    bootstrap["interpretation"] = "Descriptive paired comparison on a reused audit fold; no selection or independent-generalization claim"
    report["paired_block_bootstrap"] = bootstrap
    for transform in engine.TRANSFORMS:
        test = np.mean(np.stack(test_by_transform[transform]), axis=0, dtype=np.float64).astype(np.float32)
        np.save(output / f"ensemble_{transform}.npy", test)
        report["equal_seed_ensemble"]["test_candidates"][transform] = engine.write_submission(root, output / f"ensemble_{transform}.csv", test)
    report["completed_utc"] = engine.utc_now()
    engine.atomic_json(final_path, report)
    return report


def smoke(engine: Any) -> dict[str, Any]:
    """Test invariance, numerical boundaries, rejection and CPU checkpoint reload."""
    import torch
    torch.set_num_threads(2)
    rng = np.random.default_rng(112)
    values = rng.normal(size=(6, 15, 5)).astype(np.float64)
    first, _ = auxiliary_shape(values)
    slope = np.array([0.35, 1.7, 4.0, 0.875, 10.0])
    offset = np.array([20.0, -3.0, 0.0, 100.0, -55.0])
    second, _ = auxiliary_shape(values * slope + offset)
    active = (first[:, 75:] == 0) & (second[:, 75:] == 0)
    shapes1, shapes2 = first[:, :75].reshape(-1, 5, 15), second[:, :75].reshape(-1, 5, 15)
    invariance_error = float(np.max(np.abs(shapes1[active] - shapes2[active])))
    assert active.all() and invariance_error <= 2e-6
    assert np.all((first[:, :75] >= 0) & (first[:, :75] <= 1))
    for value in (0.0, -3.0, 1e20, float(np.finfo(np.float32).max)):
        constant, info = auxiliary_shape(np.full((2, 15, 5), value, dtype=np.float64))
        assert np.array_equal(constant[:, :75], np.zeros((2, 75)))
        assert np.array_equal(constant[:, 75:], np.ones((2, 5)))
        if value == float(np.finfo(np.float32).max):
            assert info["backward_spacing_fallback_values"] == 150
    ulp = float(np.spacing(np.float32(1.0)))
    boundary = np.broadcast_to((1.0 + np.linspace(0, 8 * ulp, 15))[:, None], (15, 5))[None].copy()
    flag8, _ = auxiliary_shape(boundary)
    assert np.all(flag8[:, 75:] == 1)
    boundary9 = np.broadcast_to((1.0 + np.linspace(0, 9 * ulp, 15))[:, None], (15, 5))[None].copy()
    flag9, _ = auxiliary_shape(boundary9)
    assert np.all(flag9[:, 75:] == 0)
    tiny = float(np.nextafter(np.float32(0), np.float32(1)))
    subnormal = np.broadcast_to((np.arange(15) * tiny)[:, None], (15, 5))[None].copy()
    subnormal_shape, _ = auxiliary_shape(subnormal)
    assert np.all(subnormal_shape[:, 75:] == 0)
    assert np.allclose(subnormal_shape[:, :75].reshape(1, 5, 15), np.arange(15) / 14, atol=1e-7)
    simple = np.broadcast_to(np.linspace(0, 1, 15)[:, None], (15, 5))[None].copy()
    flag_before, _ = auxiliary_shape(simple)
    flag_after, _ = auxiliary_shape(simple + 1e8)
    expected_flag_changes = int(np.count_nonzero(flag_before[:, 75:] != flag_after[:, 75:]))
    assert expected_flag_changes == 5
    rejected = 0
    for invalid in (np.nan, np.inf, -np.inf, float(np.finfo(np.float32).max) * 2):
        test = values.copy()
        test[0, 0, 0] = invalid
        try:
            auxiliary_shape(test)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("An invalid auxiliary value was accepted")
    try:
        auxiliary_shape(np.zeros((2, 15, 4)))
    except ValueError:
        rejected += 1
    else:
        raise AssertionError("Wrong auxiliary shape was accepted")
    ordered_values = np.empty((1, 15, 5), dtype=np.float64)
    for index in range(5):
        ordered_values[0, :, index] = np.linspace(0, 1, 15) ** (index + 1)
    ordered, _ = auxiliary_shape(ordered_values)
    assert np.allclose(ordered[0, :75].reshape(5, 15), ordered_values[0].T, atol=1e-7)
    state = rng.uniform(0, 25, size=(3, 15, 7)).astype(np.float32)
    full = feature_matrix(state, engine)
    assert full.shape == (45, 116)
    assert np.array_equal(full[:, :36], engine.features(None, state, "curve2"))
    # Exercise the real decoder with valid encoded zero offsets and auxiliary
    # profiles outside the parent's cleaning range. Those five channels must
    # retain their exact float32 bits, including values below zero and above 1e6.
    decoder_state = state.copy()
    decoder_state[0, 0, :2] = 0
    decoder_state[:, :, 2:] = (np.linspace(-2e6, 2e6, 15, dtype=np.float32)[None, :, None]
                               * np.array([1, .5, 2, .25, 4], dtype=np.float32))
    encoded_state = decoder_state.copy()
    encoded_state[:, :, :2] = encoded_state[:, :, :2] * engine.AFFINE_SLOPE + engine.AFFINE_OFFSET
    encoded_snapshot = encoded_state.copy()
    decoded_state = decode_initial_state(encoded_state, "community_affine", engine)
    raw_state = decode_initial_state(encoded_state, "raw", engine)
    assert np.array_equal(encoded_state.view(np.uint32), encoded_snapshot.view(np.uint32))
    assert np.array_equal(raw_state.view(np.uint32), encoded_state.view(np.uint32))
    assert np.array_equal(decoded_state[:, :, 2:].view(np.uint32),
                          decoder_state[:, :, 2:].view(np.uint32))
    assert np.allclose(decoded_state[:, :, :2], decoder_state[:, :, :2], atol=1e-6, rtol=1e-6)
    shapes_before_decoder, _ = auxiliary_shape(decoder_state[:, :, 2:])
    shapes_after_decoder, _ = auxiliary_shape(decoded_state[:, :, 2:])
    assert np.array_equal(shapes_before_decoder, shapes_after_decoder)
    decoder_roundtrip_error = float(np.max(np.abs(decoded_state[:, :, :2] - decoder_state[:, :, :2])))
    configuration = {"width": 16, "depth": 2, "dropout": 0.2}
    model = engine.make_model(torch, 116, configuration).eval()
    payload = {"study": STUDY, "configuration": configuration,
               "code_sha256": engine.sha256_file(Path(__file__).resolve()), "engine_sha256": ENGINE_SHA256,
               "feature_dimension": 116, "feature_mean": np.zeros(116, dtype=np.float32),
               "feature_std": np.ones(116, dtype=np.float32), "model_state": model.state_dict()}
    before = infer_state(model, payload, state, engine, "cpu")
    with tempfile.TemporaryDirectory(prefix="aux-shape-smoke-") as directory:
        checkpoint = Path(directory) / "checkpoint.pt"
        engine.save_checkpoint(checkpoint, payload, torch)
        restored, restored_payload = load_model(checkpoint, engine)
        after = infer_state(restored, restored_payload, state, engine, "cpu")
    assert np.array_equal(before, after)
    return {"passed": True, "device": "cpu", "gpu_used": False, "plan_prepared": False,
            "active_profile_affine_invariance_max_abs": invariance_error,
            "expected_resolution_induced_flag_changes": expected_flag_changes,
            "target_decoder_roundtrip_max_abs": decoder_roundtrip_error,
            "auxiliary_decoder_bit_exact": True,
            "invalid_cases_rejected": rejected,
            "checks": ["variable-major order", "range [0,1]", "constant profiles", "8ULP boundary",
                       "9ULP active profile", "zero/subnormal spacing", "float32 maximum spacing fallback",
                       "positive-affine active-profile invariance", "documented flag non-invariance",
                       "nonfinite and overflow rejection", "exact parent 36-feature prefix",
                       "target decoder roundtrip", "auxiliary decoder bit-exact preservation",
                       "decoder input immutability", "CPU checkpoint exact reload"]}


def main() -> None:
    """Run only the explicitly selected preparation, experiment, or inference action."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "prepare", "run", "finalize", "predict"))
    parser.add_argument("--base-script", type=Path, default=Path(__file__).with_name("transfer_experiment_20261113.py"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--checkpoints", type=Path, nargs="+")
    parser.add_argument("--test-transform", choices=("raw", "community_affine"), default="community_affine")
    parser.add_argument("--prediction-path", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    engine = load_engine(args.base_script.resolve())
    import torch
    torch.set_num_threads(4)
    if args.action == "smoke":
        print(json.dumps(smoke(engine), indent=2))
        return
    if args.root is None:
        parser.error("--root is required except for smoke")
    root = args.root.resolve()
    output = (args.output or root / "research" / STUDY / "run").resolve()
    if args.action == "prepare":
        if args.controls is None:
            parser.error("prepare requires --controls pointing to the completed parent experiment")
        print(json.dumps(prepare(root, output, args.controls.resolve(), engine), indent=2))
        return
    if args.action == "predict":
        if not args.checkpoints:
            parser.error("predict requires --checkpoints")
        values = [test_predictions(root, path, args.test_transform, engine, "cpu")[0] for path in args.checkpoints]
        prediction = np.mean(np.stack(values), axis=0, dtype=np.float64).astype(np.float32)
        destination = args.prediction_path or output / f"reproduced_{args.test_transform}.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps(engine.write_submission(root, destination, prediction), indent=2))
        return
    plan = verify_study(root, output, engine)
    if args.action == "finalize":
        print(json.dumps(finalize(root, output, plan, engine, "cpu"), indent=2))
        return
    signal.signal(signal.SIGTERM, engine.request_stop)
    signal.signal(signal.SIGINT, engine.request_stop)
    with engine.exclusive_gpu(root):
        for seed in SEEDS:
            train_one(root, output, seed, plan, engine)
        if not engine.STOP:
            print(json.dumps(finalize(root, output, plan, engine, "cuda:0"), indent=2))


if __name__ == "__main__":
    main()
