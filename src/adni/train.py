"""Training loop for multimodal transformer (PIPELINE_v2 §6)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from adni.models import MultimodalTransformer


def set_seed(seed: int) -> None:
    """Set all seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    features_keys: list[str],
    modality_dims: dict[str, int],
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in loader:
        # Unpack batch
        y = batch[0].to(device)
        features = {}
        for i, key in enumerate(features_keys):
            features[key] = batch[1 + i].to(device)
        
        optimizer.zero_grad()
        logits, _, _ = model(features, training=True)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def evaluate(
    model: nn.Module,
    X: dict[str, np.ndarray],
    y: np.ndarray,
    device: torch.device,
    features_keys: list[str],
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model. Returns (logits, cls_tokens, effective_masks)."""
    model.eval()
    n = len(y)
    
    all_logits = []
    all_cls = []
    all_masks = []
    
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_features = {k: torch.from_numpy(X[k][start:end]).float().to(device) for k in features_keys}
            y_batch = torch.from_numpy(y[start:end]).to(device)
            
            logits, cls_token, eff_mask = model(batch_features, training=False)
            
            all_logits.append(logits.cpu().numpy())
            all_cls.append(cls_token.cpu().numpy())
            all_masks.append(eff_mask.cpu().numpy())
    
    logits = np.concatenate(all_logits, axis=0)
    cls_tokens = np.concatenate(all_cls, axis=0)
    effective_masks = np.concatenate(all_masks, axis=0)
    
    return logits, cls_tokens, effective_masks


def train_model(
    model: MultimodalTransformer,
    train_data: dict[str, np.ndarray],
    train_y: np.ndarray,
    val_data: dict[str, np.ndarray],
    val_y: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    features_keys: list[str],
    save_path: Path | None = None,
) -> dict[str, Any]:
    """Train model with early stopping on val macro-F1.
    
    Returns dict with training history and best model info.
    """
    from adni.metrics import macro_f1
    
    set_seed(config["seed"])
    
    # Create dataloader
    tensors = [torch.from_numpy(train_y).long()]
    for k in features_keys:
        tensors.append(torch.from_numpy(train_data[k]).float())
    
    dataset = TensorDataset(*tensors)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=False)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    
    # Training loop
    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    patience_counter = 0
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
        "val_macro_auroc": [],
    }
    
    n_epochs = config["max_epochs"]
    patience = config["early_stopping_patience"]
    
    for epoch in range(n_epochs):
        t0 = time.time()
        
        # Train
        model.train()
        train_loss = train_epoch(model, loader, optimizer, device, features_keys, {})
        
        # Validate
        val_logits, val_cls, val_masks = evaluate(
            model, val_data, val_y, device, features_keys
        )
        val_loss = F.cross_entropy(
            torch.from_numpy(val_logits), torch.from_numpy(val_y)
        ).item()
        
        # Compute macro-F1
        val_preds = val_logits.argmax(axis=1)
        val_f1 = macro_f1(val_y, val_preds)
        
        # Compute macro-AUROC
        from sklearn.metrics import roc_auc_score
        val_auroc = roc_auc_score(val_y, val_logits, multi_class="ovr", average="macro")
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_f1)
        history["val_macro_auroc"].append(val_auroc)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if save_path and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_F1={val_f1:.4f}, val_AUROC={val_auroc:.4f}, "
                  f"time={time.time()-t0:.1f}s")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}, best epoch {best_epoch+1}")
            break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    result = {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "history": history,
        "n_epochs_run": len(history["train_loss"]),
    }
    
    if save_path:
        torch.save(model.state_dict(), save_path)
        result["save_path"] = str(save_path)
    
    return result
