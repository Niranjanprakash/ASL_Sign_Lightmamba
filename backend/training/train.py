import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from backend.config import (
    CLASSES, BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    PROCESSED_DIR, OUTPUT_DIR, RANDOM_SEED, USE_HORIZONTAL_FLIP,
    EARLY_STOPPING_PATIENCE, CHECKPOINT_DIR
)
from backend.utils import set_seed, get_device, logger
from backend.data.dataset import ASLVideoDataset
from backend.models.lightmamba_asl import LightMambaASL
from backend.training.losses import get_loss_fn
from backend.training.metrics import calculate_metrics
from backend.training.validate import validate_epoch
from backend.training.early_stopping import EarlyStopping
from backend.training.scheduler import get_scheduler

def save_plots(history: dict):
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # 1. Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / "training_curve_loss.png")
    plt.close()
    
    # 2. Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Validation Acc")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / "training_curve_accuracy.png")
    plt.close()

def main():
    set_seed(RANDOM_SEED)
    device = get_device()
    
    # 1. Paths verification
    train_csv = PROCESSED_DIR / "splits" / "train.csv"
    val_csv = PROCESSED_DIR / "splits" / "val.csv"
    
    if not train_csv.exists() or not val_csv.exists():
        logger.error("Dataset splits csv files not found. Please run: python -m backend.data.prepare_dataset first.")
        return
        
    # 2. Read dataset and check class imbalance
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    class_counts = [len(train_df[train_df["label_id"] == i]) for i in range(len(CLASSES))]
    logger.info(f"Class sample distribution: {list(zip(CLASSES, class_counts))}")
    
    # 3. Create Datasets & Dataloaders
    train_dataset = ASLVideoDataset(str(train_csv), is_training=True, use_horizontal_flip=USE_HORIZONTAL_FLIP)
    val_dataset = ASLVideoDataset(str(val_csv), is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. Instantiate Model
    model = LightMambaASL(pretrained=True, freeze_backbone=True).to(device)

    # 5. Define Loss, Optimizer, Scheduler, Early Stopping
    loss_fn = get_loss_fn(class_counts, use_weights=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)  # stronger regularization
    scheduler = get_scheduler(optimizer, scheduler_type="ReduceLROnPlateau")
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True)

    # Training History tracker
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": []
    }

    # Resume from last checkpoint if exists
    start_epoch = 1
    STAGE2_EPOCH = 20
    stage2_active = False
    resume_path = CHECKPOINT_DIR / "last_model.pth"
    if resume_path.exists():
        print(f"[RESUME] Found checkpoint — resuming from {resume_path.name}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        early_stopping.best_loss = ckpt["validation_loss"]
        # Load history if saved
        history_path = OUTPUT_DIR / "metrics" / "training_history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
        if start_epoch > STAGE2_EPOCH:
            model.rgb_branch.unfreeze_final_blocks()
            stage2_active = True
        print(f"[RESUME] Resuming from epoch {start_epoch}")
    else:
        print("[TRAIN] No checkpoint found — starting fresh.")

    logger.info("Starting training loop...")
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        epoch_start = time.time()
        
        # Transition to Stage 2: Unfreeze MobileNet backbone and fine-tune with a lower LR
        if epoch == STAGE2_EPOCH and not stage2_active:
            logger.info("="*60)
            logger.info("[TWO-STAGE TRAINING] Unfreezing MobileNetV3 final layers for fine-tuning!")
            logger.info("="*60)
            model.rgb_branch.unfreeze_final_blocks()
            # Lower learning rate for fine-tuning
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE * 0.1
            stage2_active = True
            
        model.train()
        train_loss = 0.0
        all_logits = []
        all_targets = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} [Train]", leave=False):
            # Move tensors to device
            rgb = batch["rgb"].to(device)
            landmarks = batch["landmarks"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["label"].to(device)
            
            optimizer.zero_grad()
            logits = model(rgb, landmarks, mask)
            loss = loss_fn(logits, targets)
            loss.backward()
            
            # Clip gradients to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * targets.size(0)
            all_logits.append(logits.detach())
            all_targets.append(targets)
            
        # Compute training metrics
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        epoch_train_loss = train_loss / len(train_dataset)
        train_metrics = calculate_metrics(all_logits, all_targets)
        
        # Validation epoch
        epoch_val_loss, val_metrics = validate_epoch(model, val_loader, loss_fn, device)
        
        # Update scheduler
        current_lr = optimizer.param_groups[0]['lr']
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_val_loss)
        else:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        
        # Log results
        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {train_metrics['accuracy'] * 100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_metrics['accuracy'] * 100:.2f}% | "
            f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
        )
        
        # Save history
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(val_metrics["accuracy"])
        history["lr"].append(current_lr)
        
        # Early Stopping check
        early_stopping(epoch_val_loss, model, optimizer, epoch, val_metrics, CLASSES)
        if early_stopping.early_stop:
            logger.info(f"[EARLY STOPPING] Training stopped early at epoch {epoch}")
            break

    # Save training curves and JSON logs
    save_plots(history)
    metrics_dir = OUTPUT_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)
        
    logger.info("Training process complete!")

if __name__ == "__main__":
    main()
