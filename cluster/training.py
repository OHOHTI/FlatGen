# Model training, loss functions, and embedding generation.
#
# Loads the perturbed graph lists produced by preprocess.py, trains the
# GNN encoder defined in gnn.py using InfoNCE (or Barlow Twins), then
# generates final embeddings from the unperturbed graph lists.
# Can be run standalone or called from run_pipeline.py.

# =========================================
# 1) IMPORTS & GLOBAL SETTINGS
# =========================================

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.data import DataLoader

torch.backends.cuda.matmul.allow_tf32 = True
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)

from config import (
    DATASET,
    SUPERCELL_SIZE_1,
    SUPERCELL_SIZE_2,
    MODEL_NAME,
    BATCH_SIZE,
    NUM_EPOCHS,
)

from data_utils import PairedGraphDataset
from gnn import GNNEncoder

# Hyperparameters
INPUT_FOLDER = f"Graphs/{MODEL_NAME}"
LEARNING_RATE = 1e-4
EMBEDDING_DIMENSION = 192
HIDDEN_DIMENSION = 256
LAYER_NUMBER = 6
DROPOUT_RATE = 0.1
WEIGHT_DECAY = 1e-5
TEMPERATURE = 0.1
LAMBDA = 5e-3


# =========================================
# 2) LOSS FUNCTIONS
# =========================================

def info_nce_loss(z1, z2, temperature=0.07):
    """InfoNCE (NT-Xent) contrastive loss over two embedding batches."""
    z1 = F.normalize(z1.float(), dim=1)
    z2 = F.normalize(z2.float(), dim=1)
    N = z1.size(0)

    z = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.matmul(z, z.t()) / temperature

    pos_indices = torch.arange(N, 2 * N, device=z.device)
    neg_indices = torch.arange(0, N, device=z.device)
    labels = torch.cat([pos_indices, neg_indices], dim=0)

    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

    return F.cross_entropy(sim_matrix, labels)


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Flattened view of the off-diagonal elements of a square matrix."""
    n, m = x.shape
    assert n == m
    return x.flatten()[1:].view(n - 1, n + 1)[:, :-1].flatten()


def barlow_twins_loss(z1, z2, lambd=LAMBDA):
    """Barlow Twins redundancy-reduction loss."""
    z1_norm = (z1 - z1.mean(0)) / z1.std(0)
    z2_norm = (z2 - z2.mean(0)) / z2.std(0)

    N, D = z1_norm.shape
    c = (z1_norm.T @ z2_norm) / N

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = _off_diagonal(c).pow_(2).sum()
    return on_diag + lambd * off_diag


# =========================================
# 3) TRAINING HELPERS
# =========================================

def _compute_loss(model, data1, data2, device, scaler):
    data1, data2 = data1.to(device), data2.to(device)
    with torch.amp.autocast(enabled=scaler.is_enabled(), device_type='cuda'):
        e1 = model(data1)
        e2 = model(data2)
        return info_nce_loss(e1, e2, temperature=TEMPERATURE)


def _train_one_epoch(model, loader, optimizer, device, scaler):
    model.train()
    total = 0.0
    count = 0
    for d1, d2 in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = _compute_loss(model, d1, d2, device, scaler)
        if not torch.isfinite(loss):
            continue
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        total += loss.item()
        count += 1
    return total / max(count, 1)


def _validate(model, loader, device, scaler):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for d1, d2 in loader:
            total += _compute_loss(model, d1, d2, device, scaler).item()
    return total / len(loader)


def train_model(model, train_loader, val_loader,
                optimizer, num_epochs, scheduler,
                checkpoint_path, device, scaler):
    """Full training loop with early-best checkpointing."""
    tr_losses, va_losses = [], []
    best_val = float('inf')
    try:
        for ep in range(num_epochs):
            tr = _train_one_epoch(model, train_loader, optimizer, device, scaler)
            va = _validate(model, val_loader, device, scaler)
            scheduler.step()
            tr_losses.append(tr)
            va_losses.append(va)

            if va < best_val:
                best_val = va
                torch.save(model.state_dict(), checkpoint_path)
                print(f"✓ epoch {ep+1:3d}: new best val={best_val:.4f}")

            print(f"epoch {ep+1:3d}/{num_epochs} | train {tr:.4f} | val {va:.4f}")
    except KeyboardInterrupt:
        print("Training interrupted — best model saved.")
    return tr_losses, va_losses


def plot_losses(tr, va, save_path):
    """Save a training/validation loss curve."""
    if not tr:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(tr) + 1), tr, label='Training', marker='o')
    plt.plot(range(1, len(va) + 1), va, label='Validation', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs. Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


# =========================================
# 4) EMBEDDING GENERATION
# =========================================

def generate_embeddings(model, loader, device, scaler):
    """Run the encoder in eval mode and return (embeddings, labels)."""
    model.eval()
    embeddings_list = []
    labels_list = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            with torch.amp.autocast(enabled=scaler.is_enabled(), device_type='cuda'):
                emb = model(data)
            embeddings_list.append(emb.cpu())
            labels_list.extend(data.label)
    return torch.cat(embeddings_list, dim=0), labels_list


# =========================================
# 5) MAIN SCRIPT
# =========================================

def main():
    fixed_seed = 44
    np.random.seed(fixed_seed)
    torch.manual_seed(fixed_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    os.makedirs("Models", exist_ok=True)
    os.makedirs("training_plots", exist_ok=True)
    os.makedirs("training_splits", exist_ok=True)

    # ── Load perturbed graph lists ──────────────────────────────────────
    print("Loading perturbed graph lists for training/validation...")
    graph_list_set_1 = torch.load(
        f"{INPUT_FOLDER}/graph_list_set_1.pt", weights_only=False
    )
    graph_list_set_2 = torch.load(
        f"{INPUT_FOLDER}/graph_list_set_2.pt", weights_only=False
    )

    dataset_length = len(graph_list_set_1)
    train_count = int(dataset_length * 0.8)
    val_count = dataset_length - train_count

    # ── Train / val split ───────────────────────────────────────────────
    split_path = os.path.join("training_splits", f"train_val_indices_{MODEL_NAME}.pt")

    if os.path.exists(split_path):
        train_indices, val_indices = torch.load(split_path, weights_only=False)
        print(f"Loaded existing train/val split from {split_path}")
    else:
        indices = np.random.permutation(dataset_length)
        train_indices = indices[:train_count]
        val_indices = indices[train_count:]
        torch.save((train_indices, val_indices), split_path)
        print(f"Created new train/val split and saved to {split_path}")

    train_set_1 = [graph_list_set_1[i] for i in train_indices]
    train_set_2 = [graph_list_set_2[i] for i in train_indices]
    val_set_1 = [graph_list_set_1[i] for i in val_indices]
    val_set_2 = [graph_list_set_2[i] for i in val_indices]

    train_loader = DataLoader(
        PairedGraphDataset(train_set_1, train_set_2),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        PairedGraphDataset(val_set_1, val_set_2),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    print(f"Training set: {train_count}, Validation set: {val_count}")

    # ── Build model ─────────────────────────────────────────────────────
    print(f"""
-----------initialising---------------
BATCH_SIZE:           {BATCH_SIZE}
NUM_EPOCHS:           {NUM_EPOCHS}
Learning Rate:        {LEARNING_RATE}
Hidden Dimension:     {HIDDEN_DIMENSION}
Embedding Dimension:  {EMBEDDING_DIMENSION}
Temperature (InfoNCE):{TEMPERATURE}
""")

    model = GNNEncoder(
        num_node_features=graph_list_set_1[0].x.size(1),
        num_edge_features=graph_list_set_1[0].edge_attr.size(1),
        hidden_dim=HIDDEN_DIMENSION,
        embedding_dim=EMBEDDING_DIMENSION,
        num_layers=LAYER_NUMBER,
        dropout_rate=DROPOUT_RATE,
    ).to(device)

    checkpoint_path = f"Models/{MODEL_NAME}.pth"

    # ── Train or skip ───────────────────────────────────────────────────
    if NUM_EPOCHS == 0:
        print("NUM_EPOCHS == 0 → skipping training.")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"No checkpoint at {checkpoint_path}. "
                "Set NUM_EPOCHS > 0 to train a fresh model."
            )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        train_losses, val_losses = [], []
    else:
        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        warmup = LinearLR(optimizer, start_factor=1e-3,
                          total_iters=min(NUM_EPOCHS, 5))
        if NUM_EPOCHS > 5:
            cosine = CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS - 5, eta_min=1e-6
            )
            scheduler = SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[5]
            )
        else:
            scheduler = warmup

        if os.path.exists(checkpoint_path):
            model.load_state_dict(
                torch.load(checkpoint_path, map_location=device)
            )
            print(f"Resumed from checkpoint: {checkpoint_path}")

        train_losses, val_losses = train_model(
            model, train_loader, val_loader,
            optimizer, NUM_EPOCHS, scheduler,
            checkpoint_path, device, scaler,
        )

    plot_losses(train_losses, val_losses,
                f"training_plots/{MODEL_NAME}.png")

    # ── Free training data ──────────────────────────────────────────────
    del graph_list_set_1, graph_list_set_2
    del train_set_1, train_set_2, val_set_1, val_set_2

    # ── Reload best model for embedding generation ──────────────────────
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=False)
    )
    print("Reloaded the best model from disk.")

    os.makedirs("Embeddings", exist_ok=True)
    os.makedirs("Labels", exist_ok=True)

    # ── Embeddings from unperturbed set 1 ───────────────────────────────
    print("Loading unperturbed graph list 1 for final embeddings...")
    graphs_1 = torch.load(
        f"{INPUT_FOLDER}/graph_list_unperturbed_1.pt", weights_only=False
    )
    loader_1 = DataLoader(graphs_1, batch_size=BATCH_SIZE, shuffle=False)
    emb_1, lab_1 = generate_embeddings(model, loader_1, device, scaler)
    print(f"Embeddings shape (set 1): {emb_1.shape}")

    emb_path_1 = f"Embeddings/embeddings_{MODEL_NAME}_{SUPERCELL_SIZE_1}x{SUPERCELL_SIZE_1}.pt"
    lab_path_1 = f"Labels/labels_{MODEL_NAME}_{SUPERCELL_SIZE_1}x{SUPERCELL_SIZE_1}.pt"
    torch.save(emb_1, emb_path_1)
    torch.save(lab_1, lab_path_1)
    print(f"Saved: {emb_path_1}, {lab_path_1}")
    del graphs_1, emb_1, lab_1
    torch.cuda.empty_cache()

    # ── Embeddings from unperturbed set 2 (if it exists) ────────────────
    unpert_path_2 = f"{INPUT_FOLDER}/graph_list_unperturbed_2.pt"
    if os.path.exists(unpert_path_2):
        print("Loading unperturbed graph list 2 for final embeddings...")
        graphs_2 = torch.load(unpert_path_2, weights_only=False)
        loader_2 = DataLoader(graphs_2, batch_size=BATCH_SIZE, shuffle=False)
        emb_2, lab_2 = generate_embeddings(model, loader_2, device, scaler)

        emb_path_2 = f"Embeddings/embeddings_{MODEL_NAME}_{SUPERCELL_SIZE_2}x{SUPERCELL_SIZE_2}.pt"
        lab_path_2 = f"Labels/labels_{MODEL_NAME}_{SUPERCELL_SIZE_2}x{SUPERCELL_SIZE_2}.pt"
        torch.save(emb_2, emb_path_2)
        torch.save(lab_2, lab_path_2)
        print(f"Saved: {emb_path_2}, {lab_path_2}")
        del graphs_2, emb_2, lab_2
    else:
        print("No second unperturbed list (equal supercell sizes) — skipping set 2.")

    torch.cuda.empty_cache()
    print("=== Training and Embedding Generation Completed ===")


if __name__ == "__main__":
    main()
