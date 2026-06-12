import torch
from time import time
from torch.utils.data import Dataset
import torch.nn as nn
import pickle
import os
import numpy as np
import random
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import mean_squared_error, r2_score
from configs import get_cfg_defaults
from model import Main_model

torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def collate_fn(batch):
    # batch is a list of tuples: (g, lg, text_string, flatness_score)
    graphs_g, graphs_lg, texts, flatness_score = zip(*batch)
    
    flatness_score = torch.tensor(flatness_score, dtype=torch.float32).unsqueeze(-1)
    
    # Batch the crystal graphs and line graphs separately
    batch_g = Batch.from_data_list(list(graphs_g))
    batch_lg = Batch.from_data_list(list(graphs_lg))
    
    return (
        batch_g,
        batch_lg,
        texts,
        flatness_score,
    )

class PyGDataset(Dataset):
    def __init__(self, data_dict, mask_ratio=0.1, mask_mode="random", train=True):
        self.data = list(data_dict.items())
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        key, value = self.data[index]
        # value["structure_graph"] is expected to be a tuple (g, lg) from GraphGen_pyg
        g, lg = value["structure_graph"]
        flatness_score = value["flatness_score"]
        text_string = value["text_string"]
        return g, lg, text_string, flatness_score

def train_and_evaluate(model, optimizer, train_loader, val_loader, device, mode="train"):
    model.train() if mode == "train" else model.eval()
    total_loss = 0
    all_band_preds, all_band_true = [], []

    criterion = nn.MSELoss(reduction='mean')

    loader = train_loader if mode == "train" else val_loader
    
    for batch_idx, batch in enumerate(loader):
        graph1, graph2, text_input, flatness_score = batch
        graph1 = graph1.to(device)
        graph2 = graph2.to(device)
        flatness_score = flatness_score.to(device)

        optimizer.zero_grad() if mode == "train" else None

        with torch.no_grad() if mode != "train" else torch.enable_grad():
            predicted_score = model(graph1, graph2, text_input, device, mode=mode)
            loss = criterion(predicted_score.squeeze(), flatness_score)

        if mode == "train":
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) 
            optimizer.step()

        total_loss += loss.item()
        all_band_preds.extend(predicted_score.cpu().detach().numpy().reshape(-1))
        all_band_true.extend(flatness_score.cpu().detach().numpy().reshape(-1))

    avg_loss = total_loss / len(loader)
    mse = mean_squared_error(all_band_true, all_band_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_band_true, all_band_preds)

    return avg_loss, mse, rmse, r2

def preprocess_band_structure(dataset_dic, flatness_dic):
    common_keys = set(dataset_dic.keys()) & set(flatness_dic.keys())
    dataset_full = {key: dataset_dic[key].copy() for key in common_keys}
    for key in common_keys:
        score = flatness_dic[key]
        dataset_full[key]['flatness_score'] = score
    return dataset_full

def load_existing_scores(scores_file):
    scores_dict = {}
    if os.path.exists(scores_file):
        import pandas as pd
        df = pd.read_csv(scores_file, sep='\t')
        for _, row in df.iterrows():
            scores_dict[row['Key']] = float(row['S_total'])
    return scores_dict

def main():
    cfg = get_cfg_defaults()
    print(f"Learning Rate: {cfg['MODEL']['LR']}") 

    # Update output paths to avoid overwriting existing
    cfg.merge_from_list(["DIR.OUTPUT_DIR", cfg["DIR"]["OUTPUT_DIR"] + "_pyg"])
    if not os.path.exists(cfg["DIR"]["OUTPUT_DIR"]):
        os.makedirs(cfg["DIR"]["OUTPUT_DIR"])
        
    tracking_file = cfg["DIR"]["OUTPUT_DIR"] + '/tracking_printer.txt'
    with open(tracking_file, 'w') as f:
        f.write("Epoch, Train Loss, Val Loss, Train MSE, Train RMSE, Train R2, Val MSE, Val RMSE, Val R2\n")

    pickle_file_path = cfg["DIR"]["picklefile"].replace(".pkl", "_pyg.pkl")
    if not os.path.exists(pickle_file_path):
        fallback_pickle = cfg["DIR"]["picklefile"]
        if os.path.exists(fallback_pickle):
            pickle_file_path = fallback_pickle
        else:
            raise FileNotFoundError(
                f"PyG pickle not found at {pickle_file_path}. "
                "Run scripts/preprocess_pyg.py first."
            )
    
    print(f"Loading data from {pickle_file_path}")
    with open(pickle_file_path, 'rb') as f:
        dataset_dic = pickle.load(f)

    scores_dict = load_existing_scores(cfg["DIR"]["bandfile"])
    dataset_full = preprocess_band_structure(dataset_dic, scores_dict)

    # Check data distribution
    flatness_scores = [v["flatness_score"] for v in dataset_full.values()]
    print(f"Flatness score mean: {np.mean(flatness_scores)}, std: {np.std(flatness_scores)}")

    best_model_path = cfg["DIR"]["OUTPUT_DIR"] + '/best_model.pth'
    keys = list(dataset_full.keys())
    print(f"Total number of samples: {len(keys)}")
    random.shuffle(keys)
    split_idx = int(0.8 * len(keys))
    train_data = {k: dataset_full[k] for k in keys[:split_idx]}
    val_data = {k: dataset_full[k] for k in keys[split_idx:]}

    model = Main_model(**cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.MODEL.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_dataset = PyGDataset(train_data, train=True)
    val_dataset = PyGDataset(val_data, train=False)
    
    # Use PyG DataLoader
    train_loader = DataLoader(train_dataset, batch_size=cfg["MODEL"]["BATCH_SIZE"], shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["MODEL"]["BATCH_SIZE"], shuffle=False, collate_fn=collate_fn, drop_last=True)

    best_val_r2 = -float("inf")

    for epoch in range(cfg["MODEL"]["NUM_EPOCHS"]):
        start_time = time()

        train_loss, train_mse, train_rmse, train_r2 = train_and_evaluate(
            model, optimizer, train_loader, val_loader, device, mode="train"
        )

        val_loss, val_mse, val_rmse, val_r2 = train_and_evaluate(
            model, optimizer, train_loader, val_loader, device, mode="infer"
        )

        with open(tracking_file, 'a') as f:
            f.write(f"{epoch + 1}, {train_loss:.4f}, {val_loss:.4f}, {train_mse:.4f}, {train_rmse:.4f}, {train_r2:.4f}, {val_mse:.4f}, {val_rmse:.4f}, {val_r2:.4f}\n")

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch + 1}.")

        scheduler.step(val_loss)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']}")
        print(f"Epoch {epoch + 1} finished in {time() - start_time:.2f} seconds.")

if __name__ == '__main__':
    s = time()
    main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
