import torch
from torch.utils.data import Dataset
import torch.nn as nn
import pickle
import os
import numpy as np
import random
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from configs import get_cfg_defaults
from model import Main_model

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def collate_fn(batch):
    # batch is list of tuples (g, lg, key, text_string, flatness_score)
    graphs_g, graphs_lg, keys, texts, flatness_scores = zip(*batch)
    flatness_scores = torch.tensor(flatness_scores, dtype=torch.float32).unsqueeze(-1)
    
    batch_g = Batch.from_data_list(list(graphs_g))
    batch_lg = Batch.from_data_list(list(graphs_lg))
    
    return (
        batch_g,
        batch_lg,
        list(keys),  
        list(texts),  
        flatness_scores,
    )

class PyGDataset(Dataset):
    def __init__(self, data_dict):
        self.data = list(data_dict.items())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        key, value = self.data[index]
        g, lg = value["structure_graph"]
        flatness_score = value["flatness_score"]
        text_string = value["text_string"]
        
        return g, lg, key, text_string, flatness_score

def predict(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_true_values = []
    all_keys = []

    with torch.no_grad():
        for batch in data_loader:
            graph1, graph2, keys, text_input, flatness_score = batch
            graph1 = graph1.to(device)
            graph2 = graph2.to(device)
            flatness_score = flatness_score.to(device)

            
            predicted_score = model(graph1, graph2, text_input, device, mode="infer")

            all_predictions.extend(predicted_score.cpu().numpy().reshape(-1))
            all_true_values.extend(flatness_score.cpu().numpy().reshape(-1))
            all_keys.extend(keys)  

    return all_keys, all_predictions, all_true_values

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

    # Update output paths to point to PyG locations if needed
    cfg.merge_from_list(["DIR.OUTPUT_DIR", cfg["DIR"]["OUTPUT_DIR"] + "_pyg"])
    if not os.path.exists(cfg["DIR"]["OUTPUT_DIR"]):
        os.makedirs(cfg["DIR"]["OUTPUT_DIR"])

    best_model_path = cfg["DIR"]["OUTPUT_DIR"] + '/best_model.pth'
    prediction_output_file = cfg["DIR"]["OUTPUT_DIR"] + '/predictions.txt'
    
    pickle_file_path = cfg["DIR"]["picklefile"].replace(".pkl", "_pyg.pkl")
    if not os.path.exists(pickle_file_path):
        print(f"Warning: Expected PyG pickle file at {pickle_file_path} not found. Using default.")
        pickle_file_path = cfg["DIR"]["picklefile"]

    print(f"Loading data from {pickle_file_path}")
    with open(pickle_file_path, 'rb') as f:
        dataset_dic = pickle.load(f)

    scores_dict = load_existing_scores(cfg["DIR"]["bandfile"])
    dataset_full = preprocess_band_structure(dataset_dic, scores_dict)

    flatness_scores = [v["flatness_score"] for v in dataset_full.values()]
    print(f"Flatness score mean: {np.mean(flatness_scores)}, std: {np.std(flatness_scores)}")

    val_data = dataset_full

    val_dataset = PyGDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=cfg["MODEL"]["BATCH_SIZE"], shuffle=False, collate_fn=collate_fn)

    model = Main_model(**cfg).to(device)
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from {best_model_path}")
    else:
        raise FileNotFoundError(f"Best model not found at {best_model_path}. Please run train_pyg.py first.")

    keys, predictions, true_values = predict(model, val_loader, device)

    with open(prediction_output_file, 'w') as f:
        f.write("Key\tPredicted\tTrue\n")
        for key, pred, true in zip(keys, predictions, true_values):
            f.write(f"{key}\t{pred:.6f}\t{true:.6f}\n")

    print(f"Predictions saved to {prediction_output_file}")

if __name__ == '__main__':
    main()
