import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
import sys
import numpy as np
import argparse
from torch_geometric.data import Batch
from configs import get_cfg_defaults
from model import Main_model
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from GraphGen_pyg import Graph, c_lattice_enlarge
from jarvis.core.atoms import pmg_to_atoms

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def ase_to_pymatgen(atoms):
    """Convert ASE Atoms to pymatgen Structure."""
    from pymatgen.core.structure import Structure
    from pymatgen.core.lattice import Lattice

    lattice = Lattice(atoms.cell[:])
    species = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    structure = Structure(lattice, species, positions, coords_are_cartesian=True)
    return structure


def build_text_string(structure):
    """Build a text string from a pymatgen Structure, matching the format used in training."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    lattice = structure.lattice
    lattice_params = {
        'a': lattice.a,
        'b': lattice.b,
        'c': lattice.c,
        'alpha': lattice.alpha,
        'beta': lattice.beta,
        'gamma': lattice.gamma,
    }

    try:
        sga = SpacegroupAnalyzer(structure)
        sg_symbol = sga.get_space_group_symbol()
        point_group = sga.get_point_group_symbol()
        crystal_system = sga.get_crystal_system()
    except Exception:
        sg_symbol = "unknown"
        point_group = "unknown"
        crystal_system = "unknown"

    formula_pretty = structure.composition.reduced_formula
    formula_anonymous = structure.composition.anonymized_formula
    elements = [str(el) for el in structure.composition.elements]

    info_str = (
        f"formula_anonymous: {formula_anonymous}, "
        f"formula_pretty: {formula_pretty}, "
        f"sg_symbol: {sg_symbol}, "
        f"point_group: {point_group}, "
        f"crystal_system: {crystal_system}, "
        f"lattice_mat: {lattice.matrix.tolist()}, "
        f"lattice_params: {lattice_params}, "
        f"species: {elements}, "
        f"lattice_volume: {structure.volume}"
    )
    return info_str


def atoms_to_graph(atoms):
    """Convert ASE Atoms to PyG graph (g, lg) for the model."""
    structure = ase_to_pymatgen(atoms)
    jarvis_atoms = pmg_to_atoms(structure)
    g, lg = Graph.atom_pyg_multigraph(
        atoms=jarvis_atoms,
        neighbor_strategy="k-nearest",
        cutoff=8.0,
        max_neighbors=6,
        atom_features="cgcnn",
        max_attempts=3,
        cutoff_extra=3.5,
        compute_line_graph=True,
        dtype="float32",
    )
    text_string = build_text_string(structure)
    return g, lg, text_string


def collate_fn(batch):
    graphs_g, graphs_lg, keys, texts = zip(*batch)
    batch_g = Batch.from_data_list(list(graphs_g))
    batch_lg = Batch.from_data_list(list(graphs_lg))
    return (
        batch_g,
        batch_lg,
        list(keys),
        list(texts),
    )


class AtomsDataset(Dataset):
    def __init__(self, atoms_dict):
        """
        Args:
            atoms_dict: dict mapping key -> {"atoms": ase.Atoms} or
                        key -> {"g": g, "lg": lg, "text_string": str}
        """
        self.data = list(atoms_dict.items())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        key, value = self.data[index]
        g = value["g"]
        lg = value["lg"]
        text_string = value["text_string"]
        return g, lg, key, text_string


def predict(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_keys = []

    with torch.no_grad():
        for batch in data_loader:
            graph1, graph2, keys, text_input = batch
            graph1 = graph1.to(device)
            graph2 = graph2.to(device)
            predicted_score = model(graph1, graph2, text_input, device, mode="infer")
            all_predictions.extend(predicted_score.cpu().numpy().reshape(-1))
            all_keys.extend(keys)

    return all_keys, all_predictions


def predict_with_mc_dropout(model, data_loader, device, n_samples=30):
    model.train()  # enable dropout
    all_keys = []
    all_predictions = []

    for batch in data_loader:
        graph1, graph2, keys, text_input = batch
        graph1 = graph1.to(device)
        graph2 = graph2.to(device)

        mc_predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = model(graph1, graph2, text_input, device, mode="infer")
                mc_predictions.append(pred.cpu().numpy().reshape(-1))

        mc_predictions = np.stack(mc_predictions, axis=0)
        all_predictions.append(mc_predictions)
        all_keys.extend(keys)

    all_predictions = np.concatenate(all_predictions, axis=1)
    return all_keys, all_predictions


def plot_flatness_distribution(predictions, output_dir):
    plt.figure(figsize=(10, 6))
    plt.hist(predictions, bins=50, density=True, alpha=0.7)
    plt.xlabel('Predicted Flatness Score')
    plt.ylabel('Density')
    plt.title('Distribution of Predicted Flatness Scores')
    plt.grid(True, alpha=0.3)
    output_path = os.path.join(output_dir, 'flatness_distribution.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Flatness distribution plot saved to {output_path}")


def plot_overall_uncertainty(keys, mc_predictions, output_dir):
    mean_preds = np.mean(mc_predictions, axis=0)
    std_preds = np.std(mc_predictions, axis=0)

    sort_indices = np.argsort(mean_preds)
    sorted_means = mean_preds[sort_indices]
    sorted_stds = std_preds[sort_indices]

    plt.figure(figsize=(9, 8))
    plt.plot(sorted_means, 'b-', label='Mean Prediction')
    plt.fill_between(range(len(sorted_means)),
                     sorted_means - 1.96 * sorted_stds,
                     sorted_means + 1.96 * sorted_stds,
                     alpha=0.3, color='blue',
                     label='95% Confidence Interval')
    plt.xlabel('Samples (sorted by prediction)', fontsize=18)
    plt.ylabel('Predicted Flatness Score', fontsize=18)
    plt.title('Overall Prediction Uncertainty', fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=18)
    output_path = os.path.join(output_dir, 'overall_uncertainty_sorted.png')
    plt.savefig(output_path, dpi=800)
    plt.show()
    plt.close()
    print(f"Overall uncertainty plot saved to {output_path}")

    plt.figure(figsize=(10, 6))
    plt.hist(std_preds, bins=50, density=True, alpha=0.7, color='purple')
    plt.xlabel('Prediction Uncertainty (Standard Deviation)')
    plt.ylabel('Density')
    plt.title('Distribution of Prediction Uncertainty')
    plt.grid(True, alpha=0.3)
    output_path = os.path.join(output_dir, 'uncertainty_distribution.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Uncertainty distribution plot saved to {output_path}")


def load_cif_files(cif_dir):
    """Load all CIF files from a directory as ASE Atoms."""
    from ase.io import read

    atoms_dict = {}
    cif_files = sorted([f for f in os.listdir(cif_dir) if f.endswith('.cif')])
    for cif_file in cif_files:
        key = os.path.splitext(cif_file)[0]
        try:
            atoms = read(os.path.join(cif_dir, cif_file))
            atoms_dict[key] = atoms
        except Exception as e:
            print(f"Error reading {cif_file}: {e}")
    return atoms_dict


def prepare_data(atoms_dict):
    """Convert a dict of {key: ase.Atoms} to model-ready format."""
    data = {}
    for key, atoms in atoms_dict.items():
        try:
            g, lg, text_string = atoms_to_graph(atoms)
            data[key] = {
                "g": g,
                "lg": lg,
                "text_string": text_string,
            }
        except Exception as e:
            print(f"Error processing {key}: {e}")
    return data


def predict_from_atoms(atoms_list, model_path=None, batch_size=8, mc_dropout=True, n_samples=30):
    """
    Predict flatness scores from a list of ASE Atoms.

    Args:
        atoms_list: list of ASE Atoms, or dict of {key: ASE Atoms}
        model_path: path to the trained model checkpoint
        batch_size: batch size for prediction
        mc_dropout: whether to use MC dropout for uncertainty estimation
        n_samples: number of MC dropout samples

    Returns:
        dict of {key: {"prediction": float, "uncertainty": float (if mc_dropout)}}
    """
    cfg = get_cfg_defaults()

    if model_path is None:
        model_path = cfg["DIR"]["SAVEMODEL"] + '/best_model.pth'

    if isinstance(atoms_list, dict):
        atoms_dict = atoms_list
    else:
        atoms_dict = {f"structure_{i}": a for i, a in enumerate(atoms_list)}

    print(f"Processing {len(atoms_dict)} structures...")
    data = prepare_data(atoms_dict)
    print(f"Successfully processed {len(data)} structures")

    dataset = AtomsDataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=False, collate_fn=collate_fn)

    model = Main_model(**cfg).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")

    keys, predictions = predict(model, data_loader, device)

    results = {}
    for key, pred in zip(keys, predictions):
        results[key] = {"prediction": float(pred)}

    if mc_dropout:
        mc_keys, mc_predictions = predict_with_mc_dropout(
            model, data_loader, device, n_samples=n_samples)
        mean_preds = np.mean(mc_predictions, axis=0)
        std_preds = np.std(mc_predictions, axis=0)
        for key, mean, std in zip(mc_keys, mean_preds, std_preds):
            results[key]["mc_mean"] = float(mean)
            results[key]["uncertainty"] = float(std)

    return results


def main():
    parser = argparse.ArgumentParser(description="Predict flatness scores from CIF files or ASE Atoms")
    parser.add_argument("--cif_dir", type=str, default=None,
                        help="Directory containing CIF files")
    parser.add_argument("--cif_file", type=str, default=None,
                        help="Single CIF file to predict")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for predictions and plots")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for prediction")
    parser.add_argument("--mc_dropout", action="store_true", default=True,
                        help="Use MC dropout for uncertainty estimation")
    parser.add_argument("--no_mc_dropout", action="store_true",
                        help="Disable MC dropout")
    parser.add_argument("--n_samples", type=int, default=30,
                        help="Number of MC dropout samples")
    args = parser.parse_args()

    from ase.io import read

    cfg = get_cfg_defaults()

    if args.model_path is None:
        args.model_path = cfg["DIR"]["SAVEMODEL"] + '/best_model.pth'
    if args.output_dir is None:
        args.output_dir = cfg["DIR"]["SAVEMODEL"] + '/cif_predictions'

    os.makedirs(args.output_dir, exist_ok=True)

    mc_dropout = args.mc_dropout and not args.no_mc_dropout

    # Load structures
    atoms_dict = {}
    if args.cif_file is not None:
        key = os.path.splitext(os.path.basename(args.cif_file))[0]
        atoms_dict[key] = read(args.cif_file)
    elif args.cif_dir is not None:
        atoms_dict = load_cif_files(args.cif_dir)
    else:
        print("Please provide --cif_dir or --cif_file")
        return

    print(f"Loaded {len(atoms_dict)} structures")

    # Prepare graph data
    data = prepare_data(atoms_dict)
    print(f"Successfully built graphs for {len(data)} structures")

    dataset = AtomsDataset(data)
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn)

    # Load model
    model = Main_model(**cfg).to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded best model from {args.model_path}")
    else:
        raise FileNotFoundError(f"Best model not found at {args.model_path}")

    # Predict
    keys, predictions = predict(model, data_loader, device)

    prediction_output_file = os.path.join(args.output_dir, 'predictions.txt')
    with open(prediction_output_file, 'w') as f:
        f.write("Key\tPredicted Flatness Score\n")
        for key, pred in zip(keys, predictions):
            f.write(f"{key}\t{pred:.6f}\n")

    plot_flatness_distribution(predictions, args.output_dir)

    if mc_dropout:
        mc_keys, mc_predictions = predict_with_mc_dropout(
            model, data_loader, device, n_samples=args.n_samples)
        plot_overall_uncertainty(mc_keys, mc_predictions, args.output_dir)

        mc_output_file = os.path.join(args.output_dir, 'predictions_with_uncertainty.txt')
        mean_preds = np.mean(mc_predictions, axis=0)
        std_preds = np.std(mc_predictions, axis=0)
        with open(mc_output_file, 'w') as f:
            f.write("Key\tPredicted Flatness Score\tMC Mean\tUncertainty (Std)\n")
            for key, pred, mean, std in zip(keys, predictions, mean_preds, std_preds):
                f.write(f"{key}\t{pred:.6f}\t{mean:.6f}\t{std:.6f}\n")
        print(f"Predictions with uncertainty saved to {mc_output_file}")

    print(f"Predictions saved to {prediction_output_file}")


if __name__ == '__main__':
    main()
