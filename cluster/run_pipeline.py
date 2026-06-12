"""
Master script to run the GNN training pipeline in sequence:
  1) Sublattice Generation
  2) Supercell Generation
  3) Graph Generation
  4) Model Training / Embedding Generation

Based on the config file settings, certain steps will be skipped.
"""

from config import (
    DATASET,
    API_KEY,
    SUPERCELL_SIZE_1,
    SUPERCELL_SIZE_2,
    BATCH_SIZE,
    NUM_EPOCHS,
    GRAPH_GENERATION,
    TRAINING,
)

from preprocess import generate_sublattices, generate_supercells, generate_graphs
from training import main as run_training


def main():
    print("===== Running GNN Training Pipeline =====")
    print(f"DATASET: {DATASET}")
    print(f"API_KEY: {'(provided)' if API_KEY else '(missing)'}")
    print(f"SUPERCELL_SIZE_1: {SUPERCELL_SIZE_1}")
    print(f"SUPERCELL_SIZE_2: {SUPERCELL_SIZE_2}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"NUM_EPOCHS: {NUM_EPOCHS}")
    print()

    if GRAPH_GENERATION.lower() != 'no':
        print("\n---- Step 1: Sublattice Generation ----")
        generate_sublattices()

        print("\n---- Step 2: Supercell Generation ----")
        generate_supercells()

        print("\n---- Step 3: Graph Generation ----")
        generate_graphs()
    else:
        print("Skipping Steps 1-3: Graph generation disabled by config.")

    if TRAINING.lower() != 'no':
        print("\n---- Step 4: Model Training ----")
        run_training()
    else:
        print("Skipping Step 4: Model Training disabled by config.")

    print("\n===== GNN Training Pipeline Completed Successfully! =====")


if __name__ == "__main__":
    main()
