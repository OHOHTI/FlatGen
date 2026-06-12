# Global Variables needed to run the full pipeline
# Note: if the code crashes, it is likely due to either running out of RAM or VRAM.
#       VRAM can be solved by lowering batch size
#       RAM can be solved by using a smaller dataset or by manually limiting the training set size in step 4)

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
FLATGEN_DIR = BASE_DIR.parent
CRYSTAL_NET_DIR = FLATGEN_DIR / "crystal_net"

DATASET = "MaterialsProject"

# These are the dimensions of the supercell of your two sets of of graphs/supercells
# Note: the supercell sizes can be set to the same or different values but must larger than or equal to 3
SUPERCELL_SIZE_1 = 3
SUPERCELL_SIZE_2 = 3

API_KEY = ""            # Paste your Materials Project API key here
INPUT_FILE_MP_LIST = "/home/yihao/vaebo/Struct2Flat/data/mp_complete_summary.csv"  # Optional: text file containing a list of Materials Project IDs to limit the dataset

# GNN encoder parameters:
# These should be adjusted based on VRAM limits and computation time limits

# With 8GB of VRAM, recommended values are 48 for the 2D datasets and 16 for 3D datasets
BATCH_SIZE = 16      # Number of graphs/materials per mini-batch
# Good results can be obtained with 150 epochs. Smaller numbers will provide acceptable results (>50)
NUM_EPOCHS = 150      # Number of training epochs

# Select which parts of the pipeline you like to run with either 'yes' or 'no'
GRAPH_GENERATION = 'no' # Select yes if the graphs have not been generated; no if you would like to skip straight to training
TRAINING = 'yes'

MODEL_NAME = f"{DATASET}_V1_sublattice" # The name used for trained models, embeddings and other outputs

# Default input locations expected by the clustering workflow.
EMBEDDINGS_PATH = BASE_DIR / f"Embeddings/embeddings_{MODEL_NAME}_{SUPERCELL_SIZE_1}x{SUPERCELL_SIZE_1}.pt"
LABELS_PATH = BASE_DIR / f"Labels/labels_{MODEL_NAME}_{SUPERCELL_SIZE_1}x{SUPERCELL_SIZE_1}.pt"
EMBEDDING_UMAP_PATH = BASE_DIR / f"Embeddings/embedding_umap_{MODEL_NAME}_{SUPERCELL_SIZE_1}x{SUPERCELL_SIZE_1}.npy"
MATERIALS_DIR = BASE_DIR / "Materials" / DATASET
CLUSTER_OUTPUT_DIR = BASE_DIR / f"Clusters_fb/{MODEL_NAME}"
CLUSTER_PLOT_DIR = BASE_DIR / f"Clustering Plots_fb/{MODEL_NAME}"
LATENT_OUTPUT_DIR = BASE_DIR / "latent_interpretation"
OUTLIERS_PATH = BASE_DIR / "outliers.txt"

# Inputs produced by the crystal_net stage.
FB_LIST = CRYSTAL_NET_DIR / "TB_search_flat_bands_lat_ids.xlsx"
DUP_LIST = CRYSTAL_NET_DIR / "crystal_net_results" / "duplicate_id_species_rcsr_mismatch.csv"
