#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

#SBATCH --job-name=crystal_net
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

source ~/anaconda3/etc/profile.d/conda.sh
conda activate crynet

#python main.py
python collect_and_output.py 
python get_graph_invariants.py
python output_graph_invariants.py 
