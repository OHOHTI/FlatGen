# skelegen configuration — external model paths for screening.
# All paths are resolved relative to the FlatGen repository layout:
#   <FlatGen root>/SkeleGen/skelegen/config_skelegen.py  (this file)
#   <FlatGen root>/SkeleGen/GNN_EVAL/     (GNN stability classifier weights)
#   <FlatGen root>/SkeleGen/Flat_screen/  (flatness surrogate inference code)
#   <FlatGen root>/flat_screen/results/   (flatness surrogate weights, produced
#                                          by flat_screen/model/train_pyg.py)
import os

home_dir = os.path.dirname(os.path.abspath(__file__))          # .../skelegen
_repo_root = os.path.dirname(home_dir)                         # .../SkeleGen
_flatgen_root = os.path.dirname(_repo_root)                    # FlatGen root

gnn_eval_path = os.path.join(_repo_root, 'GNN_EVAL')
flat_screen_path = os.path.join(_repo_root, 'Flat_screen')
flat_model_path = os.path.join(_flatgen_root, 'flat_screen', 'results', 'flatness_pyg', 'best_model.pth')
stab_pred_name_A = "stab_240409-113155"
stab_pred_name_B = "stab_240402-111754"
mag_pred_name = "mag_240815-085301"
