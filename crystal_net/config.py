import os
import numpy as np
from pathlib import Path

class ParamObj:
    """Stores all user-controllable parameters."""
    verbose = False
    save_results = True
    run_parallel = True
    generated = False
    MP_KEY = ""  # Keep empty in the published copy.

    max_dist = 1.2
    decay_rate = None
    dist_units = "NN"
    max_fb_width = 0.01
    flatness_tol = 0.9
    full_flat = False
    check_to_N_hops = 5
    high_symm_only = False

    reduce_structure = False
    print_structure_data = False
    plot_structure_3D = False
    simple_graphics = True
    plot_TB = False
    just_save_plots = True
    plot_TBDOS = False
    print_NN_dists = False
    print_TB_params = True
    check_TB_flat = True
    plot_BS = False
    plot_DOS = False
    plot_DOS_and_BS = False
    plot_BZ = False
    save_cgd = True

    script_path = Path(os.path.dirname(os.path.realpath(__file__)))

    def __init__(self, mat_list=None):
        if self.generated:
            from pymatgen.core import Structure
            cif_files = sorted((self.script_path / "Materials_gen").glob("*.cif"))
            
            ids = []
            names = []
            nsites = []
            
            for cif_file in cif_files:
                try:
                    s = Structure.from_file(cif_file)
                    ids.append(cif_file.stem)
                    names.append(s.composition.reduced_formula)
                    nsites.append(len(s))
                except Exception as e:
                    print(f"Could not read {cif_file}: {e}")
            
            self.material_IDs = np.array(ids)
            self.names_list = np.array(names)
            self.nsites_list = np.array(nsites)

            if mat_list is not None:
                if isinstance(mat_list[0], (int, np.int64)):
                    self.material_nums = mat_list
                elif isinstance(mat_list[0], str):
                    self.material_nums = [int(np.where(self.material_IDs == i)[0][0]) for i in mat_list]
                else:
                    self.material_nums = range(len(self.material_IDs))
            else:
                self.material_nums = range(len(self.material_IDs))
        else:
            from io_utils import load_all_MP_names 
            try:
                self.material_IDs, self.names_list, self.nsites_list = load_all_MP_names(self.script_path)
                if mat_list is not None:
                    if isinstance(mat_list[0], (int, np.int64)):
                        self.material_nums = mat_list
                    elif isinstance(mat_list[0], str):
                        self.material_nums = [int(np.where(self.material_IDs == i)[0][0]) for i in mat_list]
                    else:
                        self.material_nums = range(len(self.material_IDs))
                else:
                    self.material_nums = range(len(self.material_IDs))
            except Exception as e:
                print(f"Warning: Materials info not loaded: {e}")

    def as_dict(self):
        r"""Converts class to a dictionary so it is json serializeable."""
        if isinstance(self.material_nums, np.ndarray):
            mat_nums_ser = self.material_nums.tolist()
        elif isinstance(self.material_nums, range):
            mat_nums_ser = list(self.material_nums)
        else:
            mat_nums_ser = self.material_nums
        this_dict = {
            "verbose": self.verbose,
            "save_results": self.save_results,
            "run_parallel": self.run_parallel,
            "MP_KEY": self.MP_KEY,
            "max_dist": self.max_dist,
            "decay_rate": self.decay_rate,
            "dist_units": self.dist_units,
            "max_fb_width": self.max_fb_width,
            "flatness_tol": self.flatness_tol,
            "full_flat": self.full_flat,
            "check_to_N_hops": self.check_to_N_hops,
            "high_symm_only": self.high_symm_only,
            "reduce_structure": self.reduce_structure,
            "print_structure_data": self.print_structure_data,
            "plot_structure_3D": self.plot_structure_3D,
            "simple_graphics": self.simple_graphics,
            "plot_TB": self.plot_TB,
            "just_save_plots": self.just_save_plots,
            "plot_TBDOS": self.plot_TBDOS,
            "print_NN_dists": self.print_NN_dists,
            "print_TB_params": self.print_TB_params,
            "check_TB_flat": self.check_TB_flat,
            "plot_BS": self.plot_BS,
            "plot_DOS": self.plot_DOS,
            "plot_DOS_and_BS": self.plot_DOS_and_BS,
            "plot_BZ": self.plot_BZ,
            "save_cgd": self.save_cgd,
            "script_path": str(self.script_path),
            "material_nums": mat_nums_ser,
            "material_IDs": self.material_IDs.tolist(),
            "names_list": self.names_list.tolist(),
            "nsites_list": self.nsites_list.tolist()
        }
        return this_dict
