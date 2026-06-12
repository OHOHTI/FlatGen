import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import colorcet as cc
import pickle
import copy
from pymatgen.core import IStructure
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.core.composition import Composition
from io_utils import TBoutputs
from pathlib import Path
from config import ParamObj
# plt.style.use("paul_style.mplstyle")
#from auto_TB_MP_v2 import *

plt.close(fig="all")


class mp_saved_info:
    def __init__(self, file_name="MP_info.txt"):
        r"""load general info about MP materials"""
        self.mp_ids = np.genfromtxt(
            file_name, delimiter=", ", skip_header=1, usecols=0, dtype=str
        )
        self.formulas = np.genfromtxt(
            file_name, delimiter=", ", skip_header=1, usecols=1, dtype=str
        )
        self.nmatsites = np.genfromtxt(
            file_name, delimiter=", ", skip_header=1, usecols=2, dtype=int
        )
        self.nelements = np.genfromtxt(
            file_name, delimiter=", ", skip_header=1, usecols=3, dtype=int
        )
        self.spacegroup_symbol = np.genfromtxt(
            file_name, delimiter=", ", skip_header=1, usecols=4, dtype=str
        )
        self.spacegroup_number = np.genfromtxt(
            file_name, delimiter=", ", skip_header=1, usecols=5, dtype=int
        )
        self.energy_per_atom = np.genfromtxt(
            file_name, delimiter=", ", skip_header=1, usecols=6, dtype=float
        )
        self.band_gap = np.genfromtxt(
            file_name, delimiter=", ", skip_header=1, usecols=7, dtype=float
        )
        self.has_bs = np.genfromtxt(
            file_name, delimiter=", ", skip_header=1, usecols=8, dtype=int
        )
        self.icsd_ids = np.genfromtxt(
            file_name, delimiter=", ", skip_header=1, usecols=9, dtype=int
        )


class gathered_results:
    def __init__(self, suffix, mp_info=None):
        r"""returns and pickles an object with lots of info about a data run"""
        # load info about FBs from a given calc run
        tbout = TBoutputs(f"crystal_net_results/TB_{suffix}/{suffix}_TB_search_results.txt")
        self.mp_id = tbout.mp_id
        self.chem_name = tbout.chem_name
        self.specie_names = tbout.specie_names
        self.nFBs = tbout.nFBs
        self.nsites = tbout.nsites
        self.NN_dists = tbout.NN_dists
        self.NN_dists_max = tbout.NN_dists_max
        self.NNN_dists = tbout.NNN_dists
        self.nFB_lats = tbout.nFB_lats

        # load systre results
        file_name = f"crystal_net_results/{suffix}_systre_keys.txt"
        systre_ID = np.genfromtxt(
            file_name, delimiter=", ", skip_header=4, usecols=0, dtype=str
        )
        systre_FB_el = np.genfromtxt(
            file_name, delimiter=", ", skip_header=4, usecols=1, dtype=str
        )
        systre_FB_dim = np.genfromtxt(
            file_name, delimiter=", ", skip_header=4, usecols=3, dtype=int
        )
        systre_key = np.genfromtxt(
            file_name, delimiter=", ", skip_header=4, usecols=4, dtype=str
        )
        systre_sg = np.genfromtxt(
            file_name, delimiter=", ", skip_header=4, usecols=5, dtype=str
        )
        systre_rcsr = np.genfromtxt(
            file_name, delimiter=", ", skip_header=4, usecols=6, dtype=str
        )
        systre_vec = np.genfromtxt(
            file_name, delimiter=",", skip_header=4, usecols=7, dtype=str
        )

        # preallocate results variables
        self.nmatsites = np.zeros(self.nFB_lats)
        self.nelements = np.zeros(self.nFB_lats)
        self.spacegroup_symbol = [""] * self.nFB_lats
        self.spacegroup_number = np.zeros(self.nFB_lats)
        self.energy_per_atom = np.zeros(self.nFB_lats)
        self.band_gap = np.zeros(self.nFB_lats)
        self.has_bs = np.zeros(self.nFB_lats)
        self.icsd_ids = np.zeros(self.nFB_lats)
        self.FB_dims = [[]] * self.nFB_lats
        self.systre_keys = [[]] * self.nFB_lats
        self.systre_sg = [[]] * self.nFB_lats
        self.systre_rcsr = [[]] * self.nFB_lats
        self.systre_vec = [[]] * self.nFB_lats
        ind_list = []

        # look at each flat band lattice
        for FB_ind in range(self.nFB_lats):
            if mp_info is not None:
                this_ind = np.nonzero(self.mp_id[FB_ind] == mp_info.mp_ids)[0]
                
                # find the corresponding mp information
                if len(this_ind) > 0:
                    self.nmatsites[FB_ind] = mp_info.nmatsites[this_ind[0]]
                    self.nelements[FB_ind] = mp_info.nelements[this_ind[0]]
                    self.spacegroup_symbol[FB_ind] = mp_info.spacegroup_symbol[this_ind[0]]
                    self.spacegroup_number[FB_ind] = mp_info.spacegroup_number[this_ind[0]]
                    self.energy_per_atom[FB_ind] = mp_info.energy_per_atom[this_ind[0]]
                    self.band_gap[FB_ind] = mp_info.band_gap[this_ind[0]]
                    self.has_bs[FB_ind] = mp_info.has_bs[this_ind[0]]
                    self.icsd_ids[FB_ind] = mp_info.icsd_ids[this_ind[0]]

            # find the corresponding systre keys
            these_inds = np.nonzero(
                np.logical_and(
                    self.mp_id[FB_ind] == systre_ID,
                    self.specie_names[FB_ind] == systre_FB_el,
                )
            )
            for this_ind in these_inds[0].tolist():
                self.FB_dims[FB_ind] = self.FB_dims[FB_ind] + [
                    int(systre_FB_dim[this_ind])
                ]
                self.systre_keys[FB_ind] = self.systre_keys[FB_ind] + [
                    systre_key[this_ind]
                ]
                self.systre_sg[FB_ind] = self.systre_sg[FB_ind] + [systre_sg[this_ind]]
                self.systre_rcsr[FB_ind] = self.systre_rcsr[FB_ind] + [
                    systre_rcsr[this_ind]
                ]
                self.systre_vec[FB_ind] = self.systre_vec[FB_ind] + [
                    systre_vec[this_ind]
                ]
                ind_list.append(this_ind)
        if len(systre_FB_el) > 0:
            print(len(ind_list))
            print(len(systre_FB_el))
            # print(systre_ID[np.setdiff1d(list(range(len(systre_FB_el))), ind_list)])
            print(np.setdiff1d(list(range(len(systre_FB_el))), ind_list))

        # pickle the results
        with open(f"crystal_net_results/{suffix}_TB_more_info.pickle", "wb") as f:
            pickle.dump({"gathered_results": self}, f)

max_dists = [1.02, 1.05, 1.1, 1.2, 1.4, 1.02, 1.05, 1.1, 1.2, 1.4]
decay_rates = [False, False, False, False, False, True, True, True, True, True]

if __name__ == "__main__":
    # get data you need to make the figures
    refresh_data = True
    param = ParamObj()
    
    if param.generated:
        mp_info = None
        unique_filename = "gen_unique_results.pickle"
    else:
        mp_info = mp_saved_info()
        unique_filename = "unique_results.pickle"

    data_objs = []
    
    for run_ind in range(len(max_dists)):
        if param.generated:
            file_suffix = f"gen_dmax{max_dists[run_ind]}_decay{decay_rates[run_ind]}"
        else:
            file_suffix = f"dmax{max_dists[run_ind]}_decay{decay_rates[run_ind]}"
            
        if refresh_data:
            data_objs.append(gathered_results(file_suffix, None))
        else:
            with open(f"crystal_net_results/{file_suffix}_TB_more_info.pickle", "rb") as f:
                data_objs.append(pickle.load(f)["gathered_results"])
                print(
                    f"max_dist {max_dists[run_ind]} with decay "
                    + f"{decay_rates[run_ind]} had {data_objs[run_ind].nFB_lats} flat band lattices"
                )
    
    # find unique flatband lattices
    if refresh_data:
        unique_FBs = copy.deepcopy(data_objs[0])
        unique_FBs.flat_with_no_decay = [True] * len(unique_FBs.mp_id)
        unique_FBs.flat_with_decay = [False] * len(unique_FBs.mp_id)
        unique_FBs.max_dists = [1.02] * len(unique_FBs.mp_id)
        unique_FBs.p02_no_decay = [True] * len(unique_FBs.mp_id)
        unique_FBs.p05_no_decay = [False] * len(unique_FBs.mp_id)
        unique_FBs.p1_no_decay = [False] * len(unique_FBs.mp_id)
        unique_FBs.p2_no_decay = [False] * len(unique_FBs.mp_id)
        unique_FBs.p4_no_decay = [False] * len(unique_FBs.mp_id)
        unique_FBs.p02_decay = [False] * len(unique_FBs.mp_id)
        unique_FBs.p05_decay = [False] * len(unique_FBs.mp_id)
        unique_FBs.p1_decay = [False] * len(unique_FBs.mp_id)
        unique_FBs.p2_decay = [False] * len(unique_FBs.mp_id)
        unique_FBs.p4_decay = [False] * len(unique_FBs.mp_id)
        
        for search_ind, search in enumerate(data_objs[1:]):
            for mat_ind, mat_id in enumerate(search.mp_id):
                spec_nm = search.specie_names[mat_ind]

                # look for other entries with the same ID, species, and NNN distance
                corr_ind = np.intersect1d(
                    np.nonzero(mat_id == unique_FBs.mp_id)[0],
                    np.nonzero(spec_nm == unique_FBs.specie_names)[0],
                )
                new_lat = True
                for test_ind in corr_ind:
                    # Fix: Handle empty NNN_dists or scalar comparison
                    val1 = unique_FBs.NNN_dists[test_ind]
                    val2 = search.NNN_dists[mat_ind]
                    if np.abs(val1 - val2) < 0.0001:
                        new_lat = False
                        # adds info on flat band hopping decay dependence
                        if decay_rates[search_ind + 1]:
                            unique_FBs.flat_with_decay[test_ind] = True
                            if max_dists[search_ind + 1] == 1.02:
                                unique_FBs.p02_decay[test_ind] = True
                            if max_dists[search_ind + 1] == 1.05:
                                unique_FBs.p05_decay[test_ind] = True
                            if max_dists[search_ind + 1] == 1.1:
                                unique_FBs.p1_decay[test_ind] = True
                            if max_dists[search_ind + 1] == 1.2:
                                unique_FBs.p2_decay[test_ind] = True
                            if max_dists[search_ind + 1] == 1.4:
                                unique_FBs.p4_decay[test_ind] = True
                        else:
                            unique_FBs.flat_with_no_decay[test_ind] = True
                            if max_dists[search_ind + 1] == 1.02:
                                unique_FBs.p02_no_decay[test_ind] = True
                            if max_dists[search_ind + 1] == 1.05:
                                unique_FBs.p05_no_decay[test_ind] = True
                            if max_dists[search_ind + 1] == 1.1:
                                unique_FBs.p1_no_decay[test_ind] = True
                            if max_dists[search_ind + 1] == 1.2:
                                unique_FBs.p2_no_decay[test_ind] = True
                            if max_dists[search_ind + 1] == 1.4:
                                unique_FBs.p4_no_decay[test_ind] = True

                # if it's new, add it to the unique_FBs object
                if new_lat:
                    unique_FBs.mp_id = np.append(
                        unique_FBs.mp_id, search.mp_id[mat_ind]
                    )
                    unique_FBs.chem_name = np.append(
                        unique_FBs.chem_name, search.chem_name[mat_ind]
                    )
                    unique_FBs.specie_names = np.append(
                        unique_FBs.specie_names, search.specie_names[mat_ind]
                    )
                    unique_FBs.nFBs = np.append(unique_FBs.nFBs, search.nFBs[mat_ind])
                    unique_FBs.nmatsites = np.append(
                        unique_FBs.nmatsites, search.nmatsites[mat_ind]
                    )
                    unique_FBs.nsites = np.append(
                        unique_FBs.nsites, search.nsites[mat_ind]
                    )
                    unique_FBs.NN_dists = np.append(
                        unique_FBs.NN_dists, search.NN_dists[mat_ind]
                    )
                    unique_FBs.NN_dists_max = np.append(
                        unique_FBs.NN_dists_max, search.NN_dists_max[mat_ind]
                    )
                    unique_FBs.NNN_dists = np.append(
                        unique_FBs.NNN_dists, search.NNN_dists[mat_ind]
                    )
                    unique_FBs.nelements = np.append(
                        unique_FBs.nelements, search.nelements[mat_ind]
                    )
                    unique_FBs.spacegroup_symbol.append(
                        search.spacegroup_symbol[mat_ind]
                    )
                    unique_FBs.spacegroup_number = np.append(
                        unique_FBs.spacegroup_number, search.spacegroup_number[mat_ind]
                    )
                    unique_FBs.energy_per_atom = np.append(
                        unique_FBs.energy_per_atom, search.energy_per_atom[mat_ind]
                    )
                    unique_FBs.band_gap = np.append(
                        unique_FBs.band_gap, search.band_gap[mat_ind]
                    )
                    unique_FBs.has_bs = np.append(
                        unique_FBs.has_bs, search.has_bs[mat_ind]
                    )
                    unique_FBs.icsd_ids = np.append(
                        unique_FBs.icsd_ids, search.icsd_ids[mat_ind]
                    )
                    unique_FBs.FB_dims.append(search.FB_dims[mat_ind])
                    unique_FBs.systre_keys.append(search.systre_keys[mat_ind])
                    unique_FBs.systre_sg.append(search.systre_sg[mat_ind])
                    unique_FBs.systre_rcsr.append(search.systre_rcsr[mat_ind])
                    unique_FBs.systre_vec.append(search.systre_vec[mat_ind])
                    unique_FBs.flat_with_decay.append(decay_rates[search_ind + 1])
                    unique_FBs.flat_with_no_decay.append(
                        not decay_rates[search_ind + 1]
                    )
                    unique_FBs.max_dists.append(max_dists[search_ind + 1])
                    unique_FBs.p02_no_decay.append(False)
                    unique_FBs.p05_no_decay.append(False)
                    unique_FBs.p1_no_decay.append(False)
                    unique_FBs.p2_no_decay.append(False)
                    unique_FBs.p4_no_decay.append(False)
                    unique_FBs.p02_decay.append(False)
                    unique_FBs.p05_decay.append(False)
                    unique_FBs.p1_decay.append(False)
                    unique_FBs.p2_decay.append(False)
                    unique_FBs.p4_decay.append(False)
                    if decay_rates[search_ind + 1]:
                        if max_dists[search_ind + 1] == 1.02:
                            unique_FBs.p02_decay[-1] = True
                        if max_dists[search_ind + 1] == 1.05:
                            unique_FBs.p05_decay[-1] = True
                        if max_dists[search_ind + 1] == 1.1:
                            unique_FBs.p1_decay[-1] = True
                        if max_dists[search_ind + 1] == 1.2:
                            unique_FBs.p2_decay[-1] = True
                        if max_dists[search_ind + 1] == 1.4:
                            unique_FBs.p4_decay[-1] = True
                    else:
                        if max_dists[search_ind + 1] == 1.02:
                            unique_FBs.p02_no_decay[-1] = True
                        if max_dists[search_ind + 1] == 1.05:
                            unique_FBs.p05_no_decay[-1] = True
                        if max_dists[search_ind + 1] == 1.1:
                            unique_FBs.p1_no_decay[-1] = True
                        if max_dists[search_ind + 1] == 1.2:
                            unique_FBs.p2_no_decay[-1] = True
                        if max_dists[search_ind + 1] == 1.4:
                            unique_FBs.p4_no_decay[-1] = True
        
        if not param.generated:
            print(f"{len(unique_FBs.mp_id)} flat band sublattices found in mp")
        else:
            print(f"{len(unique_FBs.mp_id)} flat band sublattices found in generated materials")

        with open(f"crystal_net_results/{unique_filename}", "wb") as f:
            pickle.dump({"unique_FBs": unique_FBs}, f)

        # save cases where duplicate (mp_id, species) have different RCSR names
        from collections import defaultdict
        import csv

        mp = list(unique_FBs.mp_id)
        sp = list(unique_FBs.specie_names)
        nnn = list(unique_FBs.NNN_dists)
        rcsr = list(unique_FBs.systre_rcsr)

        by = defaultdict(list)
        for i in range(len(mp)):
            by[(mp[i], sp[i])].append(i)

        rows = []
        for (mp_id, species), idxs in by.items():
            if len(idxs) < 2:
                continue
            entry_rcsr = []
            for i in idxs:
                rset = sorted({r.strip() for r in rcsr[i] if isinstance(r, str) and r.strip()})
                entry_rcsr.append((float(nnn[i]), tuple(rset)))
            unique_sets = {rc for _, rc in entry_rcsr}
            if len(unique_sets) > 1:
                for nnn_val, rset in sorted(entry_rcsr, key=lambda x: x[0]):
                    rows.append({
                        "mp_id": mp_id,
                        "species": species,
                        "NNN_dist": nnn_val,
                        "rcsr_names": ";".join(rset) if rset else "",
                    })

        out_path = "crystal_net_results/duplicate_id_species_rcsr_mismatch.csv"
        with open(out_path, "w", newline="") as f_csv:
            writer = csv.DictWriter(
                f_csv, fieldnames=["mp_id", "species", "NNN_dist", "rcsr_names"]
            )
            writer.writeheader()
            writer.writerows(rows)
    else:
        with open(f"crystal_net_results/{unique_filename}", "rb") as f:
            unique_FBs = pickle.load(f)["unique_FBs"]
            if param.generated:
                print(
                    f"loaded all {len(unique_FBs.mp_id)} unique flat band sublattices found in generated materials"
                )
            else:
                print(
                    f"loaded all {len(unique_FBs.mp_id)} unique flat band sublattices found in the MP"
                )
                print(f"")
                print(
                    f"{len(np.unique(unique_FBs.mp_id))} "
                    f"({np.round(100*len(np.unique(unique_FBs.mp_id))/len(unique_FBs.mp_id), 3)} %) "
                    f"materials contain at least one flat band sublattice"
                )
                print(
                    f"{len(mp_info.mp_ids)} materials in Materials Project with "
                    f"{np.sum(mp_info.nelements)} elemental sublattices\n"
                )

    # NOTE: graph-invariant-dependent analysis and outputs were moved to
    # collect_and_output_graph_invariants.py. Run that script after
    # get_graph_invariants to generate the remaining outputs.
    print("done!")