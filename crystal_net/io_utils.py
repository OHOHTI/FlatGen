from concurrent.futures import ThreadPoolExecutor
import pickle
import json
import os
import pandas as pd
import traceback
from mp_api.client import MPRester
from pymatgen.core import IStructure
import numpy as np
from pathlib import Path
from pymatgen.core.composition import Composition
from functools import partial


# other materials science-ey libraries
from mendeleev import element

class TBoutputs():
    r"""
    A class that can store the outputs from a TB output datafile.

    Must supply with a valid file name string to the datafile you want
    to get the outputs of. Then the data is stored into the following variables:

    Properties:
        header (int): The number of header lines.
        nFB_lats (int): The number of materials in the output file.
        mp_id (str list): The nMatsx1 list of materials project IDs.
        chem_name (str list): The nMatsx1 list of materials chemical names.
        specie_names (str mat.): A nMatsx10 array of names of sublattice species for each material.
        nFBs (int mat.): The corresponding number of flat bands for each specie.
        nsites (int mat): The corresponding number of sites for each specie.
        NN_dists (float mat): The corresponding nearest neighbor distance in angstroms.
        NN_dists_max (float mat): The corresponding largest nearest neighbor distance included in the model.
        NNN_dists (float mat): The first neighbor distance not included in angstroms.
    """

    def __init__(self, file_name, rmv_nonflat=True):
        r"""Loads all lists of results from file."""
        # get number of header lines
        with open(file_name, "r") as f_txt:
            for line_ind, line in enumerate(f_txt.readlines()):
                if line.startswith("mp_ID, "):
                    self.header = line_ind + 1
                    break

        # get all the date from the file
        self.mp_id = np.genfromtxt(file_name, delimiter=', ', skip_header=self.header,
                                   usecols=0, dtype=str, invalid_raise=False)
        self.chem_name = np.genfromtxt(file_name, delimiter=', ', skip_header=self.header,
                                       usecols=1, dtype=str, invalid_raise=False)
        self.specie_names = np.genfromtxt(file_name, delimiter=', ', skip_header=self.header,
                                          usecols=2, dtype=str,
                                          missing_values='', invalid_raise=False)
        self.nFBs = np.genfromtxt(file_name, delimiter=', ', skip_header=self.header,
                                  usecols=3, dtype=np.int16,
                                  filling_values=0, invalid_raise=False)
        self.nsites = np.genfromtxt(file_name, delimiter=', ', skip_header=self.header,
                                    usecols=4, dtype=np.int16,
                                    filling_values=0, invalid_raise=False)
        self.NN_dists = np.genfromtxt(file_name, delimiter=', ', skip_header=self.header,
                                      usecols=5, dtype=np.double,
                                      filling_values=0, invalid_raise=False)
        self.NN_dists_max = np.genfromtxt(file_name, delimiter=', ', skip_header=self.header,
                                          usecols=6, dtype=np.double,
                                          filling_values=0, invalid_raise=False)
        self.NNN_dists = np.genfromtxt(file_name, delimiter=', ', skip_header=self.header,
                                       usecols=7, dtype=np.double,
                                       filling_values=0, invalid_raise=False)
        # genfromtxt with invalid_raise=False drops rows that lack col 9 entirely,

        # filter to just lattices containing flat bands
        FB_inds = np.nonzero(self.nFBs > 0)[0]
        if rmv_nonflat:
            self.mp_id = self.mp_id[FB_inds]
            self.chem_name = self.chem_name[FB_inds]
            self.specie_names = self.specie_names[FB_inds]
            self.nFBs = self.nFBs[FB_inds]
            self.nsites = self.nsites[FB_inds]
            self.NN_dists = self.NN_dists[FB_inds]
            self.NN_dists_max = self.NN_dists_max[FB_inds]
            self.NNN_dists = self.NNN_dists[FB_inds]
        self.nFB_lats = len(self.mp_id)
        for ind in range(len(self.mp_id)):
            if self.mp_id[ind] == "mp-qfh":
                print(f"ding: {self.mp_id[ind]}, {self.chem_name[ind]}, {ind}")

def init_file(param, save_path, save_path_suffix=""):
    r"""Creates and begins a TB output file."""
    if param.check_TB_flat:
        f_txt = open(
            save_path / (f"{save_path_suffix}_TB_search_results.txt"), "a", buffering=1)
        f_txt.write(f"File output from automated tight binding model "
                    "generator written by Paul M. Neves.\nSearches for (not trivially localized) flat "
                    "band hosting tight binding models using an s-orbital\ntight binding model for each "
                    "species' sublattice.\n\nParameters of search:\n")
        if param.dist_units == 'AA':
            f_txt.write(f"Maximum considered hopping distance: {param.max_dist} angstroms\n"
                        f"Decay rate of hopping strength: {param.decay_rate} (in units of angstroms)\n")
        else:
            f_txt.write(f"Maximum considered hopping distance: {param.max_dist} times the N.N. "
                        f"distance\nDecay rate of hopping strength: {param.decay_rate} (in units of "
                        f"N.N. distance)\n")
        f_txt.write(f"Maximum width of flat band (as a fraction of total band energy spread): "
                    f"{param.max_fb_width}\nFraction of all calculated k-points that need to be flat "
                    f"to count: {param.flatness_tol}\nNumber of hops considered when checking if the "
                    f"flat band is trivial: {param.check_to_N_hops}\n")
        if param.full_flat:
            f_txt.write("Required band to exist in all kpoints samples\n")
        if param.high_symm_only:
            f_txt.write(
                "Only checked band flatness along high symmetry directions\n\n")
        else:
            f_txt.write("Sampled entire 1BZ for band flatness\n\n")
        f_txt.write(
            "mp_ID, formula, El, Nfbs, nsites, nn_dist, nn_dist_max, nnn_dist, fb_str\n")

        # also save param object
        with open(save_path / (f"{save_path_suffix}_TB_params.json"), 'w') as f_json:
            json.dump(param.as_dict(), f_json, indent=4)
        with open(save_path / (f"{save_path_suffix}_TB_params.pickle"), 'wb') as f_pickle:
            pickle.dump({"param": param}, f_pickle)
    return f_txt

def save_struct_file(param, save_path, mat_ID):
    r"""Saves the structure of the desired material as a cif file."""
    structure = None
    try:
        structure = get_struct_from_MP(param, mat_ID)
    except Exception:
        print(traceback.format_exc())
        pass
    if structure is not None and structure != []:
        # save to file
        print(f"Now saving struct of: {mat_ID}")
        newfile_path = save_path / (f"{mat_ID}.cif")
        serialized_struct = structure.to(fmt="cif", filename=newfile_path)
    return

def get_struct_from_MP(param, mat_ID):
    with MPRester(param.MP_KEY, notify_db_version=False) as m:
        return m.get_structure_by_material_id(mat_ID)

def load_structure(param, mat_ID):
    if param.generated:
        file_path = param.script_path / Path(f"Materials_gen/{mat_ID}.cif")
    else:
        file_path = param.script_path / Path(f"mp_structs/{mat_ID}.cif")
    if file_path.exists():
        structure = IStructure.from_file(file_path)
    else:
        structure = get_struct_from_MP(param, mat_ID)
    
    if param.reduce_structure and structure:
        structure = structure.get_reduced_structure(reduction_algo='niggli')
    return structure

def parse_systre(fname):
    r"""Gets key, if it exists, in the Systre output saved to a file.

    If no key found (ie, in case of some error), returns None.
    If more than one key in file, returns last one found.

    Can call systre and save to a file with:
    java -cp Systre-19.6.0.jar org.gavrog.apps.systre.SystreCmdline foo.cgd --systreKey > bar.txt
    """
    systre_key = None
    dimension = 3
    systre_sg = None
    rcsr_nm = None
    fb_vec = ""
    collision_str = "!!! ERROR (STRUCTURE) - Structure has collisions between next-nearest neighbors."
    ladder_str = "!!! ERROR (STRUCTURE) - Structure is non-crystallographic (a 'ladder')."
    scd_str = "!!! ERROR (STRUCTURE) - Structure has second-order collisions."

    op_str_1 = "!!! ERROR (INTERNAL) - Unexpected java.lang.RuntimeException: Spacegroup finder messed up operators.\n"
    op_str_2 = "!!!       \tat org.gavrog.apps.systre.SystreCmdline.showSpaceGroup(SystreCmdline.java:525)\n"
    op_maybe = True
    inv_str_1 = "!!! ERROR (INTERNAL) - Unexpected java.lang.ArithmeticException: matrix has no inverse\n"
    inv_str_2 = "!!!       \tat org.gavrog.jane.compounds.Matrix.inverse(Matrix.java:748)\n"
    inv_maybe = True
    nul_str_1 = "!!! ERROR (INTERNAL) - Unexpected java.lang.NullPointerException: null\n"
    nul_str_2 = "!!!       \tat org.gavrog.jane.compounds.Matrix.setSubMatrix(Matrix.java:226)\n"
    nul_maybe = True

    try:
        with open(fname, "r") as f_systre:
            for line in f_systre:
                # what we want to see
                if line[:16] == '   Systre key: "':
                    systre_key = line[16:-2]
                if line[:18] == "      dimension = ":
                    dimension = int(line[18])
                if line.startswith("   Ideal space group is "):
                    systre_sg = line[24:-2]
                if line.startswith("       Name:		"):
                    rcsr_nm = line[14:-1]

                # things Systre can't handle
                if line[:80] == collision_str:
                    systre_key = "NN_COLLISION"
                if line[:71] == ladder_str:
                    systre_key = "LADDER"
                if line[:62] == scd_str:
                    systre_key = "SEC_ORD_COLLISION"

                # error in Systre 1
                if line == op_str_1:
                    op_maybe = True
                if line == op_str_2 and op_maybe:
                    systre_key = "OP_MESS"

                # error in Systre 2
                if line == inv_str_1:
                    inv_maybe = True
                if line == inv_str_2 and inv_maybe:
                    systre_key = "INV_ERR"

                # error in Systre 3
                if line == nul_str_1:
                    nul_maybe = True
                if line == nul_str_2 and nul_maybe:
                    systre_key = "NULL_ERR"
    except Exception:
        pass

    try:
        cgd_name = str(fname)[:-4] + ".cgd"
        with open(cgd_name, "r") as f_systre:
            start_recording = False
            for line in f_systre:
                if line.startswith("EDGES"):
                    start_recording = True
                elif start_recording:
                    if line.startswith("END"):
                        start_recording = False
                    else:
                        fb_vec = fb_vec + line.strip() + " \\newline "
    except Exception:
        fb_vec = ""
        raise
    # if systre_key is None:
    #    os.system(f"cat {fname}")
    return systre_key, dimension, systre_sg, rcsr_nm, fb_vec

def load_all_MP_names(info_path="."):
    info_path = Path(info_path)
    material_IDs = np.atleast_1d(np.genfromtxt(info_path / 'all_mp_IDs.txt', dtype='U25', autostrip=True))
    names_list = np.atleast_1d(np.genfromtxt(info_path / 'all_mp_names.txt', dtype='U25', autostrip=True))
    nsites_list = np.atleast_1d(np.genfromtxt(info_path / 'all_mp_nsites.txt', dtype='u4', autostrip=True))
    return material_IDs, names_list, nsites_list

def make_cgd_string(mat_ID, species, comp_ind, fullhop_array, comp_with_FBs):
    r"""Turns the inputs (generated by do_TB_model) into a gcd file for Gavrog"""
    cgd_str = ""
    cgd_str = cgd_str + "PERIODIC_GRAPH\n"
    cgd_str = cgd_str + f"ID {mat_ID}_{species}_{comp_ind}\n"
    cgd_str = cgd_str + "EDGES\n"
    # write edge list
    n_edges = len(fullhop_array)
    for edge_ind in range(n_edges):
        this_edge = fullhop_array[edge_ind]
        if (this_edge[0] in comp_with_FBs) or (this_edge[1] in comp_with_FBs):
            cgd_str = (cgd_str
                       + f"  {int(this_edge[0])} "
                       + f"{int(this_edge[1])} {int(this_edge[2])} "
                       + f"{int(this_edge[3])} {int(this_edge[4])}\n")
    cgd_str = cgd_str + "END\n\n"
    return cgd_str

def refresh_MP_data_required(param, ids_file, save_directory):
    r"""Downloads and saves IDs stored in a csv file ids_file, chemical names, site numbers, and structures from the MP."""
    # creates a list of all elements
    all_elements = [0]*118
    for elem_ind in range(118):
        all_elements[elem_ind] = element(elem_ind+1).symbol

    # gets... EVERY materials project ID that has ANY element in it (THIS WILL TAKE A LITTLE WHILE)
    print("Obtaining required MP materials...")
    ids_df = pd.read_csv(ids_file, sep='\t')
    target_ids = ids_df['Key'].tolist()
    compounds = []
    compounds_list = []
    for i in range(len(target_ids)):
        try:
            with MPRester(param.MP_KEY) as mpr:
                docs = mpr.materials.summary.search(
                        material_ids=target_ids[i],
                        fields=['formula_pretty', 'elements', 'nsites']
                    )

            # extracts the names, elements, and nsites of each material
            dic = {'material_id':target_ids[i], 
                'formula_pretty':docs[0]['formula_pretty'], 
                'elements': docs[0]['elements'],
                'nsites':docs[0]['nsites']}
            compounds.append(dic)
        except Exception as e:
            pass

    print(f"{len(compounds)} compounds found")
    compounds_list = ['']*len(compounds)
    best_names = ['']*len(compounds)
    num_sites = ['']*len(compounds)
    for comp_ind in range(len(compounds)):
        # collects mp IDs
        compounds_list[comp_ind] = compounds[comp_ind]['material_id']

        # collects best names
        pretty_formula = compounds[comp_ind]['formula_pretty']
        element_list = compounds[comp_ind]['elements']
        if not not pretty_formula:
            best_name = pretty_formula
        elif not not element_list:
            best_name = ', '.join(element_list)
        else:
            best_name = ''
        best_names[comp_ind] = best_name

        # collects number of sites
        num_sites[comp_ind] = compounds[comp_ind]['nsites']

    # saves all these lists
    save_directory = Path(save_directory)
    with open(save_directory / 'all_mp_IDs.txt', 'w') as f_id:
        f_id.writelines("%s\n" % mp_id for mp_id in compounds_list)

    with open(save_directory / 'all_mp_names.txt', 'w') as f_nm:
        f_nm.writelines("%s\n" % name for name in best_names)

    with open(save_directory / 'all_mp_nsites.txt', 'w') as f_ns:
        f_ns.writelines("%s\n" % nsites for nsites in num_sites)

    # downloads all the structures
    structs_directory = save_directory / 'mp_structs'
    save_all_struct_files = partial(save_struct_file, param, structs_directory)
    if not os.path.isdir(structs_directory):
        os.makedirs(structs_directory)
    with ThreadPoolExecutor() as executor:
        executor.map(save_all_struct_files, compounds_list)

    return

def collect_systre_keys(save_path_suffix):
    r"""Extracts systre strings from calculation with timestring MMDDYY_HHMMSS."""

    # get output directory
    script_path = Path(os.path.dirname(os.path.realpath(__file__)))
    try:
        param = pickle.load(
            open(script_path / f"crystal_net_results/TB_{save_path_suffix}/{save_path_suffix}_TB_params.pickle", "rb"))["param"]
    except Exception:
        print("Couldn't find param pickle for this run.")
        print(traceback.format_exc())
        raise
    save_path = param.script_path / (f"crystal_net_results/TB_{save_path_suffix}/")
    num_mats = len(param.material_IDs)

    # get list of systre out files to parse through
    systre_files = []
    mat_IDs = []
    specie_list = []
    comp_list = []
    file_list = os.listdir(save_path)
    print("Identifying systre output files...")
    for mat_ind in range(num_mats):
        mat_ID = param.material_IDs[mat_ind]
        best_name = param.names_list[mat_ind]
        if mat_ind % 10000 == 0:
            print(
                f"{np.round(100 * mat_ind / num_mats, 1)} % loaded ({mat_ind} of {num_mats})")
        try:
            for fname in os.listdir(save_path / mat_ID):
                if fname[-4:] == ".out":
                    systre_files.append(save_path / mat_ID / fname)
                    _inds = [i for i, ltr in enumerate(fname) if ltr == "_"]
                    mat_IDs.append(fname[:_inds[0]])
                    specie_list.append(fname[_inds[0]+1:_inds[1]])
                    comp_list.append(fname[_inds[1]+1:-4])
        except Exception:
            print(traceback.format_exc())
            pass
    num_FBs = len(systre_files)
    print(f"{num_FBs} systre output files found\n")

    # get systre results
    systre_keys = []
    dim_list = []
    systre_sg_list = []
    rcsr_nm_list = []
    print("Now parsing sytre output files...")
    for sys_ind, systre_file in enumerate(systre_files):
        if sys_ind % 10000 == 0:
            print(
                f"{np.round(100 * sys_ind / num_FBs, 1)} % complete ({sys_ind} of {num_FBs})")
        systre_key, dim, systre_sg, rcsr_nm, fb_vec = parse_systre(systre_file)
        systre_keys.append(systre_key)
        dim_list.append(dim)
        systre_sg_list.append(systre_sg)
        rcsr_nm_list.append(rcsr_nm)
    print("Finished parsing systre output files.\n")

    # save results
    print(f"Writing systre results to {save_path_suffix}_systre_keys.txt")
    with open(param.script_path / "crystal_net_results" / f"{save_path_suffix}_systre_keys.txt", "w") as f_sys:
        f_sys.write(
            "File containing the Systre keys (see Gavrog project) of all the flat\n")
        f_sys.write(
            f"-band models found in tight binding search {save_path_suffix}.\n\n")
        f_sys.write(
            "mat_ID, species, FB_comp, FB_dim, systre_key, spacegroup, rcsr_name\n")
        for sys_ind in range(num_FBs):
            write_str = (f"{mat_IDs[sys_ind]}, {specie_list[sys_ind]}, "
                         + f"{comp_list[sys_ind]}, {dim_list[sys_ind]}, "
                         + f"{systre_keys[sys_ind]}, {systre_sg_list[sys_ind]}, "
                         + f"{rcsr_nm_list[sys_ind]}, {fb_vec}\n")
            f_sys.write(write_str)
    print("Finished writing to file.\n")
    return
