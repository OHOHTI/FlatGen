import multiprocessing
from functools import partial
from time import perf_counter
from config import ParamObj
from io_utils import init_file, load_structure, refresh_MP_data_required, collect_systre_keys
from physics_engine import do_TB_model
import matplotlib.pyplot as plt
import os
import json
import pickle
import traceback
from pathlib import Path
from plot_utils import plot_structure, plotting_bs_stuff
import numpy as np



num_mats = None  # to be set later
# Global placeholders for multiprocessing

def do_mat_calcs(param, matInd):
    # initialization stuff
    global num_started, num_complete
    mat_ID = param.material_IDs[matInd]
    this_start_time = perf_counter()
    num_started.value = num_started.value + 1
    this_comp_num = num_started.value
    plt.close(fig='all')
    this_save_path = save_path / mat_ID
    write_str = ""
    ok_to_save = True

    if not os.path.isdir(this_save_path):
        os.makedirs(this_save_path)

    print_str = f"\nResults for material {mat_ID}:\n"
    print_str = f"{print_str} - Compound:   {int(this_comp_num)} of {num_mats}\n"
    # do all the desired calculations
    try:
        # print material chemical formula
        best_name = param.names_list[matInd]
        nsites = param.nsites_list[matInd]
        print_str = f"{print_str} - Chemical formula:   {best_name}\n"

        # load chemical structure
        structure = load_structure(param, mat_ID)

        # plot 3D structure, if requested
        if param.plot_structure_3D:
            plot_structure(param, structure, best_name, mat_ID,
                           plot_save_path=this_save_path)

        # print structure data, if requested
        if param.print_structure_data:
            print_str = f"{print_str}\n{repr(structure)}\n"

        # plotting of BS, DOS, and 1BZ
        if param.plot_BS or param.plot_DOS or param.plot_DOS_and_BS or param.plot_BZ:
            print_str = plotting_bs_stuff(param, mat_ID, print_str, save_path=this_save_path)

        # plot TB, if requested
        if param.plot_TB or param.check_TB_flat:
            specie_list = None
            (print_str, write_str, gras, n_FBs_list, specie_list, fullhop_arrays, sublat_graphs,
             k_vec_list, evals_list, comps_with_FBs_list) = do_TB_model(
                param, structure, print_str, write_str, this_save_path, best_name, mat_ID)
            if specie_list is None:
                ok_to_save = False

    # keeps going if there is an error accessing the database for a material
    except KeyboardInterrupt:
        lock.acquire()
        print("\nStopping at user request...")
        ok_to_save = False
        lock.release()
        raise
    except Exception as e:
        #lock.acquire()
        #print_str = (f"{print_str} - There was an error retrieving data on this compound (see "
        #             f"above error message)!!\n")
        #error_msg = traceback.format_exc()
        #write_str = f"ERROR - {mat_ID}\n" + str(error_msg)
        #print(error_msg)
        ok_to_save = False
        #lock.release()
        if os.path.isdir(this_save_path):
            import shutil
            shutil.rmtree(this_save_path)
            
        if param.verbose:
            print(f"Error on {mat_ID}: {e}")

    lock.acquire()
    if param.verbose:
        print(print_str)
    if param.check_TB_flat and param.save_results and ok_to_save:
        f_txt.write(write_str)
        f_txt.flush()

    # save calculated results to json file.
    if param.save_results:
        if (param.plot_TB or param.check_TB_flat) and ok_to_save:
            with open(this_save_path / (f"{mat_ID}_{best_name}_results.json"), 'w') as f_json:
                save_dict = {"best_name": best_name,
                             "nsites": int(nsites),
                             "mat_ID": mat_ID,
                             "write_str": write_str,
                             "print_str": print_str,
                             "n_FBs_list": n_FBs_list,
                             "specie_list": specie_list,
                             "structure": structure.as_dict(),
                             "fullhop_arrays": fullhop_arrays,
                             "sublat_graphs": sublat_graphs,
                             # "k_vec_list": k_vec_list,
                             # "evals_list": evals_list,
                             "comps_with_FBs_list": comps_with_FBs_list
                             }
                json.dump(save_dict, f_json, indent=4)
            with open(this_save_path / (f"{mat_ID}_{best_name}_tbmodel.pickle"), 'wb') as f_pickle:
                pickle.dump(gras, f_pickle)
        elif ok_to_save:
            with open(this_save_path / (f"{mat_ID}_{best_name}_results.json"), 'w') as f_json:
                save_dict = {"best_name": best_name,
                             "nsites": int(nsites),
                             "mat_ID": mat_ID,
                             "write_str": write_str,
                             "print_str": print_str,
                             "structure": structure.as_dict()
                             }
                json.dump(save_dict, f_json, indent=4)

    # gets time it took to do all the calculations for this material
    num_complete.value = num_complete.value + 1
    this_run_time = perf_counter() - this_start_time
    total_time = perf_counter() - start_time_all

    avg_time = total_time/(num_complete.value+1)  # in seconds
    if param.verbose:
        print(f"This material took {np.round(this_run_time,3)} s to run.\nAverage time per "
              f"material: {np.round(avg_time,3)} s ("
              f"{np.round((num_mats-num_complete.value-1) * avg_time / 3600, 3)} hrs est. left)\n")
    lock.release()
    return


def run_all_calcs(param, save_path_suffix=""):
    global num_started, save_path, num_mats, lock, names_list, nsites_list, f_txt
    global num_complete, start_time_all

    start_time_all = perf_counter()
    lock = multiprocessing.Lock()
    num_procs = multiprocessing.cpu_count()
    num_started = multiprocessing.Value('d', 0.0)
    num_complete = multiprocessing.Value('d', 0.0)
    print(f"Starting {len(param.material_nums)} materials on {num_procs} cores.")
    try:
        save_path = param.script_path / (f"crystal_net_results/TB_{save_path_suffix}/")
        if not os.path.isdir(save_path) and param.save_results:
            os.makedirs(save_path)
    except Exception:
        print("Something went wrong with establishing the output directory or finding this"
              " script\'s directory")
        print("Make sure you are not running inside an interpreter, but on the command line!!!")
        print(traceback.format_exc())
        pass

    if param.check_TB_flat and param.save_results:
        f_txt = init_file(param, save_path, save_path_suffix)
    with multiprocessing.Pool(processes=num_procs) as pool:
        func = partial(do_mat_calcs, param)
        pool.map(func, param.material_nums)
    # clean up
    if param.check_TB_flat and param.save_results:
        f_txt.close()
    return

if __name__ == "__main__":
    param = ParamObj()
    max_dists = [1.02, 1.05, 1.1, 1.2, 1.4, 1.02, 1.05, 1.1, 1.2, 1.4]
    decay_rates = [False, False, False, False, False, True, True, True, True, True]
    if False:
        ids_required_file = param.script_path / "MP_95.txt"
        refresh_MP_data_required(param, ids_required_file, '.')
    if param.generated:
        param.material_nums = param.script_path / "Materials_gen/" 
    for run_ind in range(len(max_dists)):
        this_param = ParamObj()#np.random.randint(0,120000,size=100)
        this_param.max_dist = max_dists[run_ind]
        this_param.decay_rate = decay_rates[run_ind]
        if param.generated:
            save_path_suffix=f"dmax{this_param.max_dist}_decay{this_param.decay_rate}"
        else:
            save_path_suffix=f"dmax{this_param.max_dist}_decay{this_param.decay_rate}"
        run_all_calcs(this_param, save_path_suffix=save_path_suffix)
        collect_systre_keys(save_path_suffix=save_path_suffix)
    
    