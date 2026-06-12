'''
acquire data from Materials Project API (bands and dos) and calculate S_total
'''
import gc
import os
import json
import numpy as np
import pandas as pd
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.dos import CompleteDos
from pymatgen.symmetry.bandstructure import HighSymmKpath
import pickle
import signal
from sdos_sband import calculate_s_bandwidth, calculate_dos_and_scores, save_scores
from prepare_flatestband import save_bs_label
#from pymatgen.ext.matproj import MPRester
from mp_api.client import MPRester
import argparse
import random

BATCH_SIZE = 1 # Process and save 10 materials at a time to clear memory


def get_mp_api_key():
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError("Set MP_API_KEY before running MP_sbands_sdos.py.")
    return api_key

def append_to_jsonl(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # avoid duplication
    existing_ids = set()
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                existing_ids.add(json.loads(line).get("material_id"))
    if data.get("material_id") not in existing_ids:
        with open(filename, 'a') as f:
            f.write(json.dumps(data) + '\n')
                
def timeout_handler(signum, frame):
    """Custom exception raised when the timeout is triggered."""
    raise TimeoutError("Function execution time exceeded maximum limit.")

def preprocess_band_structure(bandstructure_dict, bands_ef, num_bands, if_spin_polarized=False):
    bands_dict = bandstructure_dict["bands"]
    Efermi = float(bands_ef)
    selected_bands_dict = {}
    original_indices_dict = {}  # Store original band indices
    spins = ['1', '-1'] if if_spin_polarized else ['1']
    for spin in spins:
        if spin not in bands_dict:
            continue
        bands = np.array(bands_dict[spin])
        avg_energies = np.mean(bands, axis=1)
        avg_distances = np.abs(avg_energies - Efermi)
        sorted_indices = np.argsort(avg_distances)
        num_bands_total = min(num_bands, len(sorted_indices))
        selected_indices = sorted_indices[:num_bands_total].tolist()
        selected_bands = bands[selected_indices, :]
        selected_bands_dict[spin] = selected_bands
        original_indices_dict[spin] = selected_indices  # Map selected bands to original indices

    return selected_bands_dict, Efermi, original_indices_dict


def reduce_num_k_points(kp_original, eigenv, max_kpoints=120):
    num_kpoints = kp_original.shape[0]
    if num_kpoints <= max_kpoints:
        return kp_original, eigenv
    step = num_kpoints // max_kpoints
    reduced_kp = kp_original[::step]
    reduced_bands = {}
    for spin in eigenv:
        reduced_bands[spin] = eigenv[spin][:, ::step]
    return reduced_kp, reduced_bands

def load_and_process_bandstructure(bs, num_bands=6):
    assert isinstance(bs, BandStructureSymmLine), "Input must be a BandStructureSymmLine object."
    if_spin_polarized = bs.is_spin_polarized
    try:
        continuous_bands = HighSymmKpath.get_continuous_path(bs)
        bandstructure_dict = continuous_bands.as_dict()
        kp_original = np.array([kp.cart_coords for kp in continuous_bands.kpoints])
    except:
        bandstructure_dict = bs.as_dict()
        kp_original = np.array([kp.cart_coords for kp in bs.kpoints])
    bands_ef = bs.efermi
    eigenv, efermi, original_indices = preprocess_band_structure(
        bandstructure_dict, bands_ef, num_bands, if_spin_polarized=if_spin_polarized
    )
    kp_reduced, eigenv_reduced = reduce_num_k_points(kp_original, eigenv, max_kpoints=120)
    return kp_reduced, eigenv_reduced, efermi, if_spin_polarized, original_indices

def find_intersections_and_segments(bands, original_indices, epsilon=0.015):
    num_bands, num_kpoints = bands.shape
    potential_intersections = []
    for x in range(num_kpoints):
        energies = bands[:, x]
        for i in range(num_bands):
            for j in range(i + 1, num_bands):
                if abs(energies[i] - energies[j]) < epsilon:
                    if (x, i, j) not in potential_intersections:
                        potential_intersections.append((x, i, j))
                if x > 0:
                    prev_energies = bands[:, x - 1]
                    if abs(energies[i] - prev_energies[j]) < epsilon:
                        if (x, i, j) not in potential_intersections:
                            potential_intersections.append((x, i, j))
                if x < num_kpoints - 1:
                    next_energies = bands[:, x + 1]
                    if abs(energies[i] - next_energies[j]) < epsilon:
                        if (x, i, j) not in potential_intersections:
                            potential_intersections.append((x, i, j))
    print(f"Number of bands: {num_bands}, Number of k-points: {num_kpoints}")
    print(f"Number of potential intersections: {len(potential_intersections)}")
    potential_intersections = list(dict.fromkeys(potential_intersections))

    potential_intersections.sort(key=lambda x: (x[1], x[2], x[0]))
    merged_intersections = []

    i = 0
    while i < len(potential_intersections):
        start_x, band_i, band_j = potential_intersections[i]
        current_x = start_x
        current_band_i = band_i
        current_band_j = band_j

        j = i + 1
        while j < len(potential_intersections):
            next_x, next_band_i, next_band_j = potential_intersections[j]
            if (next_band_i != current_band_i or next_band_j != current_band_j) or (next_x != current_x + 1):
                break
            current_x = next_x
            j += 1

        merged_intersections.append((start_x, current_band_i, current_band_j))
        if j > i + 1:
            merged_intersections.append((current_x, current_band_i, current_band_j))

        i = j

    merged_intersections.sort(key=lambda x: x[0])

    band_segments = [[] for _ in range(num_bands)]
    intersection_points = {}
    for x, band_i, band_j in merged_intersections:
        if band_i not in intersection_points:
            intersection_points[band_i] = set()
        if band_j not in intersection_points:
            intersection_points[band_j] = set()
        intersection_points[band_i].add(x)
        intersection_points[band_j].add(x)
    for band_idx in range(num_bands):
        original_band_idx = original_indices[band_idx]
        if band_idx not in intersection_points:
            band_segments[band_idx].append((0, num_kpoints, original_band_idx, band_idx + 1))
        else:
            points = sorted(list(intersection_points[band_idx]))
            if not points:
                band_segments[band_idx].append((0, num_kpoints, original_band_idx, band_idx + 1))
            else:
                start = 0
                for x in points:
                    if start < x:
                        band_segments[band_idx].append((start, x, original_band_idx, band_idx + 1))
                    start = x
                if start < num_kpoints:
                    band_segments[band_idx].append((start, num_kpoints, original_band_idx, band_idx + 1))
    print(f"Number of intersections after merging: {len(merged_intersections)}")
    return band_segments, merged_intersections, potential_intersections

def calculate_flatness_score(kpoints, bands, original_indices, omega, epsilon=0.015, max_time=600):

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(max_time)
    try:
        num_bands, num_kpoints = bands.shape
        if num_kpoints != kpoints.shape[0]:
            raise ValueError(f"Mismatch in kpoints ({kpoints.shape[0]}) and bands ({num_kpoints}) dimensions")
        band_segments, intersections, potential_intersections = find_intersections_and_segments(bands, original_indices, epsilon)
        
        curves = []
        intersections_by_x = {}
        if len(intersections) > 100:
            print("Too many intersections, skipping flatness calculation.")
            return {"timeout": True}
        for x, band_i, band_j in intersections:
            if x not in intersections_by_x:
                intersections_by_x[x] = []
            intersections_by_x[x].append((x, band_i, band_j))
        for start_band in range(num_bands):
            segments = band_segments[start_band]
            if not segments or segments[0][0] != 0:
                continue
            current_seg = segments[0]
            current_x = current_seg[1]
            current_curve = [(start_band, current_seg)]
            if current_x == num_kpoints:
                curves.append(current_curve)
                continue

            def extend_curve(curve, x):
                if x >= num_kpoints:
                    if curve[-1][1][1] == num_kpoints:
                        curves.append(curve)
                    return
                current_band = curve[-1][0]
                possible_bands = {current_band}
                if x in intersections_by_x:
                    for _, band_i, band_j in intersections_by_x[x]:
                        if band_i == current_band:
                            possible_bands.add(band_j)
                        if band_j == current_band:
                            possible_bands.add(band_i)
                possible_segments = []
                for band_idx in possible_bands:
                    for seg in band_segments[band_idx]:
                        if seg[0] == x:
                            possible_segments.append((band_idx, seg))
                for band_idx, seg in possible_segments:
                    new_curve = curve + [(band_idx, seg)]
                    extend_curve(new_curve, seg[1])

            extend_curve(current_curve, current_x)
        print(f"Number of possible curves: {len(curves)}")
        if len(curves) > 20000000:
            print("Too many possible curves, skipping flatness calculation.")
            signal.alarm(0)
            return {"timeout": True}
        min_bandwidth = float('inf')
        flattest_curve = None
        best_curve_segments = None
        for curve in curves:
            energies = np.zeros(num_kpoints)
            for band_idx, (start, end, orig_idx, sel_idx) in curve:
                energies[start:end] = bands[band_idx, start:end]
            bandwidth = np.max(energies) - np.min(energies)
            if bandwidth < min_bandwidth:
                min_bandwidth = bandwidth
                flattest_curve = energies
                best_curve_segments = curve
        e_mid = (np.max(flattest_curve) + np.min(flattest_curve)) / 2 if flattest_curve is not None else None

        # Format best_curve_segments with original and selected indices
        formatted_best_curve_segments = [
            (f"Original Band {orig_idx + 1}", f"Selected Band {sel_idx}", (start, end))
            for band_idx, (start, end, orig_idx, sel_idx) in best_curve_segments
        ]
        signal.alarm(0)
        return {
            "number_of_potential_intersections": potential_intersections,
            "number_of_intersections_after_merging": intersections,
            "number_of_possible_curves": len(curves),
            "intersections": intersections,
            "band_segments": band_segments,
            "best_curve_segments": formatted_best_curve_segments,
            "flattest_curve_bandwidth": min_bandwidth,
            "flattest_curve_e_mid": e_mid,
            "flattest_curve_energies": flattest_curve.tolist() if flattest_curve is not None else None
        }
    except TimeoutError as e:
            print(f"⚠️ Processing stopped: {e}")
            # --- 3. CLEAN UP THE ALARM ---
            # The alarm must be reset in the exception block as well.
            signal.alarm(0)
            return {
            "timeout": True
        }
    except Exception as e:
        # Handle other non-timeout errors
        signal.alarm(0)
        print(f"An unexpected error occurred: {e}")
        return None

def s_bands(material_id, bs, omega_max=0.1):
    bs_label = {}
    try:
        num_bands = 6
        kp_original, eigenv, efermi, if_spin_polarized, original_indices = load_and_process_bandstructure(
            bs, num_bands=num_bands
        )
        spins = ['1', '-1']
        info = {}  
        print(f"Processing bands of {material_id}...")
        for spin_t in spins:
            if spin_t not in eigenv:
                print(f"Spin {spin_t} not found in bands_dict for {material_id}")
                continue
            bands = eigenv[spin_t]
            orig_indices = original_indices[spin_t]
            info[spin_t] = calculate_flatness_score(
                kp_original, bands, orig_indices, omega=omega_max, epsilon=0.010
            )
        
        if if_spin_polarized and info['1'] is not None and info['-1'] is not None:
            if info['-1'].get('timeout', False):
                if not info['1'].get('timeout', False):
                    band_info = info['1']
                    spin = '1'
                else:
                    band_info = info['-1']
                    spin = '-1'
            if info['1']["flattest_curve_bandwidth"] < info['-1']["flattest_curve_bandwidth"]:
                band_info = info['1']
                spin = '1'
            else:
                band_info = info['-1']
                spin = '-1'
        elif '1' in info and info['1'] is not None:
            band_info = info['1']
            spin = '1'
        elif '-1' in info and info['-1'] is not None:
            band_info = info['-1']
            spin = '-1'

        band_info["spin"] = spin
        band_info["efermi"] = efermi

        bs_label[material_id] = band_info
        print(f"Processed S_bands calculation: {material_id}")
        
    except Exception as e:
        print(f"Error processing bands of {material_id}: {str(e)}")
    return bs_label

def data_query_and_process(mp_api_key, sband_file, score_file, ids_file,
                           max_elms=10, min_elms=1, max_sites=20, data_fetching=False, restart=False):
    
    # 1. FETCH SUMMARY DATA
    if data_fetching:
        print("Fetching Material Summaries...")
        with MPRester(mp_api_key) as mpr:
            docs = mpr.materials.summary.search(
                num_elements=(min_elms, max_elms),
                num_sites=(None, max_sites),
                fields=['material_id', 'formation_energy_per_atom', 'band_gap', 'formula_pretty']
            )
        print(f"Number of materials fetched: {len(docs)}")
        data_summary = [doc.dict() for doc in docs]
        # shuffle data to avoid any ordering bias
        random.shuffle(data_summary)
        pd.DataFrame(data_summary).to_csv(ids_file, sep='\t', index=False)
        print(f"Summary saved to {ids_file}")
        del docs
    
    else:
        if os.path.exists(ids_file):
            #data_summary = pd.read_csv(ids_file, sep='\t').to_dict(orient='records')
            data_summary = pd.read_csv(ids_file).to_dict(orient='records')
            print(f"Loaded existing summary data from {ids_file}")
    
    # Extract IDs and clean up summary data from memory
    material_ids = [d['material_id'] for d in data_summary]
    del data_summary
    gc.collect()

    # 2. BATCH PROCESSING
    if restart:
        if os.path.exists(score_file): os.remove(score_file) # Or rename if you want to keep old runs
        if os.path.exists(sband_file): os.remove(sband_file) # Switched to .jsonl for efficiency
    else:
        existing_ids = set()
        if os.path.exists(score_file):
            with open(score_file, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    existing_ids.add(entry['material_id'])

        print(f"Resuming from previous run. {len(existing_ids)} materials processed.")

    print(f"Starting processing in batches of {BATCH_SIZE}...")
    
    for i in range(0, len(material_ids), BATCH_SIZE):
        batch_ids = material_ids[i : i + BATCH_SIZE]
        print(f"--- Processing Batch {i} to {i+len(batch_ids)} ---")
        
        with MPRester(mp_api_key) as mpr:
            for material_id in batch_ids:
                try:
                    if restart is False and material_id in existing_ids:
                        continue
                    else:
                        bs = mpr.get_bandstructure_by_material_id(material_id, line_mode=True)
                        bs_label = s_bands(material_id, bs, omega_max=0.1)
                        bs_data = bs_label.get(material_id, None)
                    dos = mpr.get_dos_by_material_id(material_id).as_dict()
                    if not bs_data:
                        print(f"Skipping {material_id}: No valid bands found.")
                        continue
                    if bs_data.get("timeout", False):
                        print(f"Skipping {material_id}: Band structure processing timed out.")
                        sband_entry = {'material_id': str(material_id), **bs_data}
                        score_data = {'material_id': str(material_id), 'timeout': True}
                        append_to_jsonl(sband_entry, sband_file) 
                        append_to_jsonl(score_data, score_file)
                        continue

                    S_bandwidth = calculate_s_bandwidth(bs_data, omega_max=0.3)
                    S_dos, peak_contrast = calculate_dos_and_scores(bs_data, dos, omega_max=0.3, alpha=0.1, delta=0.1, mu=0.1)
            
                    score_data = {
                        'material_id': str(material_id),
                        'S_bandwidth': S_bandwidth,
                        'S_DOS': S_dos,
                        'peak_contrast': peak_contrast,
                        'spin': bs_label.get(material_id, {}).get('spin'),
                    }
                    
                    print(f"Finished {material_id}")
                    sband_entry = {'material_id': str(material_id), **bs_data}
                    # 3. INCREMENTAL SAVING
                    append_to_jsonl(sband_entry, sband_file) 
                    append_to_jsonl(score_data, score_file)
                    

                except Exception as e:
                    print(f"Error processing {material_id}: {str(e)}")
                
                finally:
                    # 4. CLEANUP
                    if 'bs' in locals(): del bs
                    if 'dos' in locals(): del dos
                    if 'bs_label' in locals(): del bs_label
        
        gc.collect()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script to query Materials Project data and calculate S_total.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--restart', 
        action='store_true',
        default=False,
        help="Set this flag to True to restart processing (e.g., clear old files)."
    )

    parser.add_argument(
        '--batch_index', 
        type=int,
        default=1,
        help="Specify the batch index to process."
    )

    args = parser.parse_args()
    batch_index = args.batch_index
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "data"))
    mp_api_key = get_mp_api_key()
    sband_file = os.path.join(data_dir, "scores", f"bs_MP_{batch_index}.jsonl")
    score_file = os.path.join(data_dir, "scores", f"score_MP_{batch_index}.jsonl")
    ids_file = os.path.join(data_dir, "batches_2", f"batch_{batch_index}.csv")

    data_query_and_process(mp_api_key, sband_file, score_file, ids_file, restart=args.restart)
