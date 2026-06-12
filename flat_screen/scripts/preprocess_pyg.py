import os
import json
from GraphGen_pyg import main as GraphGen
import pickle
import numpy as np
import torch
from pymatgen.core.structure import Structure
import pandas as pd
from mp_api.client import MPRester

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))


def get_mp_api_key():
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError("Set MP_API_KEY before running preprocess_pyg.py.")
    return api_key


def fetch_all_materials(ids_file, batch_size=100, api_key=None):
    """Fetch structures and summary info from MP in batches (single pass)."""
    material_ids_df = pd.read_csv(ids_file, sep='\t')
    material_ids = material_ids_df['Key'].tolist()

    structures = {}
    text_info = {}

    with MPRester(api_key or get_mp_api_key()) as mpr:
        for i in range(0, len(material_ids), batch_size):
            batch_ids = material_ids[i:i + batch_size]
            try:
                docs = mpr.materials.summary.search(
                    material_ids=batch_ids,
                    fields=["material_id", "formula_pretty", "formula_anonymous",
                            "symmetry", "elements", "volume", "structure"]
                )

                for doc in docs:
                    item = doc.dict()
                    mp_id = str(item.get("material_id"))

                    # Reconstruct pymatgen Structure for graph generation
                    struct_dict = item.get("structure", {})
                    try:
                        structures[mp_id] = Structure.from_dict(struct_dict)
                    except Exception as e:
                        print(f"Could not parse structure for {mp_id}: {e}")
                        continue

                    # Build text string
                    lattice_params = {
                        'a': struct_dict.get('lattice', {}).get('a'),
                        'b': struct_dict.get('lattice', {}).get('b'),
                        'c': struct_dict.get('lattice', {}).get('c'),
                        'beta': struct_dict.get('lattice', {}).get('beta'),
                        'alpha': struct_dict.get('lattice', {}).get('alpha'),
                        'gamma': struct_dict.get('lattice', {}).get('gamma'),
                    }
                    sym_info = item.get("symmetry", {})
                    info_str = (
                        f"formula_anonymous: {item.get('formula_anonymous')}, "
                        f"formula_pretty: {item.get('formula_pretty')}, "
                        f"sg_symbol: {sym_info.get('symbol')}, "
                        f"point_group: {sym_info.get('point_group')}, "
                        f"crystal_system: {sym_info.get('crystal_system')}, "
                        f"lattice_mat: {struct_dict.get('lattice', {}).get('matrix')}, "
                        f"lattice_params: {lattice_params}, "
                        f"species: {item.get('elements')}, "
                        f"lattice_volume: {item.get('volume')}"
                    )
                    text_info[mp_id] = info_str

                print(f"Fetched batch {i}-{i+len(batch_ids)} "
                      f"({len(structures)} structures so far)")
            except Exception as e:
                print(f"Error processing batch {i} to {i+batch_size}: {e}")

    return structures, text_info


def build_graphs(structures, graph_gen):
    """Generate graph + line graph for each structure."""
    graph_dict = {}
    for idx, (mp_id, structure) in enumerate(structures.items()):
        try:
            g, lg = graph_gen.get_graph_data(structure)
            graph_dict[mp_id] = (g, lg)
        except KeyError as e:
            print(f"Error for {mp_id}: Missing key {e}. Skipping.")
        except ValueError as ve:
            print(f"Error for {mp_id}: {ve}. Skipping.")
        except Exception as e:
            print(f"Unexpected error for {mp_id}: {e}. Skipping.")
        if (idx + 1) % 500 == 0:
            print(f"  Generated graphs for {idx + 1}/{len(structures)} materials")
    return graph_dict


if __name__ == "__main__":
    output_file = os.path.join(DATA_DIR, "graph_text_mp_pyg.pkl")
    ids_file = os.path.join(DATA_DIR, "scores_with_stotal_cleaned.txt")
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Fetching materials from Materials Project...")
    structures, text_info = fetch_all_materials(ids_file, api_key=get_mp_api_key())
    print(f"Fetched {len(structures)} structures, {len(text_info)} text entries")

    print("Generating graphs...")
    graph_gen = GraphGen(None)
    graph_dict = build_graphs(structures, graph_gen)
    print(f"Generated {len(graph_dict)} graphs")

    merged_dict = {}
    for key, graph in graph_dict.items():
        merged_dict[key] = {
            'structure_graph': graph,
            'text_string': text_info.get(key, None),
        }

    print(f"Total merged entries: {len(merged_dict)}")

    with open(output_file, 'wb') as f:
        pickle.dump(merged_dict, f)

    print(f"Dictionary successfully saved to {output_file}.")
