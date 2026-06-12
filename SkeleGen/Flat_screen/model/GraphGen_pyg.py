import json
from torch.utils.data import Dataset
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from pymatgen.core.structure import Structure
import pandas as pd
from pymatgen.core.lattice import Lattice
from collections import defaultdict
from typing import List, Tuple, Sequence, Optional
from jarvis.core.atoms import Atoms
from jarvis.core.atoms import pmg_to_atoms
from jarvis.core.specie import chem_data, get_node_attributes

class Layers(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.bands = pd.read_json(root_dir)

    def __len__(self):
        return len(self.bands)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.bands.iloc[idx]

        return sample


def c_lattice_enlarge(structure, c_multi):
    lattice_matrix = np.array(structure.lattice.matrix)
    lattice_matrix[2] = c_multi * lattice_matrix[2]

    lattice_n = Lattice(lattice_matrix)

    cart_coords = [site.coords for site in structure.sites]
    frac_coords = [
        lattice_n.get_fractional_coords(cart_coord) for cart_coord in cart_coords
    ]
    structure_n = Structure(
        lattice_n,
        structure.species,
        frac_coords,
        structure.charge,
        coords_are_cartesian=False,
        labels=structure.labels,
    )
    return structure_n


def canonize_edge(
    src_id,
    dst_id,
    src_image,
    dst_image,
):

    if dst_id < src_id:
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image

    if not np.array_equal(src_image, (0, 0, 0)):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))

    assert src_image == (0, 0, 0)

    return src_id, dst_id, src_image, dst_image


def nearest_neighbor_edges(
    atoms=None,
    cutoff=8,
    max_neighbors=12,
    id=None,
    use_canonize=False,
):

    all_neighbors = atoms.get_all_neighbors(r=cutoff)

    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors)

    attempt = 0

    if min_nbrs < max_neighbors:
    
        lat = atoms.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            r_cut = max(lat.a, lat.b, lat.c)
        else:
            r_cut = 2 * cutoff
        attempt += 1

        return nearest_neighbor_edges(
            atoms=atoms,
            use_canonize=use_canonize,
            cutoff=r_cut,
            max_neighbors=max_neighbors,
            id=id,
        )

    edges = defaultdict(set)
    for site_idx, neighborlist in enumerate(all_neighbors):
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        max_dist = distances[max_neighbors - 1]

        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]

        for dst, image in zip(ids, images):
            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))

    return edges, images


def build_undirected_edgedata(
    atoms=None,
    edges={},
):

    u, v, r, all_images = [], [], [], []
    for (src_id, dst_id), images in edges.items():
        for dst_image in images:
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            d = atoms.lattice.cart_coords(dst_coord - atoms.frac_coords[src_id])

            for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
                u.append(uu)
                v.append(vv)
                r.append(dd)
                all_images.append(dst_image)
    u, v, r = (np.array(x) for x in (u, v, r))
    u = torch.tensor(u, dtype=torch.long)
    v = torch.tensor(v, dtype=torch.long)
    r = torch.tensor(r).type(torch.get_default_dtype())
    all_images = torch.tensor(all_images).type(torch.get_default_dtype())

    return u, v, r, all_images


def load_atom_attr(atomic_number, cgcnn_feature_json):
    key = str(atomic_number)
    with open(cgcnn_feature_json, "r") as f:
        i = json.load(f)
    try:
        return i[key]
    except:
        print(f"warning: could not load CGCNN features for {key}")
        print("Setting it to max atomic number available here, 100")
        return i["100"]

def compute_bond_cosines(edge_index, r):
    """
    Compute bond cosines for line graph edges.
    
    Args:
        edge_index: (2, E_lg) connectivity of line graph (bonds in original graph)
        r: (E_g, 3) bond vectors of original graph
        
    Returns:
        h: (E_lg, ) Cosine of angle between bonds
    """
    # edge_index_lg connects bond i to bond j
    # So source node in lg is bond i, target node in lg is bond j
    src_bond_idx = edge_index[0]
    dst_bond_idx = edge_index[1]
    
    r1 = -r[src_bond_idx] # Vector of first bond (pointing towards the shared atom)
    r2 = r[dst_bond_idx]  # Vector of second bond (pointing away from the shared atom)
    
    # Cosine similarity
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1) + 1e-6
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    
    return bond_cosine

class Graph(object):

    def __init__(
        self,
        nodes=[],
        node_attributes=[],
        edges=[],
        edge_attributes=[],
        color_map=None,
        labels=None,
    ):

        self.nodes = nodes
        self.node_attributes = node_attributes
        self.edges = edges
        self.edge_attributes = edge_attributes
        self.color_map = color_map
        self.labels = labels

    @staticmethod
    def atom_pyg_multigraph(
        atoms=None,
        neighbor_strategy="k-nearest",
        cutoff=8.0,
        max_neighbors=12,
        atom_features="cgcnn",
        max_attempts=3,
        id: Optional[str] = None,
        compute_line_graph: bool = True,
        use_canonize: bool = True,
        use_lattice_prop: bool = False,
        cutoff_extra=3.5,
        dtype="float32",
    ):

        if neighbor_strategy == "k-nearest":
            edges, images = nearest_neighbor_edges(
                atoms=atoms,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                id=id,
                use_canonize=use_canonize,
            )
            u, v, r, images = build_undirected_edgedata(atoms, edges)

        else:
            raise ValueError("Not implemented yet", neighbor_strategy)
        
        sps_features = []
        for ii, s in enumerate(atoms.elements):
            feat = list(get_node_attributes(s, atom_features=atom_features))
            sps_features.append(feat)
        sps_features = np.array(sps_features)
        node_features = torch.tensor(sps_features).type(torch.get_default_dtype())
        
        edge_index = torch.stack([u, v], dim=0) # (2, E)
        
        # Original Graph (g)
        g = Data(
            atom_features=node_features,
            edge_index=edge_index,
            r=r, # (E, 3) bond vectors
            images=images,
            num_nodes=len(atoms.elements)
        )
        
        vol = atoms.volume
        g.V = torch.tensor([vol for ii in range(atoms.num_atoms)])

        g.frac_coords = torch.tensor(atoms.frac_coords).type(
            torch.get_default_dtype()
        )
        
        if use_lattice_prop:
            lattice_prop = np.array(
                [atoms.lattice.lat_lengths(), atoms.lattice.lat_angles()]
            ).flatten()
            g.extra_features = torch.tensor(
                [lattice_prop for ii in range(atoms.num_atoms)]
            ).type(torch.get_default_dtype())


        if compute_line_graph:
            
            src, dst = edge_index
            num_nodes = g.num_nodes
            num_edges = g.num_edges
            
            sorted_idx = torch.argsort(src)
            src_sorted = src[sorted_idx]
            
            # Get counts of edges starting at each node
            node_degree_out = torch.bincount(src, minlength=num_nodes)
            
           
            lg_src = []
            lg_dst = []
            
            # Build adjacency list
            adj_list = [[] for _ in range(num_nodes)]
            for i, (s, d) in enumerate(zip(src.tolist(), dst.tolist())):
                adj_list[s].append(i) # Storing edge index starting at s
                
            # Iterate over all edges (u, v)
            for i, (s, d) in enumerate(zip(src.tolist(), dst.tolist())):
                # This edge 'i' ends at 'd'.
                # We want edges starting at 'd'.
                for next_edge_idx in adj_list[d]:
                    # Exclude self-loop in line graph if needed? 
                    # DGL standard line graph includes (u,v)->(v,w) for all w in adj[v].
                    # including w=u (backtracking).
                    if next_edge_idx != i: 
                        lg_src.append(i)
                        lg_dst.append(next_edge_idx)
                        
            lg_edge_index = torch.tensor([lg_src, lg_dst], dtype=torch.long)
            
            # Compute bond cosines
            h = compute_bond_cosines(lg_edge_index, r)
            
            lg = Data(
                edge_index=lg_edge_index,
                h=h,
                num_nodes=g.num_edges # Nodes in LG are edges in G
            )
            
            return g, lg
        else:
            return g


class main:
    def __init__(self, cgcnn_feature_json):
        self.cgcnn_feature_json = cgcnn_feature_json

    def get_neighbours_3d(self, structure_pymat, num_nbr):

        lattice = structure_pymat.lattice
        cart_coords = structure_pymat.cart_coords
        r0 = max(lattice.abc[0],lattice.abc[1],lattice.abc[2])

        structure_to_find_neighbours = structure_pymat
        nbr_list = structure_to_find_neighbours.get_all_neighbors(r0)

        while not all(len(nbr) >= num_nbr for nbr in nbr_list):
            r0 *= 2
            nbr_list = structure_to_find_neighbours.get_all_neighbors(r0)
        nbr_list = [
            sorted(nbr, key=lambda x: x.nn_distance)[:num_nbr] for nbr in nbr_list
        ]
        nbr_idx = [[nbr.index for nbr in neighbors] for neighbors in nbr_list]
        nbr_dis = [[nbr.nn_distance for nbr in neighbors] for neighbors in nbr_list]

        neigh_df = pd.DataFrame({"neigh_idx": nbr_idx, "ed_dis": nbr_dis})
        return neigh_df

    def node_attribute(self, structure_pymat):

        num_atom = len(structure_pymat.sites)
        attr = []
        for i in range(num_atom):
            atomic_number = structure_pymat.species[i].number
            atom_attr = load_atom_attr(atomic_number, self.cgcnn_feature_json)
            attr.append(atom_attr)
        return attr

    def edge_attribute(self, dis):

        return [dis]

    def get_graph_data(self, structure, num_nbr=6, line_graph=1, dtype=torch.float32):

        g, lg = Graph.atom_pyg_multigraph(
            atoms=pmg_to_atoms(structure),
            neighbor_strategy="k-nearest",
            cutoff=8.0,
            max_neighbors=num_nbr,
            atom_features="cgcnn",
            max_attempts=3,
            cutoff_extra=3.5,
            compute_line_graph=line_graph,
            dtype="float32",
        )
        return g, lg


def load_structure_from_json(json_line):
    data = json.loads(json_line)
    structure_data = data["structure"]

    lattice_data = structure_data["lattice"]["matrix"]
    species = [site["species"][0]["element"] for site in structure_data["sites"]]
    coords = [site["abc"] for site in structure_data["sites"]]

    structure = Structure(lattice_data, species, coords)
    return structure
