import os
import numpy as np
from pythtb import TBModel, Lattice
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from pymatgen.symmetry.kpath import KPathSetyawanCurtarolo, KPathLatimerMunro
from pymatgen.core import IStructure
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

from io_utils import make_cgd_string

def is_cyclic_hopping(hop_array, start_site=0, numCycles=5, verbose=False):
    r"""
    Checks if this hopping array is trivially localized.

    Trivially localized means that the hopping cannot extent to infinity,
    in other words, whether it is possible to take a path along hoppings out
    to arbitrary distances.

    The condition that is checked is actually whether there is any hopping path
    of numCycles hops or fewer that will take you from the start_site in the
    first unit cell to the same start_site in any other unit cell.

    This is not rigorously flawless, in the case of many sites in the first
    unit cell or pathological cases, but it is a compromise between ease of
    coding and perfection. Note that this code is not currently very efficient,
    but it is beyond my capabilities to make it more efficient.

    hop_array is an Mx5 matrix that describes the allowed hoppings, where M is
    the number of allowed hoppings. The first and second columns identify the
    two sites involved, and the last three columns describe the change in unit
    cell. The order of rows does not matter.
    For example, a hopping that connects atom 3 in one cell to atom 7 in
    the cell that is one forward in x and one back in y, but on the same z
    would have a row in hop_array that looks like
    (3 7 1 -1 0)
    """

    # initialize values
    connected_sites = np.array([start_site, 0, 0, 0], ndmin=2)
    if verbose:
        print("\nHopping list (site 1, site 2, dx, dy, dz):")
        print(hop_array)
        print(f"\nStarting site: {start_site}")
        print("\nInitially connected sites:")
        print(connected_sites)
    num_hops = np.shape(hop_array)[0]
    is_cyclic = False
    keep_going = True
    loop_num = 0

    # iterate hopping out from start_site
    while keep_going:
        if verbose:
            print(f"\nStarting cycle {loop_num+1} of {numCycles}")

        # hop out from each site included so far
        new_connected_sites = connected_sites
        for site_ind in range(np.shape(connected_sites)[0]):
            this_site = connected_sites[site_ind, 0]
            for hop_ind in range(num_hops):
                if hop_array[hop_ind, 0] == this_site:
                    dx = connected_sites[site_ind, 1] + hop_array[hop_ind, 2]
                    dy = connected_sites[site_ind, 2] + hop_array[hop_ind, 3]
                    dz = connected_sites[site_ind, 3] + hop_array[hop_ind, 4]
                    new_site = np.array(
                        [hop_array[hop_ind, 1], dx, dy, dz], ndmin=2)
                    new_connected_sites = np.append(
                        new_connected_sites, new_site, axis=0)
                if hop_array[hop_ind, 1] == this_site:
                    dx = connected_sites[site_ind, 1] - hop_array[hop_ind, 2]
                    dy = connected_sites[site_ind, 2] - hop_array[hop_ind, 3]
                    dz = connected_sites[site_ind, 3] - hop_array[hop_ind, 4]
                    new_site = np.array(
                        [hop_array[hop_ind, 0],  dx, dy, dz], ndmin=2)
                    new_connected_sites = np.append(
                        new_connected_sites, new_site, axis=0)

        # remove duplicate sites
        new_connected_sites = np.unique(new_connected_sites, axis=0)
        connected_sites = new_connected_sites
        if verbose:
            print("\nCurrently connected sites:")
            print(connected_sites)

        # check if any site is visited in more than one unit cell
        _, site_counts = np.unique(connected_sites[:, 0], return_counts=True)
        if np.any(site_counts > 1):
            is_cyclic = True
            if verbose:
                print("Found return to a site in a different unit cell, ending")

        # check whether to keep hopping out
        loop_num = loop_num + 1
        if is_cyclic or loop_num >= numCycles:
            keep_going = False

    return is_cyclic



def get_NN_dist(structure, specie=None):
    r"""
    Gets the nearest neighbor distance in a structure, in angstroms.

    If optional input "specie" is supplied (must be an element name string, eg. "Fe"), then the
    nearest neighbor distance on that sublattice is returned. Otherwise the smallest distance
    overall is returned.
    """
    lat_vecs = structure.lattice.matrix
    dist_mat = np.round(structure.distance_matrix, 5)

    # finds the sites that are of this specie
    sublat_ind_to_lat_ind = []
    for site_ind in range(len(structure.sites)):
        if structure.sites[site_ind].species_string == specie:
            sublat_ind_to_lat_ind.append(site_ind)

    # get nearest neighbor distance on this sublattice
    if specie is not None:
        this_dist_mat = dist_mat[sublat_ind_to_lat_ind, :]
        this_dist_mat = this_dist_mat[:, sublat_ind_to_lat_ind]
    else:
        this_dist_mat = dist_mat
    nn_dist = np.sort(np.unique(this_dist_mat))[1]

    for lat_vec_ind in range(3):
        if np.linalg.norm(lat_vecs[lat_vec_ind]) < nn_dist:
            nn_dist = np.linalg.norm(lat_vecs[lat_vec_ind])

    return nn_dist


# %% plot TB, if requested
def do_TB_model(param, structure, print_str, write_str, this_save_path, best_name, mat_ID):
    # define lattice vectors
    lat_vecs = structure.lattice.matrix

    # get list of unique site types (by element composition)
    specie_list = set()
    for site_ind in range(len(structure.sites)):
        specie_list.add(structure.sites[site_ind].species_string)
    specie_list = list(specie_list)
    n_species = len(specie_list)
    print_str = f"{print_str} - Detected {n_species} unique site types\n"

    # do TB for each sublattice
    dist_mat = np.round(structure.distance_matrix, 5)
    has_FBs = False
    gras = []
    n_FBs_list = []
    fullhop_arrays = []
    sublat_graphs = []
    k_vec_list = []
    evals_list = []
    comps_with_FBs_list = []
    for specie_ind in range(n_species):
        print_str = print_str + '\n'

        # create list of site fractional coordinates in this sublattice
        orb = None
        sublat_ind_to_lat_ind = []
        lat_ind_to_sublat_ind = [0]*len(structure.sites)
        for site_ind in range(len(structure.sites)):
            if structure.sites[site_ind].species_string == specie_list[specie_ind]:
                if orb is None:
                    orb = structure.sites[site_ind].frac_coords
                    lat_ind_to_sublat_ind[site_ind] = 0
                else:
                    orb = np.vstack(
                        (orb, structure.sites[site_ind].frac_coords))
                    lat_ind_to_sublat_ind[site_ind] = np.shape(orb)[0]-1
                sublat_ind_to_lat_ind.append(site_ind)

        n_this_species = len(sublat_ind_to_lat_ind)
        print_str = (f"{print_str} - Now considering the {specie_list[specie_ind]} sublattice"
                     f" (containing {n_this_species} sites)\n")

        # don't do TB for just a single atom sublattice
        if n_this_species > 1:
            # get nearest neighbor distance on this sublattice
            this_dist_mat = dist_mat[sublat_ind_to_lat_ind, :]
            this_dist_mat = this_dist_mat[:, sublat_ind_to_lat_ind]
            nn_dist = np.sort(np.unique(this_dist_mat))[1]

            for lat_vec_ind in range(3):
                if np.linalg.norm(lat_vecs[lat_vec_ind]) < nn_dist:
                    nn_dist = np.linalg.norm(lat_vecs[lat_vec_ind])

            print_str = f"{print_str}   - Nearest-neighbor distance is {np.round(nn_dist, 2)}\n"

            # decide on param.max_dist and param.decay_rate for this sublattice:
            
            if param.dist_units == 'AA':
                this_max_dist = param.max_dist
                # rate at which hopping decays in units of angstroms
                this_decay_rate = param.decay_rate
            else:
                this_max_dist = param.max_dist * nn_dist
                if param.decay_rate:
                    this_decay_rate = param.decay_rate * nn_dist
                else:
                    this_decay_rate = param.decay_rate

            if param.decay_rate:
                this_amplitude = np.exp(- nn_dist / (this_decay_rate))
            else:
                this_amplitude = 1.0
            # find the smallest neighbor distance in this sublattice NOT considered:
            # checks out to the longest bond distance allowed plus the shortest of
            # the three lattice constants a, b, c (times 1.05 for tolerance)
            diag_dist = 1.05 * np.amin([np.linalg.norm(lat_vecs[0]),
                                        np.linalg.norm(lat_vecs[1]), np.linalg.norm(lat_vecs[2])])
            NNN_neighs = structure.get_all_neighbors(
                r=this_max_dist+diag_dist, numerical_tol=1e-8,
                sites=[structure.sites[i] for i in sublat_ind_to_lat_ind])

            nnn_dist = this_max_dist + diag_dist + 1
            nn_dist_max = nn_dist
            for sublat_ind_i in range(n_this_species):
                lat_ind_i = sublat_ind_to_lat_ind[sublat_ind_i]
                if len(NNN_neighs[sublat_ind_i]) != 0:
                    for neigh_ind in range(len(NNN_neighs[sublat_ind_i])):
                        lat_ind_j = NNN_neighs[sublat_ind_i][neigh_ind].index
                        sublat_ind_j = lat_ind_to_sublat_ind[lat_ind_j]
                        if (lat_ind_j in sublat_ind_to_lat_ind):  # and (lat_ind_i <= lat_ind_j)
                            ind_R = NNN_neighs[sublat_ind_i][neigh_ind].image
                            this_dist = structure.get_distance(
                                lat_ind_i, lat_ind_j, ind_R)
                            if this_dist > nn_dist_max and this_dist < (this_max_dist + 1e-8):
                                nn_dist_max = this_dist
                            if this_dist > (this_max_dist + 1e-8) and this_dist < nnn_dist:
                                nnn_dist = this_dist

            print_str = f"{print_str}   - First non-considered neighbor distance is {np.round(nnn_dist, 2)}\n"

            # print near-neighbor distances being used, if requested
            if param.print_NN_dists:
                neighs = structure.get_all_neighbors(
                    r=this_max_dist, numerical_tol=1e-8,
                    sites=[structure.sites[i] for i in sublat_ind_to_lat_ind])
                dist_list = []
                for sublat_ind_i in range(n_this_species):
                    lat_ind_i = sublat_ind_to_lat_ind[sublat_ind_i]
                    if False:
                        print(f"\nthis site sublattice ind: {sublat_ind_i}")
                        print(f"this site lattice ind: {lat_ind_i}\n")
                    if len(neighs[sublat_ind_i]) != 0:
                        for neigh_ind in range(len(neighs[sublat_ind_i])):
                            lat_ind_j = neighs[sublat_ind_i][neigh_ind].index
                            sublat_ind_j = lat_ind_to_sublat_ind[lat_ind_j]
                            if ((lat_ind_i <= lat_ind_j) and
                                    (lat_ind_j in sublat_ind_to_lat_ind)):
                                ind_R = neighs[sublat_ind_i][neigh_ind].image
                                this_dist = structure.get_distance(
                                    lat_ind_i, lat_ind_j, ind_R)
                                dist_list.append(this_dist)
                                if False:
                                    print(
                                        repr(neighs[sublat_ind_i][neigh_ind]))
                                    print(f"this neighbor ind: {neigh_ind}")
                                    print(
                                        f"this neighbor lattice ind: {lat_ind_j}")
                                    print(
                                        f"this neighbor sublattice ind: {sublat_ind_j}")
                                    print(f"this neighbor ind_R: {ind_R}")
                                    print(f"this neighbor dist: {this_dist}\n")
                                    input()

                dist_list = np.round(dist_list, 2)
                dist_list, dist_counts = np.unique(
                    dist_list, return_counts=True)
                for dist_ind in range(len(dist_list)):
                    print_str = (f"{print_str}   - {dist_counts[dist_ind]} neighbors at "
                                 f"distance {dist_list[dist_ind]}\n")

            # create TB model object
            lat = Lattice(lat_vecs=lat_vecs, orb_vecs=orb, periodic_dirs="all")
            gra = TBModel(lattice=lat, spinful=False)

            # create hopping parameters based on exponential distance
            neighs = structure.get_all_neighbors(
                r=this_max_dist, numerical_tol=1e-8,
                sites=[structure.sites[i] for i in sublat_ind_to_lat_ind])
            # graph of atom connections
            sublat_graph = np.zeros((n_this_species, n_this_species))
            fullhop_array = None  # list of all hoppings with cell-border-crossing information
            # whether atom has a hop out of the unit cell
            leaves_cell = [False] * n_this_species
            for sublat_ind_i in range(n_this_species):
                lat_ind_i = sublat_ind_to_lat_ind[sublat_ind_i]
                if len(neighs[sublat_ind_i]) != 0:
                    for neigh_ind in range(len(neighs[sublat_ind_i])):
                        lat_ind_j = neighs[sublat_ind_i][neigh_ind].index
                        sublat_ind_j = lat_ind_to_sublat_ind[lat_ind_j]
                        if (lat_ind_j in sublat_ind_to_lat_ind):  # and (lat_ind_i <= lat_ind_j)
                            ind_R = neighs[sublat_ind_i][neigh_ind].image
                            is_first = True  # bool for not including hopping pairs
                            if lat_ind_i > lat_ind_j:
                                is_first = False
                            if lat_ind_i == lat_ind_j:
                                if ind_R[0] < 0:
                                    is_first = False
                                if ind_R[0] == 0 and ind_R[1] < 0:
                                    is_first = False
                                if ind_R[0] == 0 and ind_R[1] == 0 and ind_R[2] < 0:
                                    is_first = False
                            if is_first:
                                this_dist = structure.get_distance(
                                    lat_ind_i, lat_ind_j, ind_R)
                                # print(str(sublat_ind_i)+', '+str(sublat_ind_j)+', '+str(ind_R))
                                if this_decay_rate:
                                    gra.set_hop(-this_amplitude * np.exp(-this_dist/(this_decay_rate)),
                                                sublat_ind_i, sublat_ind_j, ind_R, 'add')
                                else:
                                    gra.set_hop(-this_amplitude,
                                                sublat_ind_i, sublat_ind_j, ind_R, 'add')

                                # constructs scipy sparse graph of atom connections in sublattice
                                sublat_graph[sublat_ind_i][sublat_ind_j] = 1
                                sublat_graph[sublat_ind_j][sublat_ind_i] = 1
                                if (ind_R[0]**2 + ind_R[1]**2 + ind_R[2]**2) > 0.001:
                                    leaves_cell[sublat_ind_i] = True
                                    leaves_cell[sublat_ind_j] = True

                                # builds list of hops
                                if fullhop_array is None:
                                    fullhop_array = np.array(
                                        [sublat_ind_i, sublat_ind_j,
                                            ind_R[0], ind_R[1], ind_R[2]],
                                        ndmin=2, dtype=int)
                                else:
                                    fullhop_array = np.append(
                                        fullhop_array,
                                        [[sublat_ind_i, sublat_ind_j, ind_R[0], ind_R[1],
                                          ind_R[2]]],
                                        axis=0)
            fullhop_arrays.append(fullhop_array.tolist())

            # counts number of atoms in components that do not exit unit cell
            sublat_graph = csr_matrix(sublat_graph)
            sublat_graphs.append((sublat_graph.toarray()).tolist())
            n_components, labels = connected_components(
                csgraph=sublat_graph, directed=False, return_labels=True)
            comps_leave_cell = np.unique(labels[leaves_cell])
            comps_no_leave_cell = np.setdiff1d(labels, comps_leave_cell)
            atoms_with_no_hops = 0
            for i_comp in range(len(comps_no_leave_cell)):
                atoms_with_no_hops = (atoms_with_no_hops
                                      + np.count_nonzero(labels == comps_no_leave_cell[i_comp]))
            # double checks whether the components that *do* leave the cell are *really*
            # not trivially localized
            fullhop_array = fullhop_array.astype(np.int32)  # makes all ints
            comps_list = []
            for i_comp in range(len(comps_leave_cell)):
                # gets sites in this component
                in_this_comp = np.flatnonzero(
                    labels == comps_leave_cell[i_comp])
                # finds which lines of the hopping array are from this component
                temp = np.isin(fullhop_array, in_this_comp)
                mask = np.logical_or(temp[:, 0], temp[:, 1])
                # filters to just hopping in this component
                thishop_array = fullhop_array[mask, :]
                if not is_cyclic_hopping(thishop_array, start_site=in_this_comp[0],
                                         numCycles=param.check_to_N_hops):
                    atoms_with_no_hops = atoms_with_no_hops + len(in_this_comp)
                else:
                    comps_list.append(in_this_comp.tolist())
            # reports number of trivially localized sites
            if atoms_with_no_hops != 0:
                print_str = (f"{print_str}   - {atoms_with_no_hops} sites with only localized "
                             f"hopping\n")

            # solve model on a high symm. path in k-space
            if mat_ID == "mp-773163":
                kpath = KPathLatimerMunro(structure)
            else:
                kpath = KPathSetyawanCurtarolo(structure)
            # kpath._kpath["kpoints"]
            # kpath = HighSymmKpath(structure)
            k_vec = kpath.get_kpoints(100, False)[0]
            k_lab = kpath.get_kpoints(100, False)[1]
            # (k_vec,k_dist,k_node)=gra.k_path(k, 100)
            evals = gra.solve_ham(k_vec)
            # text = gra.info(show=True, short=True)
            evals = np.transpose(evals)
            evals_list.append(evals.tolist())
            save_k_vec = k_vec
            for vec_ind in range(len(k_vec)):
                save_k_vec[vec_ind] = k_vec[vec_ind].tolist()
            k_vec_list.append(save_k_vec)
            n_pts = len(k_vec)

            # create figure
            if param.plot_TB:
                plt.rcParams["font.family"] = "serif"
                fig = plt.figure(constrained_layout=True)
                gs = GridSpec(1, 4, figure=fig)

                # get DOS, if requested
                if param.plot_TBDOS:
                    grid_spacing = [16, 16, 16]
                    k_point_mesh = gra.k_uniform_mesh(grid_spacing)

                    energy_array = gra.solve_ham(k_point_mesh)
                    n_k_pts = grid_spacing[0] * \
                        grid_spacing[1] * grid_spacing[2]
                    Ehist, Ebins = np.histogram(energy_array, bins=100)

                    # plot results
                    axDOS = fig.add_subplot(gs[:, -1])
                    Evals = (Ebins[:-1] + Ebins[1:]) / 2
                    axDOS.fill_betweenx(
                        Evals, 0, Ehist, edgecolor='k', facecolor='#929591')
                    axDOS.set_xlabel('DOS')
                    Erange = np.ptp(Evals)
                    axDOS.set_ylim(np.min(Evals) - Erange * 0.03,
                                   np.max(Evals) + Erange * 0.03)
                    axDOS.set_xlim(0, np.max(Ehist))

                # plot bandstructure
                if param.plot_TBDOS:
                    ax = fig.add_subplot(gs[:, :-1], sharey=axDOS)
                else:
                    ax = fig.add_subplot(gs[:, :])
                for plot_ind in range(np.shape(evals)[0]):
                    ax.plot(np.arange(0, n_pts), evals[plot_ind, :])
                ax.set_xlabel('Momentum')
                ax.set_ylabel('E (arb. units)')
                fig.suptitle(f"T.B. for {specie_list[specie_ind]} sublattice of {best_name} "
                             f"({mat_ID})")
                # fig.set_tight_layout(True)

                # label x-axis
                tick_list = []
                tick_labels = []
                for tick_ind in range(n_pts):
                    if '\\' in k_lab[tick_ind]:
                        # make latex friendly
                        k_lab[tick_ind] = f"${k_lab[tick_ind]}$"
                for tick_ind in range(n_pts):
                    if bool(k_lab[tick_ind]):
                        if (tick_ind == 0) or (tick_ind == n_pts - 1):
                            tick_list.append(tick_ind)
                            tick_labels.append(k_lab[tick_ind])
                        elif bool(k_lab[tick_ind - 1]):
                            tick_list.append((2 * tick_ind - 1) / 2)
                            if k_lab[tick_ind] == k_lab[tick_ind - 1]:
                                tick_labels.append(k_lab[tick_ind])
                            else:
                                tick_labels.append(
                                    f"{k_lab[tick_ind - 1]}|{k_lab[tick_ind]}")
                ax.set_xticks(tick_list)
                ax.set_xticklabels(tick_labels)
                ax.set_xlim(tick_list[0], tick_list[-1])
                for tick_ind in range(len(tick_list)):
                    plt.axvline(x=tick_list[tick_ind], linewidth=1, color='k')

            # display TB model parameters, if requested
            if param.print_TB_params and param.verbose:
                gra.display()

            # check for flat band in TB, if requested
            n_FBs = 0
            comps_with_FBs = []
            fb_str = ""
            net_dim_str = ""
            if param.check_TB_flat:
                for comp_ind in range(len(comps_list)):
                    # remove all except for this component
                    rmv_list = []
                    for rmv_ind in range(n_this_species):
                        if rmv_ind not in comps_list[comp_ind]:
                            rmv_list.append(rmv_ind)
                    if len(rmv_list) != 0:
                        this_gra = gra.remove_orb(rmv_list)
                    else:
                        this_gra = gra.copy()
                    # text = this_gra.info(show=True, short=False)
                    if param.high_symm_only:
                        energy_array = this_gra.solve_ham(k_vec)
                        n_k_pts = n_pts
                    else:
                        grid_spacing = [16, 16, 16]
                        n_k_pts = grid_spacing[0] * \
                            grid_spacing[1] * grid_spacing[2]
                        k_point_mesh = this_gra.k_uniform_mesh(grid_spacing)
                        # k_point_mesh = np.random.rand(n_k_pts, 3)
                        energy_array = this_gra.solve_ham(k_point_mesh)
                    # determine number of nontrivial flat bands:
                    energy_array = np.transpose(np.round(energy_array, 12))
                    hist, bins = np.histogram(
                        energy_array, bins=int(np.ceil(1 / param.max_fb_width)))
                    if param.full_flat:
                        check_inds = np.nonzero(
                            np.floor(hist / (n_k_pts * param.flatness_tol)))[0]
                        new_FBs = 0
                        if len(check_inds) > 0:
                            for check_ind in check_inds:
                                new_FBs = new_FBs + np.amin(np.count_nonzero(
                                    np.logical_and(energy_array >= bins[check_ind],
                                                   energy_array <= bins[check_ind+1]), axis=0))
                    else:
                        new_FBs = int(
                            np.sum(np.floor(hist / (n_k_pts * param.flatness_tol))))
                    n_FBs = n_FBs + new_FBs
                    print(f"Component {comp_ind} has {n_FBs} flat bands")
                    if new_FBs > 0:
                        comps_with_FBs.append(
                            [i for i in comps_list[comp_ind]])
                        fb_str = fb_str + \
                            f"{new_FBs}/{len(comps_list[comp_ind])} "

                        # create systre input file, run systre, get key from systre
                        if param.save_cgd:
                            component_num = len(comps_with_FBs)-1
                            cgd_name = f"{mat_ID}_{specie_list[specie_ind]}_{component_num}.cgd"
                            with open(this_save_path / cgd_name, 'w') as f_cgd:
                                write_cgd = make_cgd_string(mat_ID,
                                                            specie_list[specie_ind],
                                                            component_num,
                                                            fullhop_array,
                                                            comps_with_FBs[component_num])
                                f_cgd.write(write_cgd)
                            out_name = f"{mat_ID}_{specie_list[specie_ind]}_{component_num}.out"
                            systre_path = param.script_path/"Systre-exp-fix2.jar"
                            systre_cmd = ("timeout 600 "
                                          + f"java -cp {systre_path} "
                                          + f"org.gavrog.apps.systre.SystreCmdline "
                                          + f"{this_save_path/cgd_name} --systreKey"
                                          + f"> {this_save_path/out_name}")
                            try:
                                os.system(systre_cmd)
                            except Exception as e:
                                print(e)

                n_FBs = int(n_FBs)
                comps_with_FBs_list.append(comps_with_FBs)
                print_str = (f"{print_str}   - Contains {n_FBs} (potentially) interesting "
                             f"flat bands\n")

            if param.just_save_plots:
                plt.ion()
                plt.ioff()
            if param.save_results and param.plot_TB:
                fig.savefig(this_save_path / (f"{n_FBs}_FBs_{mat_ID}_{best_name}_"
                                              f"{specie_list[specie_ind]}_sublat_TB.png"))

            n_FBs_list.append(n_FBs)
            write_str = (f"{write_str}{mat_ID}, {best_name}, {specie_list[specie_ind]}, "
                         f"{n_FBs}, {n_this_species}, {np.round(nn_dist, 5)}, {np.round(nn_dist_max, 5)}, "
                         f"{np.round(nnn_dist, 5)}, {fb_str.strip()}\n")
            gras.append(gra)

        else:
            print_str = f"{print_str}   - Only contains one atom per unit cell, skipping...\n"
            gras.append(None)
            fullhop_arrays.append(None)
            sublat_graphs.append(None)
            evals_list.append(None)
            k_vec_list.append(None)
            n_FBs_list.append(0)
            comps_with_FBs_list.append([])
            write_str = (f"{write_str}{mat_ID}, {best_name}, {specie_list[specie_ind]}, "
                         f"0, 1, 0, 0, 0, na\n")

    return (print_str, write_str, gras, n_FBs_list, specie_list, fullhop_arrays, sublat_graphs,
            k_vec_list, evals_list, comps_with_FBs_list)