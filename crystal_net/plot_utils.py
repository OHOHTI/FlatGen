import numpy as np
import pandas as pd
import itertools
from mendeleev import element
from mp_api.client import MPRester
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from pymatgen.electronic_structure.plotter import BSPlotter, DosPlotter

def sphere_array(center=[0, 0, 0], radius=1, fineness=20):
    r"""Create an array used to plot wireframe spheres."""
    u = np.linspace(0, 2 * np.pi, fineness)
    v = np.linspace(0, np.pi, fineness)
    x = float(radius) * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = float(radius) * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = float(radius) * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    return x, y, z


def set_axes_equal(ax):
    r'''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.25*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_unit_cell(ax, lat_vecs, origin):
    r"""Plots the unit cell outline starting from origin with lattice vectors lat_vecs"""
    draw_mat = np.zeros([16, 3])
    draw_mat[0] = origin
    draw_mat[1] = draw_mat[0] + lat_vecs[0]
    draw_mat[2] = draw_mat[1] + lat_vecs[1]
    draw_mat[3] = draw_mat[2] - lat_vecs[0]
    draw_mat[4] = draw_mat[3] - lat_vecs[1]
    draw_mat[5] = draw_mat[4] + lat_vecs[2]
    draw_mat[6] = draw_mat[5] + lat_vecs[0]
    draw_mat[7] = draw_mat[6] - lat_vecs[2]
    draw_mat[8] = draw_mat[7] + lat_vecs[2]
    draw_mat[9] = draw_mat[8] + lat_vecs[1]
    draw_mat[10] = draw_mat[9] - lat_vecs[2]
    draw_mat[11] = draw_mat[10] + lat_vecs[2]
    draw_mat[12] = draw_mat[11] - lat_vecs[0]
    draw_mat[13] = draw_mat[12] - lat_vecs[2]
    draw_mat[14] = draw_mat[13] + lat_vecs[2]
    draw_mat[15] = draw_mat[14] - lat_vecs[1]
    ax.plot(draw_mat[:, 0], draw_mat[:, 1],
            draw_mat[:, 2], color='black', linewidth=0.25)


def plot_structure(param, structure, best_name, mat_ID, plot_save_path=None, specie=None,
                   plot_unit_cell_outline=True, num_cells_to_plot=[1, 1, 1], max_bond_dist=None,
                   fig=None):
    r"""
    Plots crystal structure in 3D.

    specie (str): Optional string that, when provided, selects just one sublattice to plot.
    plot_unit_cell_outline: True plots one unit cell. "all" plots all unit cells. False plots none.
    num_cells_to_plot (int list): A 3 element int list of the number of repeated unit cells to plot
        in the x, y, and z directions respectively.
    max_bond_dist (float): The longest bond length to plot. None if you don't want any bonds plotted
    """
    if not param.just_save_plots:
        plt.ion()
    if fig is None:
        fig = plt.figure()
    else:
        plt.figure(fig.number)
        plt.clf()
    ax = fig.add_subplot(111, projection='3d', position=[
                         0.05, 0.1, 0.75, 0.85])

    # get origin of each unit cell you will plot
    lat_vecs = structure.lattice.matrix
    norigins = num_cells_to_plot[0]*num_cells_to_plot[1]*num_cells_to_plot[2]
    origin_list = np.zeros([norigins, 3])
    orig_ind = 0
    for x_ind in range(num_cells_to_plot[0]):
        for y_ind in range(num_cells_to_plot[1]):
            for z_ind in range(num_cells_to_plot[2]):
                origin_list[orig_ind] = x_ind*lat_vecs[0] + \
                    y_ind*lat_vecs[1] + z_ind*lat_vecs[2]
                orig_ind = orig_ind + 1

    # for each unit cell that needs to be plot
    atom_list = []
    atom_list_coords = None
    for orig_ind in range(norigins):
        this_origin = origin_list[orig_ind]
        # plot unit cell outline once or for each
        if plot_unit_cell_outline is True and (this_origin == origin_list[0]).all():
            plot_unit_cell(ax, lat_vecs, this_origin)
        elif plot_unit_cell_outline == 'all':
            plot_unit_cell(ax, lat_vecs, this_origin)

        # plot atoms
        param.simple_graphics = True  # whether to plot atoms in 3D or not
        legend_elements = []
        elem_list = []
        plotted_sites_list = []
        for site_ind in range(len(structure.sites)):
            # only plot sites that are the desired specie, if given
            if (structure.sites[site_ind].species_string == specie) or (specie is None):
                # options are average_ionic_radius, atomic_radius, metallic_radius,
                # van_der_waals_radius, and atomic_radius_calculated
                rad = structure.sites[site_ind].specie.average_ionic_radius
                # only first atom in species listed in cases of fractional occupancy
                nm = list(structure.sites[site_ind].species.as_dict())[0]
                # color options are cpk_color, jmol_color, and molcas_gv_color
                colr = element(nm).jmol_color
                if nm not in elem_list:  # constructs custom element legend
                    elem_list.append(nm)
                    legend_elements.append(Line2D([0], [0], marker='o', color='w', label=nm,
                                                  markerfacecolor=colr, markersize=12*rad))

                abc = structure.sites[site_ind].frac_coords
                perm_a, perm_b, perm_c = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1],
                                                     indexing='xy')
                # all equivalent sites within +/- 1 of given site
                all_abc = np.tile(abc, (27, 1)) + np.stack((perm_a.flatten(), perm_b.flatten(),
                                                            perm_c.flatten()), axis=1)

                for trans_ind in range(27):
                    this_abc = all_abc[trans_ind, :]
                    if (this_abc >= -0.01).all() and (this_abc <= 1.01).all():
                        xyz = this_origin + (this_abc[0]*lat_vecs[0] + this_abc[1]*lat_vecs[1]
                                             + this_abc[2]*lat_vecs[2])
                        atom_list.append(site_ind)
                        if atom_list_coords is None:
                            atom_list_coords = np.array(xyz, ndmin=2)
                        else:
                            atom_list_coords = np.vstack(
                                (atom_list_coords, xyz))

                        if param.simple_graphics:
                            ax.scatter(xyz[0], xyz[1], xyz[2],
                                       c=colr, s=100*rad)
                        else:
                            [x, y, z] = sphere_array(
                                center=xyz, radius=rad, fineness=10)
                            ax.plot_surface(x, y, z, color=colr,
                                            linewidth=0, antialiased=False)

    # now plot bonds
    if max_bond_dist is not None:
        site_pairs = list(itertools.combinations(atom_list_coords, r=2))
        for site_pair in site_pairs:
            this_bond_dist = np.sqrt((site_pair[0][0] - site_pair[1][0])**2
                                     + (site_pair[0][1] - site_pair[1][1])**2
                                     + (site_pair[0][2] - site_pair[1][2])**2)
            if this_bond_dist <= max_bond_dist:
                ax.plot([site_pair[0][0], site_pair[1][0]],
                        [site_pair[0][1], site_pair[1][1]],
                        [site_pair[0][2], site_pair[1][2]],
                        color=colr, linewidth=2)

    # plot legend
    ax.legend(handles=legend_elements,
              loc='center left', bbox_to_anchor=(1, 0.5))

    # Setting the axes properties
    # ax.set_aspect("equal") # this is broken currently (08/11/20)
    set_axes_equal(ax)
    ax.axis("off")

    # ax.set_xlabel('X (Å)')
    # ax.set_ylabel('Y (Å)')
    # ax.set_zlabel('Z (Å)')
    ax.set_title(best_name + ' (' + mat_ID + ')')

    if param.just_save_plots:
        plt.ioff()
    else:
        plt.ion()
        plt.show()

    if param.save_results and plot_save_path is not None:
        fig.savefig(plot_save_path /
                    (f"{mat_ID}_{best_name}_crystal_struct.png"))

    return fig


def plotting_bs_stuff(param, mat_ID, print_str, save_path=None):
    r"""Function that can plot BS, DOS, and 1BZ (if available from the Materials Project)."""
    bs = None
    dos = None
    with MPRester(param.MP_KEY, notify_db_version=False) as m:
        if param.plot_BS or param.plot_DOS_and_BS or param.plot_BZ:
            bs = m.get_bandstructure_by_material_id(mat_ID)
        if param.plot_DOS or param.plot_DOS_and_BS:
            dos = m.get_dos_by_material_id(mat_ID)

    # plots BS, if requested
    if (bs is not None) and param.plot_BS:
        # is the material a metal (i.e., the fermi level cross a band)
        if bs.is_metal():
            print_str = f"{print_str} - Should be a metal...\n"
        else:
            print_str = f"{print_str} - Probably non-metallic...\n"
            print_str = f"{print_str} - Bandgap: {bs.get_band_gap()['energy']} eV\n"

        # plot band structure and BZ
        plotter = BSPlotter(bs)
        plotter.get_plot()
        if save_path is not None:
            plotter.save_plot(
                save_path / (f"{mat_ID}_band_structure.png"))
        # plot BZ, if requested
        if param.plot_BZ:
            plotter.plot_brillouin()
            if save_path is not None:
                plotter.save_plot(
                    save_path / (f"{mat_ID}_bz.png"))

    # plots DOS, if requested
    if (dos is not None) and param.plot_DOS:
        plotterDOS = DosPlotter()
        plotterDOS.add_dos("Total DOS", dos)
        plotterDOS.get_plot()
        if save_path is not None:
            plotterDOS.save_plot(
                save_path / (f"{mat_ID}_dos.png"))

    # plot DOS and BS combined, if requested
    if (dos is not None and bs is not None) and param.plot_DOS_and_BS:
        # bs_projection can be 'elements' or None
        # dos_projection can be 'elements', 'orbitals' or None
        plotterBSDOS = plotterBSDOSter(
            bs_projection=None, dos_projection='orbitals')
        plotterBSDOS.get_plot(bs, dos=dos)
        if save_path is not None:
            plotterBSDOS.save_plot(
                save_path / (f"{mat_ID}_bsdos.png"))
    return print_str