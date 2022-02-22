#!/usr/bin/env python3

""" GBS to OpenVDB converter

"OpenVDB is an Academy Award-winning open-source C++ library comprising a novel
hierarchical data structure and a suite of tools for the efficient storage and
manipulation of sparse volumetric data discretized on three-dimensional grids.
It was developed by DreamWorks Animation for use in volumetric applications
typically encountered in feature film production and is now maintained by the
Academy Software Foundation (ASWF)." https://www.openvdb.org/

To enable volumetric vizualisations of GBS simulations in the context of
scientific rendering for communication purposes, we propose a converter tool to
save GBS fields to the OpenVDB format.
"""


# FIXME Currently, when GBS data is loaded, in the z-direction (toroidal), one
# considers iz=0 as phi=-pi and iz=last as phi=pi, which is inaccurate. One
# should rather have phi(izlast+1)=pi for periodicity. This could be fixed both
# in grid generation and when loading data (probably at a minor cost in
# performance).


import pyopenvdb as vdb
import os
import re
import logging
import argparse
import traceback
import numpy as np
import scipy.interpolate as spinterp
import gbspy


def extract_gbs_dimensions(sim, r_0=2.96, allow_resize=False):
    """ Guess simulation dimensions

    Extract the cylindrical domain's dimensions from the simulation's
    normalization parameters. The ion sound gyroradius is computed. Although
    these can be modified, normalization parameters of a typical JET discharge
    are used.

    Parameters
    ----------
    sim : gbspy.Sim
        GBS simulation from which dimensions should be extracted.
    r_0 : float
        Major radius of the Tokamak (in meters).

    Returns
    -------
    r_0 : float
        Major radius of the Tokamak (in meters).
    l_r : float
        Minor radius of the plasma (or rather radial extent of the simulation
        domain, in meters).
    l_z : float
        Height of the plasma (or simulation domain, in meters).
    """

    # rho_s0 = np.sqrt(te_0 * miome * ELECTRON_MASS) / (b_0 * SPEED_OF_LIGHT)

    # Extract constants (rho_s0, miome) and normalized domain size
    rorho_s = sim.attribute['rorho_s']
    l_x = sim.attribute['xmax'] - sim.attribute['xmin']
    l_y = sim.attribute['ymax'] - sim.attribute['ymin']

    # Denormalize poloidal domain size and check that l_r/2 is smaller than the
    # major radius. Otherwise, this means the poloidal plane would extend to
    # "negative radii" when transforming coordinates.
    l_r = l_x / rorho_s * r_0
    if l_r > 2*r_0:
        logging.error(('Domain radial width exceed major radius: '
            'R0={} Lr={}.').format(r_0, l_r))
        if not allow_resize:
            raise ValueError(('Inconsistent domain size. Change major radius'
                    ' or allow resizing with "allow_resize=True".'))
        l_r = 1.9 * r_0
        rorho_s = l_x / l_r * r_0

    l_z = l_y / rorho_s * r_0

    return r_0, l_r, l_z


def get_vdb_points(r_0, l_r, l_z, x, y, z):
    """ Get a set of points for the VDB domain.

    The VDB domain, if it weren't sparse, would be a simple meshgrid of
    cartesian coordinates, where only points in the original torus would be be
    filled by the "fog". However, memory limitations may easily be encountered
    if a full meshgrid is created. Here, a sparse list of points will be
    returned (only those in the orginal domain).
    Note that the points are returned in a (Z, R, phi) order.

    Parameters
    ----------
    r_0 : float
        Major radius of the torus.
    l_r : float
        Minor radius of the torus.
    l_z : float
        Height of the torus.
    x : np.array
        1D array containing the x meshgrid coordinates.
    y : np.array
        1D array containing the y meshgrid coordinates.
    z : np.array
        1D array containing the z meshgrid coordinates.

    Returns
    -------
    coords : np.ndarray
        A two dimensional array of shape (npts, 3) containing the desired
        points.
    where : np.ndarray
        Same array as coords but containing index locations in the cartesian
        grid rather than cylindrical coordinates.
    """
    points = []
    where = []
    for ix_, x_ in enumerate(x):
        for iy_, y_ in enumerate(y):
            r_ = np.sqrt(x_**2 + y_**2)
            phi_ = np.arctan2(y_, x_)
            if r_0 - 0.5*l_r < r_ < r_0 + 0.5*l_r:
                for iz_, z_ in enumerate(z):
                    if 0. <= z_ <= l_z:
                        points.extend((z_, r_, phi_))
                        where.extend((ix_, iy_, iz_))

    return np.asarray(points).reshape(-1, 3), np.asarray(where).reshape(-1, 3)


def get_gbs_coordinates(r_0, l_r, l_z, n_rzphi):
    """ Get cylindrical coordinates

    For a given major radius, minor radius and height, compute the cylindrical
    coordinates of grid points, considering n_r points along the radial
    direction, n_z points along the vertical direction and n_phi points in the
    toroidal direction.
    The coordinates are returned as (r, z, phi) vectors. By convention, along
    phi, the coordinate ranges from -pi to pi.
    An exception is raised if the minor radius exceeds twice the major radius.

    Parameters
    ----------
    r_0 : float
        Major radius.
    l_r : float
        Minor radius.
    l_z : float
        Height of the torus.
    n_rzphi : tuple
        Contains three integers giving the resolution along r, z, and phi.

    Returns
    -------
    tuple
        A tuple containing 1D coordinate arrays ordered as (Z, R, phi).
    """
    if l_r > 2*r_0:
        raise ValueError(('Torus has inconsistent dimensions: minor radius is '
                'larger than twice the major radius.'))

    coord_r = r_0 + l_r*np.linspace(-.5, .5, n_rzphi[0])
    coord_z = np.linspace(0., l_z, n_rzphi[1])
    coord_phi = np.linspace(-np.pi, np.pi, n_rzphi[2])
    return coord_z, coord_r, coord_phi


def get_vdb_coordinates(r_0, l_r, l_z, n_xyz):
    """ Generates a set of VDB coordinates.

    This routine computes geometrical informations for the VDB grids. This
    information consists of
    - "world space" coordinates, i.e. a physical location. This will be
      necessary when interpolating the data from GBS to VDB. For this reason,
      the coordinates will be stored as a tuple of (z, r, phi) coordinates,
      following the storage conventions of GBS,
    - "index space" coordinates. These will be used by VDB to locate data
      within its storage. An index is mapped to a physical location directly
      unless a transform is prescribed,
    - a transform, which will be setup for our use case (a torus).

    The VDB points (as world or index coordinates) are returned in a np.ndarray
    of shape (npts, 3) where npts is the number of points in the VDB tree.

    Parameters
    ----------
    r_0 : float
        Major radius.
    l_r : float
        Minor radius.
    l_z : float
        Height of the torus.
    n_xyz : tuple
        Contains three integers giving the resolution along x, y, and z (the
        VDB grid axes).

    Returns
    -------
    world_zrphi : np.ndarray
        Array of shape (..., 3) containing the physical location of the VDB
        points in world space.
    index_ijk : np.ndarray
        An array similar to world_zrphi, but containing the VDB tree indices.
    transform : vdb.Transform
        A transform object which correctly scales and translates the tree
        indices to world coordinates.
    """
    # World space coordinates (xyz)
    world_x = (r_0+0.5*l_r) * np.linspace(-1., 1., n_xyz[0])
    world_y = (r_0+0.5*l_r) * np.linspace(-1., 1., n_xyz[1])
    world_z = np.linspace(0., l_z, n_xyz[2])

    # World spacing (for proper scaling in transform)
    world_dx = world_x[1] - world_x[0]
    world_dy = world_y[1] - world_y[0]
    world_dz = world_z[1] - world_z[0]

    # World offset. The VDB standard sets the "center" of an object to be at
    # indices (0, 0, 0). Since we use indices in the same sense as Python (all
    # positive), need to translate the data.
    world_tx = -r_0-0.5*l_r
    world_ty = -r_0-0.5*l_r
    world_tz = -0.5*l_z

    # The usual approach to get all combinations of world_[xyz], and thus all
    # available points, may generate too many values and create memory issues.
    # Rely on extra routine (isolated in case want to rework performance later,
    # e.g. numba).
    world_zrphi, index_ijk = get_vdb_points(r_0, l_r, l_z,
            world_x, world_y, world_z)

    # Can create the VDB transform right away
    transform = vdb.createLinearTransform([
        [float(world_dx), 0., 0., 0.],
        [0., float(world_dy), 0., 0.],
        [0., 0., float(world_dz), 0.],
        [float(world_tx), float(world_ty), float(world_tz), 1.]
        ])

    return world_zrphi, index_ijk, transform


def resample_gbs_field(field, coords, new_coords):
    """ Resamples the input field to different coordinates.

    A simple structured interpolation is conducted. This step requires the
    generation of a meshgrid for the new coordinates and may cause memory
    issues if a very high resolution is chosen.

    Parameters
    ----------
    field : np.ndarray
        A 3D array containing the field discretized on a torus grid.
    coords : tuple
        Contains three 1D np.array instances holding the (z, r, phi)
        coordinates of field.
    new_coords : tuple
        Contains three 1D np.array instances holding the high resolution (z, r,
        phi) coordinates to which field must be interpolated.

    Returns
    -------
    np.ndarray
        A 3D array containing the resampled field.
    """
    # Generate every possible point on the new grid and store them in a new
    # array of shape (..., 3) for the interpolation routine.
    mgrid = np.meshgrid(*new_coords, indexing='ij')
    mgrid = np.stack(mgrid, axis=3)  # creates a new axis
    mgrid_interp = mgrid.reshape((-1, 3))

    # Interpolate data and reshape it to a 3D array.
    new_field = spinterp.interpn(coords, field, mgrid_interp, method='linear',
            bounds_error=True)
    new_field = new_field.reshape(mgrid.shape[:-1])

    return new_field



def load_source_field(sim, name, when, negative_error=False):
    """ Load and format a GBS field.

    Loads a GBS field at a given time and formats it in a three dimensional
    array expected by VDB routines. The array is also verified to be positive
    unless 'allow_negative' is passed.

    Parameters
    ----------
    sim : gbspy.Sim
        GBS simulation to read from.
    name : str
        GBS field name to load.
    when : float
        Time at which the field should be loaded.
    negative_error : bool
        Whether negative values are allowed in the array or not.

    Returns
    -------
    np.ndarray
        A 3D Numpy array containing the desired field.
    """
    logging.info('Will load "{}" field at t={}.'.format(name, when))
    gbs_field = sim.get_field(name, t1=when, t2=when)
    gbs_field = gbs_field[2:-2, 2:-2, :, 0].squeeze()
    logging.info(('Loaded field "{}" and compacted to shape '
        '{}.').format(name, gbs_field.shape))

    # Check positive values only (fog)
    if np.any(gbs_field < 0.) and negative_error:
        raise ValueError((
            'The source GBS field (without ghosts) contains negative '
            'values. Only positive values are allowed in a VDB fog.'))

    return gbs_field


def create_vdb_grid(name, coords, field, transform):
    """ Returns a new VDB fog volume.

    This routine creates and returns a vdb.FloatGrid containing input values at
    input coordinates, and applies the provided transform. It also takes care
    of converting input arguments in a type accepted by the pyopenvdb API.

    Parameters
    ----------
    name : str
        Name of the VDB grid.
    coords : np.ndarray
        Array of shape (..., 3) containing a list of ijk coordinates at which
        the field argument is given.
    field : np.array
        1D array containing the field value at the provided coordinates.
    transform : vdb.Transform
        Transform to be applied to the grid.

    Returns
    -------
    vdb.FloatGrid
        A VDB FloatGrid with the provided field.
    """
    vdb_fog = vdb.FloatGrid()
    vdb_fog.gridClass = vdb.GridClass.FOG_VOLUME
    vdb_fog.name = name
    vdb_accessor = vdb_fog.getAccessor()
    logging.info('Filling VDB tree for grid "{}".'.format(name))
    for vdb_coord, vdb_value in zip(coords, field):
        ijk = tuple(int(x) for x in vdb_coord)
        val = float(vdb_value)
        vdb_accessor.setValueOn(ijk, val)
    del vdb_accessor
    vdb_fog.transform = transform
    return vdb_fog


def main():
    """ Parses arguments and runs converter

    The tool being designed to be called from a regular terminal emulator, we
    first parse input arguments and run the converter. In a fully "automatic"
    mode, the physical dimensions of the device will be calculated (to map the
    data to physical space), the resolution of the final domain will be
    computed, the initial data will eventually be resampled to a higher
    resolution and then mapped to the physical domain. Once this is done, an
    OpenVDB file sequence is written.
    """

    # Setup logging
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    # Parse arguments
    parser = argparse.ArgumentParser(description=('Process GBS results into '
        'the OpenVDB format.'))
    parser.add_argument('simulation_path', metavar='path', type=str, nargs='?',
        default=os.getcwd(), help='path to the GBS simulation')
    parser.add_argument('-o', '--output', default=os.getcwd(),
            type=str, help=(
                'output directory'))
    parser.add_argument('-p', '--prefix', default='output',
            type=str, help=(
                'output files are written as PREFIX_XXX.vdb'))
    parser.add_argument('-q', '--quality', default=1.0,
            type=float, help=(
                'controls the VDB grid density, with 1.0 being the same '
                'density as in the GBS grid'))
    parser.add_argument('--resample', type=float, help=(
        'a multiplier indicating how much denser the resampled grid must be'),
        metavar='mult')
    args = parser.parse_args()
    print(args)

    # Early validation of arguments. This is done to avoid long computations
    # before an expected error.
    if not os.path.isdir(args.output):
        raise ValueError('Output directory does not exist.')

    if re.findall(r'[^A-Za-z0-9_\-]', args.prefix):
        raise ValueError('Prefix can only contain alphanumeric characters.')


    # Open the GBS simulation
    simulation_path =  args.simulation_path
    if not os.path.isdir(simulation_path):
        raise ValueError('Provided path "{}" is not a directory.'.format(
            simulation_path))
    simulation = gbspy.Sim(simulation_path)

    # Get the physical extent of the original domain.
    r_0, l_r, l_z = extract_gbs_dimensions(simulation)
    logging.info('Torus dimensions should be (R0, Lr, Lz)=({}, {}, {})'.format(
        r_0, l_r, l_z))
    # And then, get its resolution (array size). Although result arrays also
    # include ghost cells, we'll discard them later.
    nx = simulation.attribute['nx']
    ny = simulation.attribute['ny']
    nz = simulation.attribute['nz']

    # Generate domain coordinates for the initial (cylindrical) domain and the
    # final (cartesian) domain. We start by creating the regular grid at which
    # the original data points are located.
    coord_gbs_zrphi = get_gbs_coordinates(r_0, l_r, l_z, (nx, ny, nz))

    # When the user wants to resample data from the simulation, it is best (for
    # correctness, visualization and performance purposes) to do it in the
    # original coordinate system, i.e. we need another cylindrical system.
    want_resampling = False
    if args.resample:
        want_resampling = True
        resample_factor = args.resample
        if resample_factor <= 0.:
            raise ValueError('Resample factor must be stricly positive.')
        elif resample_factor <= 1.:
            logging.warn((
                'Resampling to a less denser grid. You may '
                'want to adjust the VDB grid size instead.'))

        nx_resample = int(resample_factor*nx - 1)
        ny_resample = int(resample_factor*ny - 1)
        nz_resample = int(resample_factor*nz - 1)
        coord_gbs_resampled_zrphi = get_gbs_coordinates(r_0, l_r, l_z,
                (nx_resample, ny_resample, nz_resample))

    # Then, guess the dimensions of the cartesian domain and create a grid
    # mapping it. The resolution of this grid is chosen such that the point
    # density remains similar to the original GBS simulation (default) or
    # provided by the user.
    if args.quality <= 0.:
        raise ValueError('VDB grid quality must be stricly positive.')

    nx_vdb = int(args.quality * nx / l_r * 2*(r_0+0.5*l_r))
    nz_vdb = int(args.quality * nz)
    points_vdb_zrphi, coords_vdb_xyz, vdb_transform = \
            get_vdb_coordinates(r_0, l_r, l_z, (nx_vdb, nx_vdb, nz_vdb))
    logging.info('VDB pts: {}'.format(points_vdb_zrphi.shape))

    # Iterate over the input simulation's "frames", that is, each time step
    # that was saved. Each frame constitutes a different OpenVDB file and those
    # will be opened as a file sequence in a 3D modeling tool (e.g. Blender,
    # 3DS MAX, ...).
    simulation_frames = len(simulation.time)
    logging.info('Found {} frames for conversion.'.format(simulation_frames))
    # Define a mapping between GBS field name and VDB grid name.
    target_fields = {
            'n': 'density',
            'Te': 'temperature'}

    # --- Loop begins ---
    for time_index, time in enumerate(simulation.time):
        # Storage list for the generated grids
        vdb_grids = []

        for gbs_key, vdb_key in target_fields.items():
            # Load requested field (do not want to save each field in the output!)
            # and remove ghost cells. Also convert it from 4D to 3D (removing the
            # time index).
            gbs_field = load_source_field(simulation, gbs_key, time)

            # If required, resample the GBS field to a higher resolution.
            if want_resampling:
                logging.info(('Resampling field "{}" at frame '
                    '{}.').format(gbs_key, time_index))
                gbs_field = resample_gbs_field(gbs_field, coord_gbs_zrphi,
                        coord_gbs_resampled_zrphi)

            # Interpolation to VDB domain.
            logging.info(('VDB interpolation of field "{}" at frame '
                '{}.').format(gbs_key, time_index))
            if not want_resampling:
                source_coord = coord_gbs_zrphi
            else:
                source_coord = coord_gbs_resampled_zrphi
            vdb_field = spinterp.interpn(
                    source_coord,  # Field coordinates
                    gbs_field,  # Field
                    points_vdb_zrphi,  # List of target points
                    method='linear',
                    bounds_error=True)

            # Format the field to a VDB FloatGrid and append it to the list.
            vdb_grids.append(create_vdb_grid(vdb_key, coords_vdb_xyz,
                vdb_field, vdb_transform))

        # Create the output VDB file. Indices start at 1 since this is commonly
        # used as a reference for the first frame in Blender (and allows
        # detection of file series easily).
        file_number = str(time_index+1).zfill(len(str(simulation_frames)))
        file_name = args.prefix + '_' + file_number + '.vdb'
        file_path = os.path.join(args.output, file_name)
        logging.info('Start writing VDB to {}.'.format(file_name))
        vdb.write(file_path, grids=vdb_grids)
        logging.info('Done writing VDB to {}.'.format(file_name))
    # --- Loop ends ---


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        logging.critical('An exception has occured and is displayed below.')
        print(traceback.format_exc())
        raise
