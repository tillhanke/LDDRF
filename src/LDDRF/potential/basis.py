import numpy as np
from pyscf.gto import Mole
from pyscf.dft.numint import eval_rho
from pyscf.dft import Grids
from LDDRF.params import XC
from LDDRF.molecules import center_of_mass as com


# NOTE: 1e-4 is approximately the same as 1e-3 for a cube based grid as was used in the original code
def find_mol_border(mol, fraction=1e-4, dft=None):
    """
    find x, y, z max, where the density first reaches fraction of maximum
    Args:
        mol: molecule
        fraction: fraction of maximum density to be reached
        dft: pre calculated dft object of molecule, if not given, it will be calculated
    Returns:
        x, y, z: distance of COM of molecule, where density first reaches fraction of maximum in units of bohr
    """
    if dft is None:
        dft = mol.RKS(xc=XC)
    if dft.e_tot == 0:
        dft.kernel()

    ao = mol.eval_gto("GTOval", dft.grids.coords)
    dens = eval_rho(mol, ao, dft.make_rdm1())
    threshol = dens.max() * fraction
    dens_low = np.where(dens < threshol + 1e-2 * threshol, dens, 0)
    pos_thres = np.where(dens_low > threshol - 1e-2 * threshol)
    # coords relative to center of mass of mol
    coords = dft.grids.coords - com(mol)
    xmin, xmax = coords[pos_thres][:, 0].min(), coords[pos_thres][:, 0].max()
    ymin, ymax = coords[pos_thres][:, 1].min(), coords[pos_thres][:, 1].max()
    zmin, zmax = coords[pos_thres][:, 2].min(), coords[pos_thres][:, 2].max()
    xmax = max(abs(xmin), abs(xmax))
    ymax = max(abs(ymin), abs(ymax))
    zmax = max(abs(zmin), abs(zmax))
    _xyzmax = np.array([xmax, ymax, zmax])
    return _xyzmax


def monomial_basis(coords: np.ndarray, order: int, xcyczc, constant: float=1e-5):
    """
    monomial basis centered around the origin, with a fixed value (constant) at (xc, yc, zc)
    Args:
        coords: grid of points
        order: maximum order of potential
        xcyczc: tuple (x, y, z) at which all potentials should have the same value
        constant: constant of potential
    """

    potential = np.copy(coords) / xcyczc
    all_pots = np.copy(coords) / xcyczc
    # blackmagic fuckery to get every combination only once
    # i.e. xz == zx is only once listed
    xyz = np.array([2, 3, 5])
    assert len(xcyczc) == 3
    list_xyz = xyz
    for ord_step in range(2, order+1):
        potential = np.einsum("ij, ik -> ijk", potential, coords).reshape(coords.shape[0], 3 ** ord_step)
        list_xyz = (list_xyz * xyz[:, np.newaxis]).reshape(3**ord_step,)
        all_pots = np.append(
                all_pots,
                potential[:, np.unique(list_xyz, return_index=True)[1]],
                axis=1
                )
    # returns x, y, z, xx, xy, yy, xz, zy, zz, ...
    # the correct list of elements can be generated via labels()
    return all_pots.T * constant


def monomial_labels(order: int):
    """
    create labels for order of potentials
    in style of x, y, z, xx, ...
    :param order: maximum order to return
    """
    labels = np.array(["x", "y", "z"], dtype=object)
    all_labels = []
    all_labels.extend(labels)
    ord_label = labels
    xyz = np.array([2, 3, 5])
    list_xyz = xyz
    if order == 1:
        return all_labels
    for i in range(2, order + 1):
        list_xyz = (list_xyz * xyz[:, np.newaxis]).flatten()
        ord_label = (ord_label + labels[:, np.newaxis]).flatten()
        all_labels.extend(ord_label[np.unique(list_xyz, return_index=True)[1]])
    return all_labels


def generate_monomial_basis(mol: Mole, order: int, dft=None, grid=None):
    """
    generate monomial basis centered around COM of the molecule
    :param mol: molecule object, which is reference for the monomial basis
    :param order: maximum order of the basis
    :param dft: DFT object, which is used to calculate the electron density of the molecule
    :param grid: Grid object, for which the values of the potentials should be returned
    :return: potential values for grid coordinates
    """
    if dft is None:
        dft = mol.RKS(xc=XC).run()

    if grid is None:
        grid = Grids(mol)

    mol_border = find_mol_border(
        mol=mol,
        dft=dft
    )
    monomials = monomial_basis(
        coords=grid.coords-com(mol),  # important, because mol_border is relative to COM
        order=order,
        xcyczc=mol_border,
        )
    return monomials
