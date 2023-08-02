import numpy as np
from scipy.spatial.transform import Rotation


def center_of_mass(mol, unit="Bohr"):
    """
    This function returns the center of mass for the mole object
    args:
        mol: Mole object
    """
    return np.dot(
        mol.atom_mass_list(),
        mol.atom_coords(unit=unit)
    ) / mol.atom_mass_list().sum()


def align_water_dimer(h2o, h2o_2, inplace=False):
    """
    Aligning two water molecules such, that the Oxygen atoms are at the origin and one of each OH bonds are parrallel
    """
    if not inplace:
        h2o_2 = h2o_2.copy()
        h2o = h2o.copy()
    # move to origin
    h2o_2.set_geom_(h2o_2.atom_coords(unit=h2o_2.unit) - h2o_2.atom_coord(0, unit=h2o_2.unit), unit=h2o_2.unit)
    h2o.set_geom_(h2o.atom_coords(unit=h2o.unit) - h2o.atom_coord(0, unit=h2o.unit), unit=h2o.unit)
    # rotate h2o_2
    rotate_OH(h2o, h2o_2, inplace=True)
    return h2o, h2o_2


def rotate_OH(h2o, h2o_2, inplace=False, unit="BOHR"):
    """
    Rotates the second h2o molecule, such that O-H1 bond is parallel to that of the first h2o molecule
    """
    def euler_from_waterdimer(h2o_1, h2o_2, with_vectors=False):
        """
        Calculates the euler angles for the water dimer to rotate the OH bond of  h2o_2 onto that of h2o_1
        These are changing h2o_2 into h2o_1
        in zxz convention
        """
        # get coord system of h2o_1
        x1 = h2o_1.atom_coord(1, unit=unit)-h2o_1.atom_coord(0, unit=unit)
        x1 /= np.linalg.norm(x1)
        z1 = np.cross(h2o_1.atom_coord(2, unit=unit)-h2o_1.atom_coord(0, unit=unit), x1)
        z1 /= np.linalg.norm(z1)
        # coord system for h2o_2
        x2 = h2o_2.atom_coord(1, unit=unit)-h2o_2.atom_coord(0, unit=unit)
        x2 /= np.linalg.norm(x2)
        z2 = np.cross(h2o_2.atom_coord(2, unit=unit)-h2o_2.atom_coord(0, unit=unit), x2)
        z2 /= np.linalg.norm(z2)
        # vector of crossection of x1y1 and x2y2 planes
        n = np.cross(z1, z2)
        n /= np.linalg.norm(n)
        # calculate euler angles
        phi = np.sign(np.dot(x1, np.cross(z1, n))) * np.arccos(np.dot(n, x1))
        theta = np.sign(np.dot(z1, np.cross(n, z2))) * np.arccos(np.dot(z1, z2))
        psi = np.sign(np.dot(n, np.cross(z2, x2))) * np.arccos(np.dot(n, x2))
        if with_vectors:
            return phi, theta, psi, z2, n, z1
        return phi, theta, psi

    if not inplace:
        h2o_2 = h2o_2.copy()

    if not np.allclose([h2o_2.atom_coord(0, unit=unit), h2o.atom_coord(0, unit=unit)], np.zeros(3)):
        raise ValueError("Both Oxygen atoms must be at the origin")
    phi, theta, psi, z2, n, z = euler_from_waterdimer(h2o, h2o_2, with_vectors=True)
    first_rotation = Rotation.from_rotvec(z2*psi)
    second_rotation = Rotation.from_rotvec(n*theta)
    third_rotation = Rotation.from_rotvec(z*phi)
    rotation = third_rotation * second_rotation * first_rotation
    h2o_2.set_geom_(rotation.apply(h2o_2.atom_coords(unit=unit)), unit=unit)
    return h2o_2

