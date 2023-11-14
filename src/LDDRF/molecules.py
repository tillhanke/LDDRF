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


def align_water_dimer(h2o, h2o_2, inplace=False, fixpoint="COM", rotation=True):
    """
    Aligning two water molecules such, that one of each OH bonds are parrallel
    and the orthogonal on the water planes are parrallel
    :param fixpoint: The point to fix, either COM (center of mass), O (oxygen position) or GEOM (geometric center)
    :param rotation: If True, the rotation matrix is returned, otherwise only the h2o_2 molecule is rotated
    """
    if not inplace:
        h2o_2 = h2o_2.copy()
    h2o = h2o.copy()
    # save fixpoint
    if fixpoint=="COM":
        fix = center_of_mass(h2o_2, unit=h2o_2.unit)  # save center of mass
    elif fixpoint=="O":
        fix = h2o_2.atom_coord(0, unit=h2o_2.unit)  # save oxygen position
    elif fixpoint=="GEOM":
        fix = h2o.atom_coords().mean(axis=0)  # save geometric center
    else:
        raise ValueError(f"fixpoint must be COM, GEOM or O, but got {fixpoint}")
    # move to origin
    h2o_2.set_geom_(h2o_2.atom_coords(unit=h2o_2.unit) - h2o_2.atom_coord(0, unit=h2o_2.unit), unit=h2o_2.unit)
    h2o.set_geom_(h2o.atom_coords(unit=h2o.unit) - h2o.atom_coord(0, unit=h2o.unit), unit=h2o.unit)
    # rotate h2o_2
    h2o_2, rot = rotate_OH(h2o, h2o_2, inplace=True, rotation=True)
    # move h2o_2 back
    if fixpoint=="COM":
        displacement = center_of_mass(h2o_2, unit=h2o_2.unit) - fix
    elif fixpoint=="GEOM":
        displacement = h2o_2.atom_coords().mean(axis=0) - fix
    # for case O fix is already the correct movement

    h2o_2.set_geom_(h2o_2.atom_coords(unit=h2o_2.unit) - displacement, unit=h2o_2.unit)
    if rotation:
        return h2o_2, rot
    return h2o_2


def rotate_OH(h2o, h2o_2, inplace=False, unit="BOHR", rotation=True):
    """
    Rotates the second h2o molecule around origin, such that O-H1 bond is parallel to that of the first h2o molecule
    :param rotation: If True, the rotation matrix is returned, otherwise only the h2o_2 molecule is rotated
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
    if rotation:
        return h2o_2, rotation
    else:
        return h2o_2


def check_water_alignment(h2o, h2o_2):
    oh1 = h2o.atom_coord(1, unit="BOHR") - h2o.atom_coord(0, unit="BOHR")
    oh1_ = h2o_2.atom_coord(1, unit="BOHR") - h2o_2.atom_coord(0, unit="BOHR")
    oh2 = h2o.atom_coord(2, unit="BOHR") - h2o.atom_coord(0, unit="BOHR")
    oh2_ = h2o_2.atom_coord(2, unit="BOHR") - h2o_2.atom_coord(0, unit="BOHR")
    # ch1-4 one oh bond is parallel
    ch1 = np.allclose(np.linalg.norm(np.dot(oh1, oh1_)), np.linalg.norm(oh1) * np.linalg.norm(oh1_))
    ch2 = np.allclose(np.linalg.norm(np.dot(oh2, oh2_)), np.linalg.norm(oh2) * np.linalg.norm(oh2_))
    ch3 = np.allclose(np.linalg.norm(np.dot(oh1, oh2_)), np.linalg.norm(oh1) * np.linalg.norm(oh2_))
    ch4 = np.allclose(np.linalg.norm(np.dot(oh2, oh1_)), np.linalg.norm(oh2) * np.linalg.norm(oh1_))
    # ch5 mols are in same plane and second oh bond is in same direction (left or right turned)
    if ch3 or ch4:
        ch5 = np.allclose(np.dot(np.cross(oh1, oh2), np.cross(oh1_, oh2_)), -np.linalg.norm(np.cross(oh1, oh2)) * np.linalg.norm(np.cross(oh1_, oh2_)))
    else:
        ch5 = np.allclose(np.dot(np.cross(oh1, oh2), np.cross(oh1_, oh2_)), np.linalg.norm(np.cross(oh1, oh2)) * np.linalg.norm(np.cross(oh1_, oh2_)))
    return (ch1 or ch2 or ch3 or ch4) and ch5
