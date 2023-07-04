import numpy as np


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
