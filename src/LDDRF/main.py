# This file contains the main class for the lddrf calculation

import numpy as np
import tempfile
from pyscf.lib import logger, param, einsum
from pyscf.gto.mole import Mole, is_au
from pyscf.dft import Grids, RKS, numint
from LDDRF.potential.basis import generate_monomial_basis
from LDDRF import __version__ as lddrf_version, __file__ as lddrf_path
from LDDRF.drks import DRKS
from LDDRF.params import XC
from LDDRF.molecules import center_of_mass as com


def _init_disturbed(mol, potentials, grid, xc=None):
    """
    Initializes multiple disturbed DRKS objects for potentials with all the same grid and mol.
    :param mol:
    :param potentials:
    :param grid:
    :return: list of DRKS objects
    """
    assert is_au(mol.unit) == is_au(grid.mol.unit), "Units of mol and grid must be the same"
    assert is_au(mol.unit), "Unit of mol must be Bohr"
    if xc is None:
        xc = XC
    drks = []
    for pot in potentials:
        drks.append(DRKS(mol=mol,
                         potential=pot,
                         grid=grid,
                         xc=xc,
                         chkfile=False))
        # make sure, that no chkfile is created for the disturbed RKS
        if isinstance(drks[-1]._chkfile, tempfile._TemporaryFileWrapper):
            logger.warn(mol, f"for drks a chkfile was created at {drks[-1]._chkfile.name}, which will be deleted.")
            drks[-1]._chkfile.close()
    return drks



class LDDRF:
    """
    This is the main class for this project. It handles the calculation of the LDDRF and the application of it.
    It uses PYSCF at the base for all DFT related calculations.
    """
    def __init__(self,
                 mol: Mole,
                 pot_base: np.ndarray = None,
                 grid: Grids = None,
                 xc=XC,
                 chkfile=""
                 ):
        """
        This initializes the LDDRF class.
        It does not calculate anything relevant yet. It just assignes the neccessary information to the object.
        :param mol: The molecule for which the LDDRF should be calculated
        :param pot_base: The potential basis to be used for the moment expansion
        :param grid: the grid, on which the potentials are defined.
        if pot_base and grid are None a monomial potential base with standard values will be generated
        :param chkfile: checkpoint file to save moments and molecule info to.
            Writing to chkfile can be disabled if this attribute is set to None or False.
            If no chkfile is given, a temporary file will be created.
        """
        assert is_au(mol.unit), "Unit of mol must be Bohr"
        if not np.isclose(com(mol), [0,0,0]).all():
            logger.warn(mol, "The center of mass of the molecule is not at the origin."
                             "This might cause problems with the calculation of the LDDRF.")
        self.mol = mol
        self.grid = grid
        if self.grid is not None:
            assert is_au(grid.mol.unit), "Unit of grid must be Bohr"
        else:
            assert pot_base is None, "If potential base is defined, grid must be defined as well"
        self.pot_base = pot_base
        self.xc = xc
        self.chkfile = chkfile
        if self.chkfile == "":
            self._chkfile = tempfile.NamedTemporaryFile(dir=param.TMPDIR, suffix='.ldrt')
            self.chkfile = self._chkfile.name

        # initialize placeholders for the calculations
        self.undisturbed = None
        self.disturbed = None
        self.lddrf = None

    def build(self):
        """
        This function builds the neccessary objects for the calculation of the LDDRF.
        Including the potential basis
        :return: LDDRF object with initialized attributes
        """
        if self.grid is None:
            assert self.pot_base is None, "if potential base is defined, grid must be defined as well"
            self.grid = Grids(self.mol)
            self.grid.build()
        else:
            assert is_au(self.grid.mol.unit), "Unit of grid must be Bohr"
            if self.grid.coords is None:
                self.grid.build()
        if self.undisturbed is None:
            self.undisturbed = self.mol.RKS(xc=self.xc)
            if isinstance(self.undisturbed._chkfile, tempfile._TemporaryFileWrapper):
                self.undisturbed._chkfile.close()
            self.undisturbed.chkfile = False  # disable writing to chkfile for calculations
            self.undisturbed.kernel()
        else:
            assert self.undisturbed.mol == self.mol, "mol of undisturbed and mol must be the same"
            assert self.undisturbed.xc == self.xc, f"xc of undisturbed and xc must be the same, but are {self.undisturbed.xc} and {self.xc}"
            if self.undisturbed.e_tot == 0:
                self.undisturbed.kernel()
        if self.pot_base is None:
            # generate a monomial basis centered at the center of mass of the molecule
            self.pot_base = generate_monomial_basis(
                mol=self.mol,
                order=2,
                grid=self.grid,
                dft=self.undisturbed
            )
        else:
            assert self.pot_base.shape[1] == self.grid.coords.shape[0], f"pot_base and grid must have the same first dimension, but are {self.pot_base.shape[1]} and {self.grid.coords.shape[0]}"
        self.disturbed = _init_disturbed(mol=self.mol,
                                         potentials=self.pot_base,
                                         grid=self.grid,
                                         xc=self.xc)
        self.lddrf = None
        return self

    def kernel(self):
        """
        This function calculates the LDDRF for the given molecule and potential basis.
        :return: ndarray with LDDRF values in molecular basis
        """
        if self.undisturbed is None or self.disturbed is None:
            self.build()
        def insert_commit():
            import git
            commit = git.Repo(lddrf_path, search_parent_directories=True).head.commit.hexsha
            return commit
        logger.info(self.mol,
                    "#####################\n" +
                    "Calculating LDDRF\n" +
                    f"xc: {self.xc}\n" +
                    f"potential base size: {self.pot_base.shape}\n" +
                    f"LDDRF Version: {lddrf_version}\n" +
                    "#####################"
                    )
        if "dev" in lddrf_version:
            logger.warn(self.mol, f"Using developement Version\nGit Commit: {(hash:=insert_commit())}")
            from pyscf.lib import chkfile as chktool
            chktool.dump(chkfile=self.chkfile, key='DevCommit', value=hash)
        if self.undisturbed.e_tot == 0:
            # if the undisturbed calculation is not yet run do that now
            self.undisturbed.kernel()

        def calc_disturbed(lddrf, drks, id):
            # this function runs the disturbed calculations for the molecule disturbed by the different potentials
            logger.info(lddrf.mol, f"\nCalculating DRKS for potential {potential_ind}")
            if drks.e_tot == 0:
                drks.kernel()
            if not drks.converged:
                # if it does not converge immediately it can be restarted and run again with the solution of the
                # undisturbed calculation as initial guess. This sometimes lets it converge even, if minao initial
                # guess did not work
                logger.note(
                    lddrf.mol,
                    f"DRKS for potential {id} did not converge, running again with undisturbed initial guess"
                )
                drks.kernel(dm0=lddrf.undisturbed.make_rdm1())
                if not drks.converged:
                    # if it still does not converge raise an error, because it needs to converge in order to use for
                    # LDDRF later on
                    raise RuntimeError(f"DRKS for potential {id} did not converge")

        def check_disturbance(lddrf, drks, id, threshold=1e-4):
            # check the energy difference to undisturbed
            # if it is to high, we might not be in a "useful" linear regime
            if (energy_diff := abs(lddrf.undisturbed.e_tot - drks.e_tot)) / abs(lddrf.undisturbed.e_tot) > threshold:
                logger.note(
                    drks.mol,
                    f"relative energy difference between undisturbed and DRKS for potential {id} is larger than 5e-7, rescaling potential"
                )
                # rescale the potential, and recalculate DRKS
                drks.potential = drks.potential * threshold / energy_diff
                lddrf.pot_base[id] = drks.potential
                drks.e_tot = 0  # reset energy and force recalculation
                calc_disturbed(lddrf, drks, id)
                energy_diff = abs(lddrf.undisturbed.e_tot - drks.e_tot)
            logger.note(drks.mol, f"Relative energy difference: {energy_diff/abs(lddrf.undisturbed.e_tot)}")

        from tqdm import tqdm
        # if the verbosity setting is low enough it is possible to just display some progressbar, so one can see, that
        # stuff is happening
        for potential_ind, drks in enumerate(self.disturbed) if self.mol.verbose >= logger.NOTE else tqdm(enumerate(self.disturbed), total=len(self.disturbed)):
            # now just call the functions to run disturbed calculations for each potential
            calc_disturbed(self, drks, potential_ind)
            check_disturbance(self, drks, potential_ind)

        # the minus sign must be there, because otherwise the overlap is not positive definite
        density_difference = _density_diff(self.mol, self.undisturbed, self.disturbed, self.grid)
        dens_pot_overlap = einsum(
            'ir, jr -> ij',
            -density_difference,
            self.pot_base * self.grid.weights[None, :]
        )
        moments = np.linalg.cholesky(dens_pot_overlap)
        # TODO: the minus sign is taken from the old code should be checked
        self.lddrf = einsum("ij, jkl -> ikl", np.linalg.inv(moments), -_dm_diffs(self.undisturbed, self.disturbed))
        self.store()
        return self

    def on_grid(self, coords:np.ndarray, mol:Mole = None):
        """
        Calculate LDDRF values on grid
        Args:
            coords: realspace grid on which the LDDRF are calculated
            mol: moleculare basis to be used
                if not provided self.mol will be used
        Returns:
            LDDRF values on grid
        """
        #TODO: find way to rotate grid, such that mol is oriented the same way as self.mol is
        assert coords.shape[1] == 3
        if mol is None:
            mol = self.mol
        ao = mol.eval_gto("GTOval", coords)
        return np.array([numint.eval_rho(mol, ao, lddrf_i) for lddrf_i in self.lddrf])

    def apply_to_pot(self, potential, grid, mol=None):
        if mol is None:
            mol = self.mol
        assert is_au(mol.unit) and is_au(grid.mol.unit), "Molecule and grid units must be BOHR"
        lddrf_grid = self.on_grid(grid.coords, mol)
        lddrf_pot_ov = einsum("ir, r -> i", lddrf_grid, potential * grid.weights)
        return einsum("i, ir -> r", lddrf_pot_ov, lddrf_grid)

    def store(self):
        """
        Store the LDDRF in the HDF5 file
        """
        if not self.chkfile:
            return
        from pyscf.lib import chkfile as chktool
        chktool.dump_mol(mol=self.mol, chkfile=self.chkfile)
        chktool.dump(chkfile=self.chkfile, key='lddrf', value=self.lddrf)
        chktool.dump(chkfile=self.chkfile, key='version', value=lddrf_version)

    @staticmethod
    def load_from_chkfile(chkfile):
        """
        Load LDDRF from a checkpoint file
        Attention: this only loads the resulting LDDRF values. Not the distrubed calculations.
        :param chkfile: path to chkfile
        :return: LDDRF object
        """
        from pyscf.lib import chkfile as chktool
        lddrf = LDDRF(
            mol=chktool.load_mol(chkfile=chkfile)
        )
        lddrf.lddrf = chktool.load(chkfile=chkfile, key='lddrt')
        return lddrf

def _density_diff(mol:Mole, dft:RKS, ddfts:list, grid:Grids):
    """
    This function calculates electron desnity differences on the grid for dft calculations of the same molecule.
    Using one undisturbed dft calculation (dft) and multiple disturbed dft calculations (ddfts)
    :param mol: molecule
    :param dft: single RKS calculation for mol
    :param ddfts: list of DRKS calculations for mol
    :param grid: grid, on which the electron density should be returned
    :return: array of density differences on grid such, that rho_0 + rho_diff = rho_ddft
    """
    assert mol == dft.mol, "mol and dft.mol must be the same"
    assert is_au(mol.unit) == is_au(grid.mol.unit), "mol and grid must have the same unit"
    # use GTO atomic orbitals of mol
    ao = mol.eval_gto("GTOval", grid.coords)
    # undistrubed density
    dm0 = dft.make_rdm1()
    dens0 = numint.eval_rho(mol, ao, dm0)
    densd = np.zeros((len(ddfts), dens0.shape[0]))
    for i, ddft in enumerate(ddfts):
        assert mol == ddft.mol, f"mol and mol of disturbed calculation {i} must be the same"
        densd[i] = numint.eval_rho(mol, ao, ddft.make_rdm1()) - dens0
    return densd

def _dm_diffs(dft:RKS, ddfts:list):
    """
    This function calculates the difference between the density matrices of the undisturbed and the disturbed dft calculations.
    :param dft: undisturbed dft calculation
    :param ddfts: list of disturbed dft calculations
    :return: array of density matrix differences
    """
    dm0 = dft.make_rdm1()
    dmd = np.zeros((len(ddfts), *dm0.shape))
    for i, ddft in enumerate(ddfts):
        assert dft.mol == ddft.mol, f"mols of dfts must be the same, check disturbed dft {i} "
        dmd[i] = ddft.make_rdm1() - dm0
    return dmd


