import numpy as np
import tempfile
from pyscf.dft.rks import RKS
from pyscf.dft import Grids
from pyscf.gto.mole import is_au
from pyscf.lib.parameters import BOHR
from pyscf.lib import einsum, logger


class DRKS(RKS):
    """
    This is a RKS class, that implements the option of a disturbed RKS calculation.
    The disturbing potential must be given on a realspace grid.
    """
    def __init__(self,
                 mol,
                 *args,
                 potential: np.ndarray=None,
                 grid: Grids=None,
                 chkfile="",
                 **kwargs
                 ):
        """
        This initializes the DRKS class.
        :param potential: the disturbing potential values for all grid points defined within the grid
        :param grid: a Grids object containing the grid points, on which the potential is defined
        :param chkfile: checkpoint file to save MOs, orbital energies etc.  Writing to
            chkfile can be disabled if this attribute is set to None or False.
        """
        self._potential = potential
        self._grid = grid
        self.mol = mol
        logger.debug(self.mol, "These kwargs are passed to DRKS:")
        logger.debug(self.mol, kwargs.__str__())
        super().__init__(*args, mol=mol, **kwargs)
        if self._grid is not None:
            assert is_au(self.mol.unit) == is_au(self._grid.mol.unit), "Units of mol and grid must be the same"
            if is_au(self.mol.unit):
                self._unit = 1
            else:
                self._unit = BOHR
        if not chkfile:
            if isinstance(self._chkfile, tempfile._TemporaryFileWrapper):
                self._chkfile.close()
                self._chkfile = None
        else:
            self.chkfile = chkfile

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, x):
        assert is_au(self.mol.unit) == is_au(x.mol.unit), "Units of mol and grid must be the same"
        self._grid = x

    @property
    def potential(self):
        return self._potential

    @potential.setter
    def potential(self, x):
        self._potential = x

    def get_hcore(self, mol=None):
        if mol is None:
            mol = self.mol
        hcore = super().get_hcore(mol)
        gto = mol.eval_gto('GTOval_sph', self._grid.coords/self._unit)
        dist = einsum('i,ij->ij', self._potential*self._grid.weights, gto)
        dist = einsum('xj, xi -> ji', dist, gto)
        return hcore + dist


