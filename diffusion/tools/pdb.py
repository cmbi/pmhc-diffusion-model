from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model as PDBModel
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio.PDB.PDBIO import PDBIO

import torch

from openfold.utils.rigid_utils import Rigid, Rotation
from openfold.np.residue_constants import rigid_group_atom_positions


def save(frames: Rigid, mask: torch.Tensor, path: str):

    structure = Structure("")
    model = PDBModel(0)
    structure.add(model)
    chain = Chain('A')
    model.add(chain)
    for i in range(frames.shape[0]):
        if mask[i]:

            frame = frames[i]

            # normalize quaternions
            trans = frame.get_trans()
            quats = frame.get_rots().get_quats()
            frame = Rigid(Rotation(quats=quats, normalize_quats=True), trans)

            res = Residue(("A", i + 1, " "), "ALA", "A")
            chain.add(res)

            for atom_name, group_id, p in rigid_group_atom_positions["ALA"]:

                if group_id == 0:

                    p = frame.apply(torch.tensor(p))

                    atom = Atom(atom_name, p, 0.0, 1.0, ' ', f" {atom_name} ", atom_name[0])
                    res.add(atom)

    io = PDBIO()
    io.set_structure(structure)
    io.save(path)

