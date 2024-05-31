import logging

from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model as PDBModel
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio.PDB.PDBIO import PDBIO

import torch

from openfold.utils.rigid_utils import Rigid, Rotation
from openfold.np.residue_constants import rigid_group_atom_positions


_log = logging.getLogger(__name__)


def save(frames: Rigid, mask: torch.Tensor, path: str):

    structure = Structure("")
    model = PDBModel(0)
    structure.add(model)
    chain = Chain('A')
    model.add(chain)
    n = 0

    atom_pos = {}
    residues = {}
    for residue_index in range(frames.shape[0]):
        if mask[residue_index]:

            frame = frames[residue_index]

            # normalize quaternions
            trans = frame.get_trans()
            quats = frame.get_rots().get_quats()
            frame = Rigid(Rotation(quats=quats, normalize_quats=True), trans)

            res = Residue(("A", residue_index + 1, " "), "ALA", "A")
            chain.add(res)
            residues[residue_index] = res

            for atom_name, group_id, p in rigid_group_atom_positions["ALA"]:

                if group_id == 0:

                    p = frame.apply(torch.tensor(p))

                    n += 1
                    atom = Atom(atom_name, p, 0.0, 1.0, ' ', f" {atom_name} ", n, element=atom_name[0])
                    res.add(atom)

                    atom_pos[(residue_index, atom_name)] = p

            if residue_index > 0:
                # can add oxygen

                cac = torch.nn.functional.normalize(atom_pos[(residue_index - 1, "C")] - atom_pos[(residue_index - 1, "CA")], dim=-1)
                nc = torch.nn.functional.normalize(atom_pos[(residue_index - 1, "C")] - atom_pos[(residue_index, "N")], dim=-1)

                co = torch.nn.functional.normalize(cac + nc, dim=-1) * 1.24

                p = atom_pos[(residue_index - 1, "C")] + co

                n += 1
                atom = Atom("O", p, 0.0, 1.0, ' ', f" O  ", n, element="O")
                residues[residue_index - 1].add(atom)



    io = PDBIO()
    io.set_structure(structure)
    io.save(path)

    _log.debug(f"saved {path}")

