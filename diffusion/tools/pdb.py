import logging
from typing import Dict, Union

from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model as PDBModel
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio.PDB.PDBIO import PDBIO

import torch

from openfold.utils.rigid_utils import Rigid, Rotation
from openfold.np.residue_constants import rigid_group_atom_positions, restype_name_to_atom14_names, restypes, restype_1to3


_log = logging.getLogger(__name__)


def save(
    batch: Dict[str, Union[Rigid, torch.Tensor]],
    batch_index: int,
    path: str,
):

    structure = Structure("")
    model = PDBModel(0)
    structure.add(model)
    chain = Chain('P')
    model.add(chain)
    n = 0

    # build peptide
    atom_pos = {}
    residues = {}
    for residue_index in range(batch['frames'].shape[1]):
        if batch['mask'][batch_index, residue_index]:

            frame = batch['frames'][batch_index, residue_index]

            # normalize quaternions
            trans = frame.get_trans()
            quats = frame.get_rots().get_quats()
            frame = Rigid(Rotation(quats=quats, normalize_quats=True), trans)

            res = Residue((" ", residue_index + 1, " "), "ALA", chain.id)
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

    # build pocket
    chain = Chain('M')
    model.add(chain)

    for res_index, aa_index in enumerate(batch['pocket_aatype'][batch_index]):
        aa_name = restype_1to3[restypes[aa_index]]

        res = Residue((" ", res_index + 1, " "), aa_name, chain.id)
        chain.add(res)

        atom_names = restype_name_to_atom14_names[aa_name]
        for atom_index, atom_name in enumerate(atom_names):

            if batch['pocket_atom14_exists'][batch_index, res_index, atom_index]:

                n += 1
                atom = Atom(
                    atom_name,
                    batch['pocket_atom14_positions'][batch_index, res_index, atom_index],
                    0.0,
                    1.0,
                    ' ',
                    f" {atom_name} ",
                    n,
                    element=atom_name[0]
                )
                res.add(atom)

    io = PDBIO()
    io.set_structure(structure)
    io.save(path)

    _log.debug(f"saved {path}")

