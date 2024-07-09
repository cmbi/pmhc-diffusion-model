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
from openfold.np.residue_constants import (
    rigid_group_atom_positions,
    restype_name_to_atom14_names,
    restypes,
    restype_1to3,
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
)
from openfold.utils.feats import torsion_angles_to_frames, frames_and_literature_positions_to_atom14_pos


_log = logging.getLogger(__name__)


ca_group_id = 0
o_group_id = 3


def save(
    batch: Dict[str, Union[Rigid, torch.Tensor]],
    batch_index: int,
    path: str,
):

    # build biopython structure
    structure = Structure("")
    model = PDBModel(0)
    structure.add(model)
    chain = Chain('P')
    model.add(chain)
    n = 0

    # convert openfold data to tensors
    default_frames = torch.tensor(
        restype_rigid_group_default_frame,
        device=batch['frames'].device,
        requires_grad=False,
    )
    group_index = torch.tensor(
        restype_atom14_to_rigid_group,
        device=batch['frames'].device,
        dtype=torch.long,
        requires_grad=False,
    )
    literature_positions = torch.tensor(
        restype_atom14_rigid_group_positions,
        device=batch['frames'].device,
        requires_grad=False,
    )
    residue_type_atom14_mask = torch.tensor(
        restype_atom14_mask,
        device=batch['frames'].device,
        dtype=torch.bool,
        requires_grad=False,
    )
    torsion_frames = torsion_angles_to_frames(
        batch['frames'],
        batch['torsions'],
        batch['aatype'],
        default_frames,
    )
    atom14_positions = frames_and_literature_positions_to_atom14_pos(
        torsion_frames,
        batch['aatype'],
        default_frames,
        group_index,
        residue_type_atom14_mask,
        literature_positions,
    )

    # build peptide from frames & torsion angles
    atom_pos = {}
    residues = {}
    for residue_index, aa_index in enumerate(batch['aatype'][batch_index]):

        if batch['mask'][batch_index, residue_index]:

            frame = batch['frames'][batch_index, residue_index]

            # normalize quaternions
            trans = frame.get_trans()
            quats = frame.get_rots().get_quats()
            frame = Rigid(Rotation(quats=quats, normalize_quats=True), trans)

            aa_name = restype_1to3[restypes[aa_index]]

            res = Residue((" ", residue_index + 1, " "), aa_name, chain.id)
            chain.add(res)
            residues[residue_index] = res

            for atom_name, group_id, p in rigid_group_atom_positions[aa_name]:

                if group_id == ca_group_id:

                    p = frame.apply(torch.tensor(p))

                    n += 1
                    atom = Atom(atom_name, p, 0.0, 1.0, ' ', f" {atom_name} ", n, element=atom_name[0])
                    res.add(atom)

                    atom_pos[(residue_index, atom_name)] = p

            # side chain, except CB
            atom_names = restype_name_to_atom14_names[aa_name]
            for atom_index, atom_name in enumerate(atom_names):
                if atom_index > 4 and len(atom_name.strip()) > 0:
                    p = atom14_positions[batch_index, residue_index, atom_index]

                    n += 1
                    atom = Atom(atom_name, p, 0.0, 1.0, ' ', f" {atom_name} ", n, element=atom_name[0])
                    res.add(atom)

                    atom_pos[(residue_index, atom_name)] = p

            if residue_index > 0:
                # can add backbone oxygen

                # calculate position from CA, C, N atoms
                cac = torch.nn.functional.normalize(atom_pos[(residue_index - 1, "C")] - atom_pos[(residue_index - 1, "CA")], dim=-1)
                nc = torch.nn.functional.normalize(atom_pos[(residue_index - 1, "C")] - atom_pos[(residue_index, "N")], dim=-1)

                co = torch.nn.functional.normalize(cac + nc, dim=-1) * 1.24

                p = atom_pos[(residue_index - 1, "C")] + co

                n += 1
                atom = Atom("O", p, 0.0, 1.0, ' ', f" O  ", n, element="O")
                residues[residue_index - 1].add(atom)

            if not batch['mask'][batch_index, residue_index + 1] or (residue_index + 1) >= batch['aatype'].shape[1]:
                # can add terminal oxygens

                c = atom_pos[(residue_index, "C")]
                cac = torch.nn.functional.normalize(c - atom_pos[(residue_index, "CA")], dim=-1)

                o_frame = torsion_frames[batch_index, residue_index, o_group_id]

                # there's only one O atom in this group
                for atom_name, group_id, p in rigid_group_atom_positions[aa_name]:

                    if group_id == o_group_id and atom_name == "O":

                        # transform using psi-angle frame
                        o = o_frame.apply(torch.tensor(p))

                        # add O atom
                        n += 1
                        atom = Atom("O", o, 0.0, 1.0, ' ', f" O  ", n, element="O")
                        res.add(atom)

                        # mirror C-O bond in CA-C bond, to find terminal oxygen
                        co = o - c
                        co_proj_on_cac = cac * (co * cac).sum(dim=-1)
                        normal = co - co_proj_on_cac

                        oxt = c + co_proj_on_cac - normal

                        # add the terminal oxygen
                        n += 1
                        atom = Atom("OXT", oxt, 0.0, 1.0, ' ', f" OXT", n, element="O")
                        res.add(atom)

    # build pocket from atom positional data
    chain = Chain('M')
    model.add(chain)

    for res_index, aa_index in enumerate(batch['protein_aatype'][batch_index]):
        aa_name = restype_1to3[restypes[aa_index]]

        res = Residue((" ", res_index + 1, " "), aa_name, chain.id)
        chain.add(res)

        atom_names = restype_name_to_atom14_names[aa_name]
        for atom_index, atom_name in enumerate(atom_names):

            if batch['protein_atom14_exists'][batch_index, res_index, atom_index]:

                n += 1
                atom = Atom(
                    atom_name,
                    batch['protein_atom14_positions'][batch_index, res_index, atom_index],
                    0.0,
                    1.0,
                    ' ',
                    f" {atom_name} ",
                    n,
                    element=atom_name[0]
                )
                res.add(atom)

    # save biopython structure
    io = PDBIO()
    io.set_structure(structure)
    io.save(path)

    _log.debug(f"saved {path}")

