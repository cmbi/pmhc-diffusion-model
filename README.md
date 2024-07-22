# pMHC diffusion model

## dependencies

 - Pytorch 2.0.0
 - h5py 3.11.0
 - Numpy 2.0.0
 - BioPython 1.84
 - OpenFold 0.0.1

## input data

The input format is like in SwiftMHC.

```
HDF5 file:
 |
 + -- complex 1:
       + -- name (str)
       |
       + -- peptide
       |     |
       |     + -- backbone_rigid_tensor (P x 4 x 4)
       |     + -- aatype (P)
       |     + -- sequence_onehot (P x 22)
       |     + -- torsion_angles_sin_cos (P x 7 x 2)
       |     + -- torsion_angles_mask (P x 7)
       |
       + -- protein
             |
             + -- backbone_rigid_tensor (M x 4 x 4)
             + -- aatype (M)
             + -- sequence_onehot (M x 22)
             + -- atom14_gt_positions (M x 14 x 3)
             + -- atom14_gt_exists (M x 14)
             + -- cross_residues_mask (M)
```
## training

```
$ python optimize.py train_set.hdf5 100 model.pth
```

## testing

```
$ python test.pymodel.pth test_set.hdf5
```
