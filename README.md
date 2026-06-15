# Mixed-Sequence MIL for Lumbar Segment Classification

This repository provides the research code associated with the manuscript:

**Mitigating MRI Sequence Bias in Weakly Supervised Lumbar Segment Classification Using Mixed-Sequence Multiple Instance Learning with Sparse Priors**

The code implements a weakly supervised multiple-instance learning framework for segment-level lumbar MRI classification using sequence-level bags derived from sagittal T2-weighted imaging (T2WI) and fat-suppressed T2-weighted imaging (T2FS).

## Repository contents

* `abmil_model.py`: model definitions, including the CNN encoder, ABMIL baseline, and Gated-Temp MIL model.
* `abmil_dataset.py`: dataset and sequence-level bag construction utilities.
* `train_abmil.py`: training and cross-validation script.
* `train_utils.py`: training, evaluation, and metric utilities.
* `examples/example_labels_schema.csv`: example label-table schema.
* `examples/config_example.yaml`: example configuration file.

## Data availability

No original MRI images, DICOM files, converted clinical arrays, or identifiable patient information are included in this repository. Access to the study imaging data is restricted by patient privacy protection protocols and institutional regulations.

## Code availability

This repository contains the reproducible research version of the code used for the manuscript. The code is shared to support academic reproducibility and reuse.

## Environment

The experiments were conducted using Python 3.10 and PyTorch 2.0. Main dependencies include PyTorch, NumPy, pandas, scikit-learn, SciPy, matplotlib, tqdm, and PyYAML.

## Notes

The code is associated with registered software copyright. Public release of this research version does not include any protected clinical data or patient-identifiable information.
