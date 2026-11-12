# Experiment F: Evo 2 causal belief-state probe

Status: **stopped at the mandatory environment check**. No chr22 admission
diagnostic and no A/B probe was run, because neither requested Evo 2 model
completed a valid forward pass on the available GPU.

## Environment check

The host has one NVIDIA GeForce RTX 4070 (12,282 MiB, compute capability 8.9),
driver 550.144.03, CUDA 12.4, Python 3.11, and PyTorch 2.5.1+cu124. Transformer
Engine 2.4.0 reports FP8 support on this GPU. The Vortex FlashAttention CUDA
extension compiled and imported successfully.

| Model | Official checkpoint | Context relevant to this experiment | Result |
|---|---:|---:|---|
| Evo 2 1B base | 2.698 GB | 8,192 nt | Failed while loading FP8 checkpoint metadata, before a 512-nt forward pass |
| Evo 2 7B base | 13.006 GB | 8,192 nt | Not loaded: checkpoint alone exceeds 12.576 GB physical VRAM |

The Evo 2 1B failure is a Vortex/Transformer Engine compatibility failure:
checkpoint FP8 state uses an amax-history length of 16, while the installed
Transformer Engine runtime allocated length 1024. The model architecture was
created, but no checkpoint-backed logits were produced. Deleting or silently
reinitializing the saved FP8 state was not accepted as a valid scientific run.
Therefore speed, peak forward memory, and maximum executable context are not
reported.

The exact source revisions, package versions, model revisions, checkpoint hash,
and failure state are recorded in `config.json`.

## Admission and probes

The mandatory chr22 CDS next-base diagnostic was not run. Consequently the
four-state A probe and coding-intron B probe were not run. This follows the
pre-specified gate rather than treating environment failure as a negative
scientific result.

For any future successful run, chr22 must be described as present in Evo 2's
pretraining corpus scope; Evo 2 did not use CHESS annotation labels during
pretraining. Thus chr22 is an annotation-label test set, not a sequence-unseen
test set for Evo 2.
