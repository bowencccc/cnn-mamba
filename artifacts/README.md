# External model artifacts

Existing best checkpoints belong under
`artifacts/checkpoints/{heldout,chrxtrain}/w{5,10,20,30,40,80}/best_model.pt`.
They are listed with hashes in `external_manifest.tsv` and are not tracked by
Git. `heldout` models are the honest chrX candidate-AUPRC models. `chrxtrain`
models intentionally include chrX and are the models used for the UniAnn
positive-control/window-size downstream experiment.
