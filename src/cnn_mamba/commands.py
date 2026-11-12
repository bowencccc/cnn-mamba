"""Stable command names exposed by the CNN--Mamba command-line interface."""

COMMAND_MODULES = {
    "prepare-droso": "cnn_mamba.prepare_droso",
    "prepare-human": "cnn_mamba.prepare_human",
    "train": "cnn_mamba.train",
    "score": "cnn_mamba.score_chromosome",
    "score-joint-frame": "cnn_mamba.score_joint_frame_chromosome",
    "auprc": "cnn_mamba.evaluate_candidates",
    "annotation-gap": "cnn_mamba.evaluate_annotation_gap",
    "export-uniann": "cnn_mamba.export_uniann_scores",
    "export-joint-frame": "cnn_mamba.export_joint_frame_tsv",
    "build-uniann-emissions": "cnn_mamba.build_uniann_emissions",
    "uniann-eval": "cnn_mamba.uniann_evaluate",
    "calibrate-joint-frame": "cnn_mamba.calibrate_joint_frame",
    "fit-vector-scaling": "cnn_mamba.fit_vector_scaling",
    "current-best": "cnn_mamba.current_best",
}
