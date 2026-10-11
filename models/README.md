# Published inference checkpoints

## Human four-state chr1-held-out model

`human_joint_frame_beta010_chr1held_epoch4_inference.pt` is the inference-only
form of the completed Cybertron Human four-state model. It retains the exact
epoch-4 `model_state_dict`, model configuration and validation provenance, but
omits optimizer, gradient-scaler and RNG state. It is suitable for scoring and
evaluation, but not for exactly resuming the interrupted optimizer trajectory.

- architecture: 10 kb CNN-Mamba-k7, 5 kb stride, frame dilation 3;
- four-state head: non-CDS/frame-0/frame-1/frame-2;
- joint auxiliary coefficient: 0.1; noncoding within-head weight: 0.1;
- training regime: chr1 held out; five epochs; best epoch 4;
- portable checkpoint SHA256:
  `429904082463e9822ec83aa15cfd48b16cf0cdb6ff15ff4e7141316117f9c037`;
- source full checkpoint SHA256:
  `30dc6f082d59acb9e8b84d2659e68b6fdaacf49bdeb1025c09364862e9acc287`.

The full 52 MiB resumable checkpoint remains external. Its optimizer state is
not needed for Rockfish inference tests. See the matching result report under
`results/human_joint_frame_beta010_v1/chr1held/`.
