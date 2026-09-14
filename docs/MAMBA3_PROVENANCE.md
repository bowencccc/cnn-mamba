# Mamba3 dependency provenance

The original environment reports `mamba_ssm` version `2.3.1`, but it was not
installed from the PyPI 2.3.1 release. Python package metadata contains:

```json
{"url":"https://github.com/state-spaces/mamba.git","vcs_info":{"commit_id":"be0303b971cd79a4fbbcfc411a01b33f6b0e3602","vcs":"git"}}
```

Verified original installation:

- `from mamba_ssm import Mamba3` succeeds;
- class source: `mamba_ssm/modules/mamba3.py`;
- `mamba3.py` SHA256:
  `de1b104865c941eb6eeac63c3a75d5913fc2e7a119d52991590ef19c381b05ac`;
- dependency versions included `tilelang==0.1.8`,
  `apache-tvm-ffi==0.1.9`, `quack-kernels==0.3.10`, and `triton==3.6.0`.

Install command:

```bash
pip install --no-build-isolation \
  "mamba-ssm @ git+https://github.com/state-spaces/mamba.git@be0303b971cd79a4fbbcfc411a01b33f6b0e3602"
```

Run `python scripts/smoke_model.py` on a GPU compute node. It checks recorded
Git provenance when available and performs an actual Mamba3 CUDA
forward/backward pass.
