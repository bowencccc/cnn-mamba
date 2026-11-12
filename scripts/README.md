# Scripts directory

The supported software interface is the installed `cnn-mamba` command and the
modules under `src/cnn_mamba/`. Files in this directory are research workflows,
figure generators, cluster launchers, and exact historical experiment recipes.
They are retained for provenance but are not a stable public API.

Rules for new scripts:

1. Never embed a user home directory or server-specific absolute path.
2. Locate the repository with `Path(__file__).resolve().parents[1]` in Python or
   `$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)` in Bash.
3. Use `CNN_MAMBA_PYTHON`, `UNIANN_ROOT`, and `ALTANN_ROOT` for external tools.
4. Put biological inputs at documented logical paths under `data/raw/`, or pass
   them as command-line/configuration arguments.
5. New reusable behavior belongs in `src/cnn_mamba/` with tests; scripts should
   only compose those public commands.
6. New UniAnn runs from CNN--Mamba four-state probabilities must preserve the
   exported TSV header and use `uniann-eval --require-cnn-mamba-tsv-header`.
   Never strip the header. Headerless emission scripts are historical controls,
   not the collaborator-facing interface.

For a portable multi-stage example, see
`configs/workflow.droso.example.json` and run:

```bash
cnn-mamba workflow configs/workflow.droso.example.json --dry-run
```
