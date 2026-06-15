"""
Entry point for HeteroNVTDrivenChain dividing-droplet runs.

Run `python main.py` and choose between:
  - a single interactive run (prompts for all parameters), or
  - a batch run from a params .txt file (sweeps one parameter).

See simulation.py for the underlying engine (RunParams,
make_seed_geometry, run_chunked_simulation) and cli.py for the
interactive/batch logic and prompts.
"""
from __future__ import annotations

from cli import prompt_yes_no, run_from_params_file, run_interactive


def main() -> None:
    print("=== HeteroNVT Droplet Runner ===\n")

    if prompt_yes_no("Pipe parameters from a .txt file?", default=False):
        run_from_params_file()
    else:
        run_interactive()


if __name__ == "__main__":
    main()