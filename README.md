# Propagule Pressure Occupancy Model (PPOM)

PPOM combines seasonal mean-field population dynamics with stochastic
establishment of rare lineages. This repository implements the model on a
five-tip phylogenetic tree and uses tip observations to reconstruct ancestral
ranges and dispersal histories.

Established populations follow a deterministic seasonal update. Colonisation
of vacant cells is sampled using branching-process establishment probabilities.
At lineage splits, descendants inherit geographically partitioned populations.
Inference samples a shared history for the whole tree, so observations from one
tip can inform the ancestral states inherited by other tips.

The example tree is `(((A,B),C),(D,E))`. Environmental suitability, carrying
capacity, dispersal, mutation and effective branch lengths are supplied.
The root can be fixed or inferred as one of 27 candidate regions. The present
implementation does not estimate tree topology, branch lengths or environmental
regression coefficients.

## Installation

```bash
git clone https://github.com/mathecoevo/PPOM.git
cd PPOM
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Use Python 3.10 or newer with a compatible Numba release. The first forward call
includes compilation of the numerical kernels.

## Example

A complete forward history from a specified origin:

```bash
ppom forward --public examples/five_tip/public.npz \
  --root-region 6 --output outputs/forward.npz
```

Unknown-root inference from observed individuals:

```bash
ppom fit --public examples/five_tip/public.npz \
  --observations examples/five_tip/observations.npz \
  --initialization examples/five_tip/initialization.npz \
  --cases original_A_f50_o0 original_A_f50_o500 \
  --chains 4 --workers 4 --steps 8000 --burn 2000 \
  --output outputs/unknown
```

The first case has 50 observations at A; the second uses the same observations
at A and adds 500 at each other tip. Samples are multinomial draws proportional
to abundance. For a known-root control, add `--known-root-region 6` and use a
different output directory. That index is not supplied to unknown-root fits.
Use `--cases original_E_f50_o0 original_E_f50_o500` for the corresponding E comparison.

Replay saved histories and check independent-chain agreement:

```bash
ppom audit --public examples/five_tip/public.npz \
  --observations examples/five_tip/observations.npz --output outputs/unknown
ppom report --output outputs/unknown
```

For benchmark evaluation, append
`--evaluation examples/five_tip/evaluation_only.npz` to the report command.
Inference does not read that file. Re-running a completed fit with a higher
`--steps` value extends its saved RNG checkpoint; retain the original burn or
omit `--burn` when extending.

## Model and validation

[Methods](docs/METHODS.md) describes the seasonal equations, effective clock,
splitting rule, observation model and MCMC. A [Chinese algorithm guide](docs/ALGORITHM.zh-CN.md)
provides additional implementation detail. Input arrays are described in
[Data](docs/DATA.md).

```bash
python -m unittest discover -s tests -v
ppom validate --output outputs/finite_validation.json
```

Validation covers conservation at splits, likelihood support, prefix replay,
known-root proposals, and the actual unknown-root proposal kernel against an
exactly summable posterior. A stored 24-cell spatial-tree enumeration provides
a separate finite reference. Migration checks confirm identical forward maps
and event counts to the working implementation for fixed innovations.

The larger single-region experiment is still being assessed for convergence.
Completing a chain is not evidence that its posterior has converged. Reported
map overlap is `1 - total variation`, not ROC AUC. The highest visited
conditional likelihood is not a global marginal maximum-likelihood estimate.

## Contents

```text
src/seasonal_tree/   Forward model, latent-history sampler and command line tools
tests/              Core numerical and inference checks
examples/five_tip/  Public landscape, observations and discarded starting histories
docs/               Methods and input specification
validation/         Finite reference and validation records
```

The regional hybrid core was supplied by Xia Hua. Tree adaptation and inference
were developed by Aobo Wang. See [NOTICE](NOTICE) for source attribution.

