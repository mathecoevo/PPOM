# PPOM

Propagule Pressure Occupancy Model: seasonal mean-field dynamics with stochastic
establishment, extended to a phylogenetic tree for ancestral-range inference.

The example uses the fixed tree `(((A,B),C),(D,E))`. Environmental ecotypes,
carrying capacities, dispersal and effective branch lengths are supplied.
MCMC samples shared dispersal histories with either a fixed root or an unknown
origin restricted to one of 27 regions.

## Install

```bash
git clone https://github.com/mathecoevo/PPOM.git
cd PPOM
python -m pip install .
```

Python 3.10 or newer is required. The first model call compiles the Numba kernels.

## Run

```bash
ppom forward --public examples/five_tip/public.npz --root-region 6 --output outputs/forward.npz
ppom fit --public examples/five_tip/public.npz \
  --observations examples/five_tip/observations.npz \
  --initialization examples/five_tip/initialization.npz \
  --cases original_A_f50_o0 original_A_f50_o500 \
  --chains 4 --workers 4 --steps 8000 --burn 2000 --output outputs/unknown
ppom report --output outputs/unknown
```

Add `--known-root-region 6` for the fixed-root control, using a separate output
directory. The paired cases retain 50 focal observations and add 500 observations
at each other tip. Evaluation truth is separate from inference inputs.

## Validation

```bash
python -m unittest discover -s tests -v
ppom validate --output outputs/finite_validation.json
```

See [Methods](docs/METHODS.md) and [Data](docs/DATA.md) for equations, inputs and
limitations. Full-landscape MCMC convergence is still under assessment.
