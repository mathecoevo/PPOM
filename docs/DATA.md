# Input data

All inputs use NumPy `.npz` archives and are read without pickle support.

| File | Contents |
|---|---|
| `public.npz` | Coordinates, carrying capacities, environmental ecotypes, root catalogue and effective lengths |
| `observations.npz` | Named cases and observed tip individual counts |
| `initialization.npz` | Public prior and discarded unknown-root training histories used only for starting states |
| `evaluation_only.npz` | Benchmark truth, used only by optional report evaluation |

Public arrays are `cells` (M × 2 integer coordinates), `K` (M), `state_blocks`
(M, values 0..26), `state_genotypes` (M, values 0..3), `shape` (2), and `times`
(8 strictly positive counts in A, B, AB, C, D, E, ABC, DE order). Optional arrays
preserve original catalogue metadata and calendar lengths.

Observations have `case` labels and `counts` of shape cases × 5 × M, tip order
A, B, C, D, E. Counts are nonnegative integers. `f5_o0` and `f50_o0` give focal
tip samples of 5 or 50; `f50_o500` adds 500 at each other tip while keeping the
same 50 focal observations. These are presence-only individual counts, not
presence/absence cell labels.

Initialization arrays are `masks`, `seeds`, `q` and `status`. Each mask selects
exactly one region. Each innovation row has sum(times)+4 seeds. Cached q values
are used to select compatible starts; the chosen history is replayed before
sampling. This pool is not a posterior approximation and contains no generating
seeds read from evaluation truth. The provided pool supports the six original
A/E sample-size cases; new datasets require compatible starting histories.

The fit command has no evaluation-file argument. Report evaluation is separate.
The current topology, four hard ecotypes and 27-region catalogue are explicit
implementation constraints; alternative trees or environmental responses need
model adaptation and validation.

