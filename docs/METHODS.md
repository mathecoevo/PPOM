# Methods

## State and environment

Each valid cell x has carrying capacity K_x and one of four non-overlapping
environmental ecotypes. The state n_x is the expected number of established
breeders of the locally viable ecotype. The occupancy fraction is phi_x = n_x/K_x.
The example landscape has 1535 valid cells and 27 candidate root regions.

Each breeder produces two offspring. An offspring stays with probability 0.98
or moves in a cardinal direction with probability 0.005 per direction.
Off-map arrivals are lost. Mutation occurs at rate 1e-7 between adjacent ecotypes;
at interior types this probability is divided between the two adjacent types.
Hard environmental selection removes incompatible arrivals.

Let W_dc be the probability that an offspring from c reaches d and is compatible
with d. Rows of W are destinations and columns are sources. Surviving arrival
probabilities are not renormalised after losses.

## Seasonal update

The fixed-two-offspring seasonal slot model is

```text
gamma_dc = 1 - (1 - W_dc/K_d)^2
phi'_d = 1 - product_c (1 - gamma_dc * phi_c)^K_c
n'_d = K_d * phi'_d
```

Established cells follow this update. It is a discrete seasonal recursion,
rather than a continuous-time ODE on tree branches.

This demographic law differs from the original tree simulator, which uses
Poisson offspring followed by carrying-capacity truncation. The examples labelled
`original` use observations from that simulator; `synthetic` labels identify
same-model validation data. They should be interpreted separately.

## Establishment of rare lineages

Vacant cells remain at zero until a lineage establishes. Competition against
the resident population gives

```text
Lambda_d = sum_c 2 * W_dc * n_c / K_d
s_d = (1 - exp(-Lambda_d))/Lambda_d, with s_d = 1 at Lambda_d = 0
```

For a founder of ecotype l at c, the clonal descendants' establishment probability
pi_c solves the branching-process fixed point

```text
a_c = sum_d Q_cd * (1 - mu) * s_d * pi_d
pi_c = 1 - (1 - a_c)^2
```

The sum is over cells compatible with the same ecotype. Subcritical and critical
components have pi = 0. Supercritical components use the positive survival
solution. The reference solver classifies components spectrally and uses Newton
iteration. The accelerated solver certifies relevant components with bounds,
eliminates the diagonal quadratic in a Gauss–Seidel update and checks residuals,
with a Newton fallback. Numerical failure raises an error.

The establishment intensity at vacant d is lambda_d = n'_d*pi_d. A hit is drawn
with probability 1-exp(-lambda_d). On a hit the population is seeded with
`min(1/max(pi_d,0.001), K_d)` individuals and subsequently follows the mean field.
This founder size is an approximation to the lineage size conditional on survival.
Established cells whose abundance drops below one become absent again.

The regional presence resolution here is one valid cell per region. The 27
regions used to define possible root origins are a separate catalogue.

## Root and lineage splits

An unknown origin is uniform over exactly one of the 27 candidate regions.
Initial abundance is K_x within the selected region and zero outside. The
origin's within-region abundance is fixed, not independently estimated.

The root splits into ABC and DE; subsequent splits are ABC into AB/C, AB into
A/B and DE into D/E. Disconnected occupied components are allocated in decreasing
mass order to balance descendant abundance. A single connected range is divided
geographically using two spatial endpoints, with random tie breaking. The
single-cell and empty-descendant cases retain the specified fallback rule.
At every split, n_left(x)+n_right(x)=n_parent(x). Descendants inherit absolute
abundance and established status without renormalisation or resampling.

## Effective branch clock

A branch tick is a season in which at least one cell never previously occupied
by that lineage becomes occupied. Multiple new cells in one season give one
tick; recolonisation of a previously visited cell gives none. Each branch stops
at the first attainment of its specified tick count. Calendar duration is latent.

The branch order A, B, AB, C, D, E, ABC, DE has counts
138, 328, 187, 438, 173, 66, 258, 320. These are neither calendar seasons nor
the number of individual dispersal attempts. A calendar endpoint with an
unobserved terminal period after the last tick cannot be identified from these
counts alone.

After ten consecutive stable resident updates, with maximum abundance change
below 1e-9 and no pending threshold extinction, the resident background can be
frozen. A geometric wait to the next establishment and a conditional Poisson
hit replace repeated empty seasons. Freezing a near-stationary background is
an approximation; it is not used to skip arbitrary changing dynamics.

## Observations and inference

Given one complete shared history H, tip probabilities are relative abundances:

```text
q_i(x) = n_i(x)/sum_x n_i(x)
log L(H) = sum_i sum_x C_i(x)*log(q_i(x))
```

C_i(x) is the observed count of sampled individuals. The multinomial constant
does not affect inference. An observed cell with q=0 gives zero likelihood.
Unobserved tips contribute no likelihood term. Tip observations are conditional
on the same ancestral history; their marginal distributions after integrating
that history are dependent.

Independent uniform 31-bit innovations parameterise each event interval and
the four splits. For the example there are 1912 seeds. Root and innovations
determine a replayable history. Local changes preserve the earlier prefix and
recompute the affected descendant subtree.

The MCMC target is proportional to prior(root)*prior(innovations)*L(H).
Proposals refresh the root and full history, change the origin alone, edit local
interval/split seeds, or use conditional importance resampling on an observed
tip branch. This last kernel includes the current history and seven draws from
the conditional branch prior. A known-root fit fixes the origin and samples
only innovations.

An auxiliary replica uses beta=0.05 and epsilon=0.001. The formal cold replica
has beta=1 and epsilon=0. Updates select the cold replica with probability 0.75
and the hot one with probability 0.25; exchanges are attempted with probability
0.25 using the two targets' correct acceptance ratio. Only cold samples are
reported. Training-based support search is discarded before collection and
does not read evaluation truth or known-root fits for unknown-root initialisation.

## Summaries and interpretation

Posterior tip and ancestor maps average cold histories. Diagnostics monitor
root indicators and tip region masses across independent chains, using rank
split Rhat, bulk ESS and disagreement in predictive means. A constant unknown
root is treated as unconfirmed, since concentration and poor mixing can look
similar. Summary checks do not establish convergence of every latent innovation.

Other tips can improve reconstruction by constraining their shared origin and
ancestral history. Paired sample-size comparisons keep the focal observations
unchanged when adding other tips. Recovery is evaluated by relative-abundance
overlap, sum_x min(prediction_x,truth_x), which equals 1-TV. It is not ROC AUC.

The largest conditional likelihood visited by a chain is not a marginal root
MLE. Marginal likelihood for a root requires integrating L(H) over its innovation
prior. This implementation reports posterior samples and does not claim a
global marginal maximum-likelihood fit.

