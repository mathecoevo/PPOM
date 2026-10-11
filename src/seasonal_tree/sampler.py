"""Cold posterior and auxiliary hot replica; initialization is discarded."""
from pathlib import Path
import json
import time
import numpy as np
from .data import load_public, load_observations, sha256, TIP_EDGES
from .proposals import SingleRootTree, propose
from .likelihood import likelihood, q_tip

BETAS = (1., .05)
EPSILONS = (0., .001)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def make_model(public_path):
    return SingleRootTree(data=load_public(public_path))


def initialize(model, counts, bank_path, rng, mode, known_region, trials=2000):
    with np.load(bank_path, allow_pickle=False) as z:
        roots, innovations, q = z['masks'], z['seeds'], z['q']
        valid = (z['status'] == 0)
        valid &= np.array([0 < int(r) < (1 << 27) and int(r).bit_count() == 1 for r in roots])
        if mode == 'known':
            valid &= roots == (1 << known_region)
        ix = np.flatnonzero(valid)
        if not len(ix) or innovations.shape[1] != model.seed_count:
            raise ValueError('No compatible one-region starting history in initialization bank')
        missing = np.sum((q[ix] <= 0) * counts, axis=(1, 2))
        scores = np.array([likelihood(p, counts, 1e-7) for p in q[ix]])
        ranked = ix[np.lexsort((-scores, missing))]
        feasible = {}
        for j in ranked:
            if np.all(q[j][counts > 0] > 0):
                feasible.setdefault(int(roots[j]), int(j))
        pool = list(feasible.values()) or list(ranked[:24])
        j = int(rng.choice(pool)); root = int(roots[j]); u = innovations[j].copy()
    h = model.run_noise(root, u)
    if h[3]:
        raise RuntimeError('Initialization history failed replay')
    p = q_tip(h)
    state = (root, u, h, p)
    best_cost = (int(counts[p <= 0].sum()), -likelihood(p, counts, 1e-7))
    best = state
    for step in range(trials + 1):
        if best_cost[0] == 0:
            return best, dict(bank_index=j, discarded_support_trials=step,
                              feasible_start_regions=[r.bit_length() - 1 for r in feasible],
                              missing_observed_individuals=0)
        if step == trials:
            break
        root, u, h, p = state
        candidate, _ = propose(model, root, u, h, rng, mode, counts, 1e-7)
        if candidate is None:
            continue
        r, v, g, q, _, _ = candidate
        cost = (int(counts[q <= 0].sum()), -likelihood(q, counts, 1e-7))
        current = (int(counts[p <= 0].sum()), -likelihood(p, counts, 1e-7))
        if cost < current or rng.random() < .02:
            state = (r, v, g, q)
        if cost < best_cost:
            best_cost, best = cost, (r, v.copy(), g, q.copy())
    raise RuntimeError(f'No positive cold likelihood found; {best_cost[0]} observations unsupported. '
                       'Increase the prior bank/support search; do not add a likelihood floor.')


def run_chain(public_path, observations_path, bank_path, case, output, chain=0,
              steps=8000, burn=None, support_trials=2000, known_region=None,
              random_seed=202610107200):
    """Save cold samples only. Resume completed checkpoints with exact RNG state."""
    if steps <= 0 or support_trials < 0:
        raise ValueError('Positive steps and nonnegative support trials are required')
    if known_region is not None and not 0 <= known_region < 27:
        raise ValueError('Known root region must lie in 0..26')
    mode = 'unknown' if known_region is None else 'known'
    out = Path(output) / f'{case}_{mode}' / f'chain{chain}'
    out.mkdir(parents=True, exist_ok=True)
    model = make_model(public_path)
    counts = load_observations(observations_path, case, len(model.K))
    identity = dict(public=sha256(public_path), observations=sha256(observations_path),
                    initialization=sha256(bank_path), case=case, mode=mode,
                    known_root_region=known_region, chain=chain, random_seed=random_seed,
                    code={p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')})
    identity_path = out / 'inputs.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError('Existing run uses different inputs/settings/code; use a new output directory')
    else:
        atomic_json(identity_path, identity)
    previous = json.loads((out / 'summary.json').read_text()) if (out / 'summary.json').exists() else None
    if previous and previous['posterior_steps'] >= steps:
        return previous
    continuing = previous is not None and (out / 'continuation.npz').exists()
    burn = previous['burn'] if continuing and burn is None else (steps // 4 if burn is None else burn)
    if not 0 <= burn < steps or (continuing and burn != previous['burn']):
        raise ValueError('Require 0 <= burn < steps; resumed runs must preserve the original burn')
    rng = np.random.default_rng(random_seed + chain * 100003)
    begin = time.perf_counter()
    draws, roots, regions, seeds, trace = [], [], [], [], []
    attempt = np.zeros((2, 3), int); accept = np.zeros_like(attempt)
    swaps = swap_attempt = offset = 0
    ancestor_sum = np.zeros((3, len(model.K)))
    best_ll = -np.inf; best_state = None
    if continuing:
        with np.load(out / 'continuation.npz', allow_pickle=False) as checkpoint:
            rng.bit_generator.state = json.loads(str(checkpoint['rng_state_json']))
            state = []
            for r, u in zip(checkpoint['roots'], checkpoint['seeds']):
                h = model.run_noise(int(r), u)
                if h[3]:
                    raise RuntimeError('Checkpoint failed replay')
                state.append((int(r), u.copy(), h, q_tip(h)))
        with np.load(out / 'posterior.npz', allow_pickle=False) as z:
            draws, roots, regions, seeds, trace = [list(z[k]) for k in ('q', 'roots', 'region_mass', 'seeds', 'trace')]
            ancestor_sum = z['ancestor_mean'].astype(float) * len(draws)
            best_state = (int(z['best_root']), z['best_seeds'].copy(), z['best_q'].copy(), z['best_starts'].copy(), z['best_ends'].copy())
        best_ll = previous['highest_visited_conditional_log_likelihood']
        offset = previous['posterior_steps']; info = previous['initialization']
        attempt = np.array([previous['attempt'], previous['hot_attempt']])
        accept = np.array([previous['accept'], previous['hot_accept']])
        swaps, swap_attempt = previous['swaps'], previous['swap_attempts']
    else:
        initial, info = initialize(model, counts, bank_path, rng, mode, known_region, support_trials)
        r, u, h, q = initial
        state = [(r, u, h, q), (r, u.copy(), h, q.copy())]
    ll = [BETAS[k] * likelihood(state[k][3], counts, EPSILONS[k]) for k in range(2)]
    if not np.isfinite(ll[0]):
        raise RuntimeError('Cold initialization has zero original likelihood')
    for step in range(offset, steps):
        k = 0 if rng.random() < .75 else 1
        r, u, h, q = state[k]
        candidate, kind = propose(model, r, u, h, rng, mode, counts * BETAS[k], EPSILONS[k])
        attempt[k, kind] += 1
        if candidate is not None:
            r, v, g, p, l, correction = candidate
            if np.isfinite(l) and np.log(rng.random()) < min(0., l - ll[k] + correction):
                state[k] = (r, v, g, p); ll[k] = l; accept[k, kind] += 1
        if rng.random() < .25:
            swap_attempt += 1
            a = likelihood(state[1][3], counts)
            b = BETAS[1] * likelihood(state[0][3], counts, EPSILONS[1])
            if np.isfinite(a) and np.log(rng.random()) < min(0., a + b - sum(ll)):
                state = [state[1], state[0]]; ll = [a, b]; swaps += 1
        r, u, h, q = state[0]
        if not np.isfinite(ll[0]):
            raise RuntimeError('Cold chain left the original likelihood support')
        if ll[0] > best_ll:
            best_ll = ll[0]
            best_state = (r, u.copy(), q.copy(), h[0].copy(), h[1].copy())
        trace.append((ll[0], r, ll[1]))
        if step >= burn:
            draws.append(q.astype(np.float32)); roots.append(r); seeds.append(u.copy())
            regions.append(np.array([q[:, model.data['state_blocks'] == b].sum(1) for b in range(27)]).T)
            a = h[1][[6, 2, 7]]; ancestor_sum += a / a.sum(1)[:, None]
        if step % 100 == 99:
            atomic_json(out / 'progress.json', dict(phase='posterior', iteration=step + 1,
                planned=steps, root=int(r), cold_ll=float(ll[0]),
                cold_accept=accept[0].tolist(), replica_swaps=swaps, seconds=time.perf_counter() - begin))
    np.savez_compressed(out / 'continuation.npz', roots=np.array([s[0] for s in state]),
        seeds=np.array([s[1] for s in state]), rng_state_json=np.array(json.dumps(rng.bit_generator.state)))
    np.savez_compressed(out / 'posterior.npz', q=np.array(draws), roots=np.array(roots, dtype=np.uint32),
        region_mass=np.array(regions), seeds=np.array(seeds), trace=np.array(trace),
        ancestor_mean=ancestor_sum / len(draws), best_root=np.array(best_state[0]),
        best_seeds=best_state[1], best_q=best_state[2], best_starts=best_state[3], best_ends=best_state[4])
    summary = dict(case=case, mode=mode, chain=chain, posterior_completed=True,
        observations_per_tip=counts.sum(1).tolist(), posterior_steps=steps, burn=burn,
        saved_draws=len(draws), attempt=attempt[0].tolist(), accept=accept[0].tolist(),
        hot_attempt=attempt[1].tolist(), hot_accept=accept[1].tolist(), swaps=swaps,
        swap_attempts=swap_attempt, hot_beta=.05, hot_epsilon=.001, cold_epsilon=0.,
        known_root_region=known_region, environment_and_branch_lengths_fixed=True,
        highest_visited_conditional_log_likelihood=float(best_ll),
        highest_likelihood_is_not_marginal_parameter_MLE=True, initialization=info,
        resumed_from_completed_checkpoint=continuing, resumed_at_iteration=offset,
        convergence_not_established_by_completion=True, model=model.manifest(),
        seconds=time.perf_counter() - begin)
    atomic_json(out / 'summary.json', summary)
    return summary

