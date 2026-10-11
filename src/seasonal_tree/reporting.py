"""Diagnostics and optional truth-based evaluation, separate from inference."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import rankdata, norm
from .sampler import make_model, atomic_json
from .data import load_observations, sha256
from .likelihood import q_tip, likelihood
from .diagnostics import split_rank_rhat, ess


def summarize_group(x):
    x = np.asarray(x)
    flat = x.reshape(x.shape[:2] + (-1,))
    active = np.ptp(flat.reshape(-1, flat.shape[-1]), axis=0) > 1e-10
    if x.shape[0] < 2 or x.shape[1] < 8 or not active.any():
        return dict(max_rhat=None, min_bulk_ess=None, active=int(active.sum()))
    rh, effective = [], []
    for i in np.flatnonzero(active):
        a = flat[:, :, i]; rh.append(split_rank_rhat(a))
        n = a.shape[1] // 2
        split = np.concatenate((a[:, :n], a[:, -n:]), axis=0)
        ranks = norm.ppf((rankdata(split.ravel()).reshape(split.shape) - .375) / (split.size + .25))
        effective.append(ess(ranks))
    maximum = float(max(rh))
    return dict(max_rhat=maximum if np.isfinite(maximum) else None,
                min_bulk_ess=float(min(effective)), active=int(active.sum()))


def audit(public, observations, output):
    model = make_model(public); records = []
    for file in sorted(Path(output).glob('*_*/chain*/posterior.npz')):
        summary = json.loads(file.with_name('summary.json').read_text())
        identity = json.loads(file.with_name('inputs.json').read_text())
        if identity['public'] != sha256(public) or identity['observations'] != sha256(observations):
            raise ValueError('Audit input hashes differ from fit inputs')
        counts = load_observations(observations, summary['case'], len(model.K))
        with np.load(file, allow_pickle=False) as z:
            roots = z['roots']; q = z['q']
            assert np.all((roots > 0) & ((roots & (roots - 1)) == 0) & (roots < (1 << 27)))
            if summary['mode'] == 'known':
                assert np.all(roots == (1 << summary['known_root_region']))
            assert np.all(q[:, counts > 0] > 0)
            assert np.max(np.abs(q.sum(2) - 1)) < 1e-6
            checks = []
            for j in sorted(set((0, len(roots) // 2, len(roots) - 1))):
                h = model.run_noise(int(roots[j]), z['seeds'][j]); assert h[3] == 0
                expected = q_tip(h)
                np.testing.assert_allclose(expected, q[j], atol=1e-7, rtol=0)
                assert abs(likelihood(expected, counts) - z['trace'][j + summary['burn'], 0]) < 1e-9
                np.testing.assert_array_equal(h[2][:, 0], model.data['times'])
                checks.append(j)
            best = model.run_noise(int(z['best_root']), z['best_seeds']); assert best[3] == 0
            np.testing.assert_allclose(q_tip(best), z['best_q'], atol=1e-12, rtol=0)
            records.append(dict(case=summary['case'], chain=summary['chain'], mode=summary['mode'],
                                replayed_draws=checks, all_observed_probabilities_positive=True))
    if not records:
        raise ValueError('No completed posterior files to audit')
    result = dict(passed=True, completed_chains=len(records), checks=records)
    atomic_json(Path(output) / 'replay_audit.json', result)
    return result


def report(output, evaluation=None):
    groups = {}; means = []; labels = []; records = []
    for file in sorted(Path(output).glob('*_*/chain*/posterior.npz')):
        if file.with_name('summary.json').exists():
            groups.setdefault(file.parent.parent.name, []).append(file)
    truth = None
    if evaluation:
        with np.load(evaluation, allow_pickle=False) as z:
            truth = {k: z[k].copy() for k in z.files}
    for label, files in groups.items():
        zs = [np.load(f, allow_pickle=False) for f in files]
        summaries = [json.loads(f.with_name('summary.json').read_text()) for f in files]
        n = min(len(z['roots']) for z in zs)
        q = np.array([z['q'][-n:].astype(float) for z in zs])
        mean = q.mean((0, 1)); mean /= mean.sum(1)[:, None]
        regions = np.array([z['region_mass'][-n:] for z in zs])
        roots = np.array([z['roots'][-n:] for z in zs])
        bits = ((roots[:, :, None] >> np.arange(27)) & 1).astype(float)
        moves = [int(np.sum(np.diff(row) != 0)) for row in roots]
        rd = summarize_group(bits); pd = summarize_group(regions)
        tv = max(float(np.max(abs(a.mean(0) - b.mean(0)).sum(1) / 2)) for a in q for b in q)
        known = summaries[0]['mode'] == 'known'
        root_checked = known or (rd['max_rhat'] is not None and rd['max_rhat'] < 1.01
            and rd['min_bulk_ess'] >= 400 and all(m > 0 for m in moves) and sum(moves) >= 10)
        passed = bool(len(files) >= 4 and pd['max_rhat'] is not None and pd['max_rhat'] < 1.01
            and pd['min_bulk_ess'] >= 400 and tv < .05 and root_checked)
        rec = dict(label=label, completed_chains=len(files), retained_per_chain=n,
            all_tip_region_diagnostics=pd, root_diagnostics=None if known else rd,
            root_changes_per_chain=moves, max_chain_mean_TV=tv, summary_checks_passed=passed,
            summary_checks_do_not_prove_full_latent_history_convergence=True,
            constant_unknown_root_is_not_convergence_proof=not known and not all(moves),
            sampled_root_region_frequencies=bits.mean((0, 1)).tolist())
        if truth is not None:
            case = summaries[0]['case']; dataset = case.split('_', 1)[0]
            actual = truth[dataset + '_q']
            rec['tip_relative_abundance_overlaps'] = np.minimum(mean, actual).sum(1).tolist()
            rec['actual_root_frequency_evaluation_only'] = float(np.mean(roots == int(truth['root_mask'])))
            rec['overlap_is_not_ROC_AUC'] = True
        records.append(rec); labels.append(label); means.append(mean)
        for z in zs:
            z.close()
    if not records:
        raise ValueError('No completed chains to report')
    atomic_json(Path(output) / 'diagnostics.json', records)
    np.savez_compressed(Path(output) / 'posterior_means.npz', labels=np.array(labels), means=np.array(means))
    return records

