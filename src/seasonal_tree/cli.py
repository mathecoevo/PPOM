"""Portable command line entry points."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import numpy as np
from .sampler import make_model, run_chain, atomic_json
from .likelihood import q_tip


def main(argv=None):
    parser = argparse.ArgumentParser(prog='ppom')
    sub = parser.add_subparsers(dest='command', required=True)
    forward = sub.add_parser('forward', help='One complete shared forward history')
    forward.add_argument('--public', required=True)
    forward.add_argument('--root-region', required=True, type=int)
    forward.add_argument('--seed', default=2026101101, type=int)
    forward.add_argument('--output', required=True)
    fit = sub.add_parser('fit', help='Known-root or single-region unknown-root MCMC')
    for name in ('public', 'observations', 'initialization', 'output'):
        fit.add_argument('--' + name, required=True)
    fit.add_argument('--cases', nargs='+')
    fit.add_argument('--known-root-region', type=int)
    fit.add_argument('--chains', default=4, type=int)
    fit.add_argument('--workers', default=4, type=int)
    fit.add_argument('--steps', default=8000, type=int)
    fit.add_argument('--burn', type=int)
    fit.add_argument('--support-trials', default=2000, type=int)
    fit.add_argument('--seed', default=202610107200, type=int)
    aud = sub.add_parser('audit', help='Replay saved cold histories and likelihoods')
    for name in ('public', 'observations', 'output'):
        aud.add_argument('--' + name, required=True)
    rep = sub.add_parser('report', help='Chain diagnostics; optional separate truth evaluation')
    rep.add_argument('--output', required=True)
    rep.add_argument('--evaluation')
    val = sub.add_parser('validate', help='Actual proposal kernel against exact finite posterior')
    val.add_argument('--output', default='outputs/finite_validation.json')
    args = parser.parse_args(argv)
    if args.command == 'forward':
        if not 0 <= args.root_region < 27:
            parser.error('root region must be 0..26')
        model = make_model(args.public)
        u = np.random.default_rng(args.seed).integers(2**31 - 1, size=model.seed_count)
        h = model.run_noise(1 << args.root_region, u)
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, starts=h[0], ends=h[1], stats=h[2], status=h[3], seeds=u,
                            root_mask=1 << args.root_region, q=q_tip(h) if h[3] == 0 else np.zeros((5, len(model.K))))
        atomic_json(path.with_suffix('.json'), dict(status=int(h[3]), model=model.manifest(),
            failed_history_is_not_a_numerical_error=True))
        print(json.dumps(dict(status=int(h[3]), output=str(path))))
    elif args.command == 'fit':
        if args.chains < 1 or args.workers < 1:
            parser.error('positive chains and workers required')
        with np.load(args.observations, allow_pickle=False) as z:
            all_cases = list(map(str, z['case']))
        cases = args.cases or [c for c in all_cases if c.startswith('original_') and c.endswith(('f5_o0', 'f50_o0', 'f50_o500'))]
        if not cases:
            parser.error('supply --cases for this observations file')
        jobs = []
        for case in cases:
            if case not in all_cases:
                parser.error(f'Unknown case {case}')
            for chain in range(args.chains):
                jobs.append(dict(public_path=args.public, observations_path=args.observations,
                    bank_path=args.initialization, case=case, output=args.output, chain=chain,
                    steps=args.steps, burn=args.burn, support_trials=args.support_trials,
                    known_region=args.known_root_region, random_seed=args.seed + all_cases.index(case) * 5011
                    + (args.known_root_region is None) * 100000007))
        if args.workers == 1:
            for job in jobs:
                result = run_chain(**job)
                print(json.dumps(dict(case=result['case'], chain=result['chain'], saved=result['saved_draws'])), flush=True)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(run_chain, **job) for job in jobs]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    print(json.dumps(dict(case=result['case'], chain=result['chain'], saved=result['saved_draws'])), flush=True)
    elif args.command == 'audit':
        from .reporting import audit
        print(json.dumps(audit(args.public, args.observations, args.output), indent=2))
    elif args.command == 'report':
        from .reporting import report
        print(json.dumps(report(args.output, args.evaluation), indent=2))
    else:
        from .validation import validate
        print(json.dumps(validate(args.output), indent=2))


if __name__ == '__main__':
    main()

