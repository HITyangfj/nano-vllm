"""Repeat identical traces in fresh processes; retain every run, mean and stddev."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--original-repo', required=True)
    parser.add_argument('--trace', required=True)
    parser.add_argument('--budgets', nargs='+', type=int, default=[512, 1024, 2048])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--kv-blocks', type=int, default=256)
    parser.add_argument('--max-num-seqs', type=int, default=32)
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--cache-state', choices=['cold', 'warm'], default='cold')
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--output-dir', default='results/matrix')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.repeats < 2 or min(args.budgets) <= 0:
        parser.error('Use at least two repetitions and positive budgets')
    here = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    groups = {}
    for budget in args.budgets:
        for repeat in range(args.repeats):
            # Rotate order to reduce a consistent thermal/order bias.
            modes = ['original', 'legacy', 'mixed']
            modes = modes[repeat % 3:] + modes[:repeat % 3]
            for mode in modes:
                dest = output_dir / f'{mode}_budget{budget}_run{repeat}.json'
                cmd = [sys.executable, str(here / 'bench_mixed.py'), '--model', args.model,
                       '--trace', args.trace, '--mode', mode, '--budget', str(budget),
                       '--repo', args.original_repo if mode == 'original' else str(here),
                       '--kv-blocks', str(args.kv_blocks), '--max-num-seqs', str(args.max_num_seqs),
                       '--max-model-len', str(args.max_model_len), '--cache-state', args.cache_state,
                       '--output', str(dest)]
                if args.eager:
                    cmd.append('--eager')
                if args.dry_run:
                    print(json.dumps(cmd))
                    continue
                subprocess.run(cmd, check=True)
                result = json.loads(dest.read_text(encoding='utf-8'))
                groups.setdefault((mode, budget), []).append(result)
    if args.dry_run:
        return
    # Refuse summaries if the supposedly shared workload/environment differed.
    runs = [run for group in groups.values() for run in group]
    for key in ['trace_sha256', 'model_manifest', 'environment']:
        if any(run[key] != runs[0][key] for run in runs):
            raise ValueError(f'Incomparable runs: {key} differs')
    summary = []
    for (mode, budget), group in groups.items():
        for metric in group[0]['summary']:
            values = [r['summary'][metric] for r in group]
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
                continue
            summary.append(dict(mode=mode, budget=budget, metric=metric, repeats=len(values),
                                mean=statistics.mean(values), stddev=statistics.stdev(values),
                                minimum=min(values), maximum=max(values)))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'aggregate.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (output_dir / 'aggregate.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == '__main__':
    main()
