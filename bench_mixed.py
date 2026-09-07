"""Wall-clock arrival replay, shared by original, legacy and mixed engines.

All times/latencies are engine-side. Run each mode/repetition in a fresh process.
The original checkout is imported with --repo; its source is never modified.
"""
import argparse
import atexit
import csv
import hashlib
import importlib
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path


def make_trace(scenario, seed=0, count=32, injection_at=0.2):
    rng = random.Random(seed)
    rows = []
    if scenario == 'offline':
        # Original bench.py's length distribution, with configurable request count.
        lengths = [rng.randint(100, 1024) for _ in range(count)]
        prompts = [[rng.randint(0, 10000) for _ in range(n)] for n in lengths]
        for i, prompt in enumerate(prompts):
            rows.append(dict(id=i, arrival_s=0.0, prompt_token_ids=prompt,
                             max_tokens=rng.randint(100, 1024)))
    else:
        for i in range(count):
            if scenario == 'injection':
                arrival = 0.0 if i < count-1 else injection_at
                length, output = (32, 256) if i < count-1 else (2048, 32)
            else:
                arrival = (i // 4) * 0.15
                length, output = (2048, 32) if i % 4 == 3 else (64, 128)
            rows.append(dict(id=i, arrival_s=arrival,
                             prompt_token_ids=[rng.randint(0, 10000) for _ in range(length)],
                             max_tokens=output))
    return {'scenario': scenario, 'seed': seed, 'requests': rows}


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    index = (len(values)-1) * p / 100
    low = int(index)
    return values[low] + (values[min(low+1, len(values)-1)]-values[low]) * (index-low)


def summarize(records, steps, elapsed):
    ttft, tpots, itls = [], [], []
    for r in records:
        times = r['token_timestamps_s']
        if times:
            ttft.append(times[0] - r['actual_enqueue_s'])
        if len(times) > 1:
            gaps = [b-a for a,b in zip(times, times[1:])]
            itls.extend(gaps)
            tpots.append(statistics.mean(gaps))
    count = sum(len(r['output_token_ids']) for r in records)
    return dict(elapsed_s=elapsed, output_tokens=count,
                output_tokens_per_s=count/elapsed, requests_per_s=len(records)/elapsed,
                mean_ttft_s=statistics.mean(ttft) if ttft else None,
                p95_ttft_s=percentile(ttft, 95),
                mean_tpot_s=statistics.mean(tpots) if tpots else None,
                p50_itl_s=percentile(itls, 50), p95_itl_s=percentile(itls, 95),
                p99_itl_s=percentile(itls, 99), itl_samples=len(itls),
                prefill_tokens=sum(s['prefill_tokens'] for s in steps),
                decode_tokens=sum(s['decode_tokens'] for s in steps),
                recomputed_tokens=sum(s['recomputed_tokens'] for s in steps),
                preemptions=sum(s['preemptions'] for s in steps),
                peak_used_kv_blocks=max((s['used_kv_blocks'] for s in steps), default=0),
                mixed_steps=sum(bool(s['prefill_tokens'] and s['decode_tokens']) for s in steps),
                itl_aggregation='all consecutive output-token intervals pooled across requests',
                tpot_aggregation='unweighted mean of per-request mean ITL, excluding <2 outputs')


def original_fixed_pool(blocks):
    """Baseline-only adapter: fix physical capacity without changing scheduling."""
    import torch
    module = importlib.import_module('nanovllm.engine.model_runner')

    def allocate(runner):
        cfg = runner.config
        hf = cfg.hf_config
        cfg.num_kvcache_blocks = blocks
        heads = hf.num_key_value_heads // runner.world_size
        dim = getattr(hf, 'head_dim', hf.hidden_size // hf.num_attention_heads)
        runner.kv_cache = torch.empty(2, hf.num_hidden_layers, blocks,
                                      cfg.kvcache_block_size, heads, dim)
        layers = [m for m in runner.model.modules() if hasattr(m, 'k_cache') and hasattr(m, 'v_cache')]
        for i, layer in enumerate(layers):
            layer.k_cache, layer.v_cache = runner.kv_cache[0, i], runner.kv_cache[1, i]
    module.ModelRunner.allocate_kv_cache = allocate


def observe_original(llm, records, steps, origin):
    """Capture CPU-ready tokens around original postprocess (partial chunks excluded)."""
    scheduler = llm.scheduler
    postprocess, preempt = scheduler.postprocess, scheduler.preempt
    high_water = {}
    preemptions = [0]

    def count_preempt(seq):
        preemptions[0] += 1
        return preempt(seq)

    def observe(seqs, tokens, is_prefill):
        now = time.perf_counter() - origin
        counts = {s.seq_id: s.num_completion_tokens for s in seqs}
        computed = sum(s.num_scheduled_tokens for s in seqs)
        recomputed = 0
        for seq in seqs:
            start, end = seq.num_cached_tokens, seq.num_cached_tokens + seq.num_scheduled_tokens
            recomputed += max(0, min(end, high_water.get(seq.seq_id, 0))-start)
            high_water[seq.seq_id] = max(end, high_water.get(seq.seq_id, 0))
        used = len(scheduler.block_manager.used_block_ids)
        postprocess(seqs, tokens, is_prefill)
        output_count = 0
        for seq in seqs:
            if seq.num_completion_tokens > counts[seq.seq_id]:
                record = records[seq.seq_id]
                record['token_timestamps_s'].append(now)
                record['output_token_ids'].append(seq.last_token)
                output_count += 1
                if seq.is_finished:
                    record['finished_s'] = now
        steps.append(dict(available_s=now, prefill_tokens=computed if is_prefill else 0,
                          decode_tokens=0 if is_prefill else computed, output_tokens=output_count,
                          recomputed_tokens=recomputed, preemptions=preemptions[0], used_kv_blocks=used))
        preemptions[0] = 0
    scheduler.preempt, scheduler.postprocess = count_preempt, observe


def git_info(repo):
    def git(*args):
        result = subprocess.run(['git', '--git-dir=' + str(repo / '.git'), '--work-tree=' + str(repo), *args],
                                capture_output=True, text=True, encoding='utf-8', errors='replace')
        return result.stdout.strip() if result.returncode == 0 else 'unavailable: ' + result.stderr.strip()
    code = hashlib.sha256()
    for path in sorted((repo / 'nanovllm').rglob('*.py')):
        code.update(str(path.relative_to(repo)).replace('\\', '/').encode())
        code.update(path.read_bytes())
    return {'revision': git('rev-parse', 'HEAD'), 'status': git('status', '--short'),
            'source_sha256': code.hexdigest(),
            'diff_sha256': hashlib.sha256(git('diff', 'HEAD').encode()).hexdigest()}


def replay(args, trace):
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.block_manager import BlockManager
    from nanovllm.config import Config
    original = args.mode == 'original'
    modern = 'enable_mixed_batching' in Config.__dataclass_fields__
    if original == modern:
        raise ValueError('Use --mode original with the original checkout, legacy/mixed with the modified checkout')
    if original:
        original_fixed_pool(args.kv_blocks)
    rows = sorted(trace['requests'], key=lambda r: r['arrival_s'])
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Trace needs nonempty, uniquely identified requests')
    for row in rows:
        if row['arrival_s'] < 0 or not row['prompt_token_ids'] or row['max_tokens'] <= 0:
            raise ValueError('Invalid trace row')
        if len(row['prompt_token_ids']) + row['max_tokens'] > args.max_model_len:
            raise ValueError('Trace prompt + output exceeds max_model_len')
    llm = LLM(args.model, enforce_eager=args.eager, tensor_parallel_size=1,
              enable_mixed_batching=args.mode == 'mixed', collect_request_metrics=True,
              max_num_batched_tokens=args.budget, max_num_seqs=args.max_num_seqs,
              max_model_len=args.max_model_len, num_kvcache_blocks=args.kv_blocks,
              gpu_memory_utilization=args.gpu_memory_utilization)
    try:
        if any(max(r['prompt_token_ids']) >= llm.model_runner.config.hf_config.vocab_size for r in rows):
            raise ValueError('Trace contains tokens outside model vocabulary')
        # Same warmup requests across modes; compile/capture is outside timing.
        llm.generate([r['prompt_token_ids'] for r in rows],
                     [SamplingParams(max_tokens=min(8, r['max_tokens']), ignore_eos=True) for r in rows],
                     use_tqdm=False)
        bm = llm.scheduler.block_manager
        if args.cache_state == 'cold':
            llm.scheduler.block_manager = BlockManager(len(bm.blocks), bm.block_size)
        if modern:
            llm.request_metrics.clear()
        torch.cuda.synchronize()
        records, steps = {}, []
        origin = time.perf_counter()
        if original:
            observe_original(llm, records, steps, origin)
        next_row = 0
        while next_row < len(rows) or not llm.is_finished():
            while next_row < len(rows) and rows[next_row]['arrival_s'] <= time.perf_counter()-origin:
                row = rows[next_row]
                params = SamplingParams(temperature=0.6, max_tokens=row['max_tokens'], ignore_eos=True)
                seq_id = llm.add_request(row['prompt_token_ids'], params)
                if original:
                    seq_id = llm.scheduler.waiting[-1].seq_id
                record = dict(trace_id=row['id'], seq_id=seq_id, planned_arrival_s=row['arrival_s'],
                              actual_enqueue_s=time.perf_counter()-origin, input_tokens=len(row['prompt_token_ids']),
                              output_token_ids=[], token_timestamps_s=[], finished_s=None)
                if modern:
                    record['actual_enqueue_s'] = llm.request_metrics[seq_id].enqueued_at-origin
                records[seq_id] = record
                next_row += 1
            if llm.is_finished():
                time.sleep(min(0.001, max(0, rows[next_row]['arrival_s']-(time.perf_counter()-origin))))
                continue
            finished, stats = llm.step()
            if modern:
                now = stats.tokens_available_at-origin
                for sid, token in stats.sampled_tokens.items():
                    records[sid]['output_token_ids'].append(token)
                    records[sid]['token_timestamps_s'].append(now)
                for sid, _ in finished:
                    records[sid]['finished_s'] = now
                steps.append(dict(available_s=now, prefill_tokens=stats.num_prefill_tokens,
                                  decode_tokens=stats.num_decode_tokens, output_tokens=stats.num_output_tokens,
                                  recomputed_tokens=stats.num_recomputed_tokens, preemptions=stats.num_preemptions,
                                  used_kv_blocks=stats.num_used_kv_blocks))
        elapsed = time.perf_counter()-origin
        records = list(records.values())
        cfg = llm.model_runner.config
        result = dict(config=vars(args), git=git_info(repo), trace=trace, requests=records, steps=steps,
                      environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
                                       gpu=torch.cuda.get_device_name(), dtype=str(cfg.hf_config.dtype),
                                       kv_blocks=cfg.num_kvcache_blocks,
                                       flash_attn=importlib.import_module('flash_attn').__version__),
                      summary=summarize(records, steps, elapsed))
        result['trace_sha256'] = hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()
        result['benchmark_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        model_dir = Path(args.model)
        result['model_manifest'] = {'config_sha256': hashlib.sha256((model_dir / 'config.json').read_bytes()).hexdigest(),
                                    'weights': {p.name: p.stat().st_size for p in sorted(model_dir.glob('*.safetensors'))}}
        if trace['scenario'] == 'injection':
            long_row = max(records, key=lambda r: r['input_tokens'])
            active = [r for r in records if r is not long_row and any(t < long_row['actual_enqueue_s'] for t in r['token_timestamps_s'])
                      and any(t > long_row['actual_enqueue_s'] for t in r['token_timestamps_s'])]
            result['summary']['decoding_requests_at_injection'] = len(active)
            result['summary']['injection_case_valid'] = len(active) >= 2
        return result
    finally:
        atexit.unregister(llm.exit)
        llm.exit()


def save_result(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    with path.with_suffix('.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(result['summary']))
        writer.writeheader()
        writer.writerow(result['summary'])
    with path.with_name(path.stem + '_tokens.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['trace_id', 'seq_id', 'output_index', 'token_id', 'available_s', 'planned_arrival_s', 'actual_enqueue_s'])
        for r in result['requests']:
            for i, (token, timestamp) in enumerate(zip(r['output_token_ids'], r['token_timestamps_s'])):
                writer.writerow([r['trace_id'], r['seq_id'], i, token, timestamp, r['planned_arrival_s'], r['actual_enqueue_s']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--make-trace', choices=['offline', 'injection', 'mixed'])
    parser.add_argument('--trace', required=True)
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--injection-at', type=float, default=0.2)
    parser.add_argument('--repo', default=str(Path(__file__).resolve().parent))
    parser.add_argument('--model')
    parser.add_argument('--mode', choices=['original', 'legacy', 'mixed'], default='mixed')
    parser.add_argument('--budget', type=int, default=512)
    parser.add_argument('--max-num-seqs', type=int, default=32)
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--kv-blocks', type=int, default=256)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.8)
    parser.add_argument('--cache-state', choices=['cold', 'warm'], default='cold')
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--output', default='results/mixed.json')
    args = parser.parse_args()
    if args.make_trace:
        if args.count <= 0 or (args.make_trace == 'injection' and args.count < 3):
            parser.error('Use positive count; injection needs at least 3 requests')
        path = Path(args.trace)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(make_trace(args.make_trace, args.seed, args.count, args.injection_at)), encoding='utf-8')
        return
    if not args.model:
        parser.error('--model is required for replay')
    result = replay(args, json.loads(Path(args.trace).read_text(encoding='utf-8')))
    save_result(result, args.output)
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
