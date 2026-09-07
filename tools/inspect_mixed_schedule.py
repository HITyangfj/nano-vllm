"""CPU-only acceptance example; deterministic fake samples, no model execution."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.scheduler_output import prepare_batch_inputs
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    args = parser.parse_args()
    scheduler = Scheduler(SimpleNamespace(max_num_seqs=3, max_num_batched_tokens=512,
        max_model_len=4096, enable_mixed_batching=True, eos=-1,
        kvcache_block_size=256, num_kvcache_blocks=32))
    seqs = [Sequence([i+1]*length, SamplingParams(max_tokens=4, ignore_eos=True))
            for i, length in enumerate([10, 20, 1200])]
    names = {seq.seq_id: name for seq, name in zip(seqs, 'ABC')}
    for seq in seqs[:2]:
        scheduler.add(seq)
    first = scheduler.schedule()
    scheduler.postprocess(first, {sid: 7 for sid in first.sample_seq_ids})
    scheduler.add(seqs[2])
    rows = []
    while not scheduler.is_finished():
        plan = scheduler.schedule()
        inputs = prepare_batch_inputs(plan, 256)
        rows.append(dict(requests=[dict(name=names[r.seq_id], start=r.start_pos, end=r.end_pos,
                                       query_len=r.num_scheduled_tokens,
                                       phase='decode' if r.is_decode else 'prefill', sample=r.needs_sample)
                                   for r in plan.requests], query_start_loc=inputs.query_start_loc,
                         logits_indices=inputs.logits_indices, num_slots=len(inputs.slot_mapping),
                         prefill_tokens=plan.num_prefill_tokens, decode_tokens=plan.num_decode_tokens))
        scheduler.postprocess(plan, {sid: 7 for sid in plan.sample_seq_ids})
    result = dict(kind='CPU scheduler example; fake sampled token=7; no GPU performance data', steps=rows)
    encoded = json.dumps(result, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding='utf-8')
    print(encoded)


if __name__ == '__main__':
    main()
