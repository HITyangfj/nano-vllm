from bench_mixed import make_trace, summarize, save_result


def test_itl_is_pooled_inter_token_latency_not_request_tpot(tmp_path):
    records = [
        dict(trace_id=0, seq_id=0, planned_arrival_s=0, actual_enqueue_s=1,
             token_timestamps_s=[2, 3, 13], output_token_ids=[7, 8, 9]),
        dict(trace_id=1, seq_id=1, planned_arrival_s=0, actual_enqueue_s=0,
             token_timestamps_s=[1, 2], output_token_ids=[7, 8]),
    ]
    steps = [dict(prefill_tokens=510, decode_tokens=2, recomputed_tokens=0,
                  preemptions=1, used_kv_blocks=4)]
    result = summarize(records, steps, elapsed=20)
    assert result['output_tokens'] == 5
    assert result['output_tokens_per_s'] == 0.25
    assert result['mean_ttft_s'] == 1
    assert result['mean_tpot_s'] == 3.25
    assert result['p99_itl_s'] == 9.82
    assert result['itl_samples'] == 3
    assert result['prefill_tokens'] == 510
    assert result['decode_tokens'] == 2
    path = tmp_path / 'result.json'
    save_result(dict(summary=result, requests=records), path)
    assert len(path.with_name('result_tokens.csv').read_text().splitlines()) == 6


def test_same_trace_has_fixed_wall_clock_arrivals():
    for scenario in ['offline', 'injection', 'mixed']:
        trace = make_trace(scenario, count=8)
        assert trace == make_trace(scenario, count=8)
        assert all(len(r['prompt_token_ids'])+r['max_tokens'] <= 4096 for r in trace['requests'])
    trace = make_trace('injection', count=3, injection_at=0.7)
    assert [r['arrival_s'] for r in trace['requests']] == [0, 0, 0.7]


def test_single_token_requests_have_no_itl():
    records = [dict(actual_enqueue_s=0, token_timestamps_s=[1], output_token_ids=[3])]
    result = summarize(records, [], 2)
    assert result['itl_samples'] == 0 and result['p99_itl_s'] is None
    assert result['mean_tpot_s'] is None and result['mean_ttft_s'] == 1
