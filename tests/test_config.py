import pytest
from nanovllm.config import Config


@pytest.mark.parametrize('kwargs', [
    {'max_num_batched_tokens': 0}, {'max_num_seqs': 0}, {'max_model_len': 0},
    {'gpu_memory_utilization': 1.1}, {'num_kvcache_blocks': 0},
    {'cuda_graph_memory_reserve': -1}, {'kvcache_block_size': 0},
    {'kvcache_block_size': 255}, {'tensor_parallel_size': 0},
])
def test_invalid_config_before_model_loading(tmp_path, kwargs):
    with pytest.raises(ValueError):
        Config(str(tmp_path), **kwargs)
