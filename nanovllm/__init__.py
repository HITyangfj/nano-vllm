from nanovllm.sampling_params import SamplingParams

__all__ = ["LLM", "SamplingParams"]


def __getattr__(name):
    # Scheduler/metadata tests do not require the CUDA runtime or model weights.
    if name == "LLM":
        from nanovllm.llm import LLM
        return LLM
    raise AttributeError(name)
