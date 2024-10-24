import torch
from transformers.models.llama.modeling_llama import LlamaSdpaAttention
from transformers import LlamaConfig
from deepspeed_profiler import FlopsProfiler
# from deepspeed.profiling.flops_profiler import FlopsProfiler
# from analyzer.utils import Timer

import torch
from diffusers import FluxPipeline
from torch.profiler import profile, record_function, ProfilerActivity

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda:1")

def single_run(pipe: FluxPipeline = pipe, batch_size=1, height=1024, width=1024, num_inference_steps=50):
    prompt = "A cat holding a sign that says hello world"
    image = pipe(
        prompt=[prompt] * batch_size,
        height=height,
        width=width,
        guidance_scale=3.5,
        num_inference_steps=num_inference_steps,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    image.save("flux-dev.png")

if __name__ == "__main__":
    single_run(batch_size=1, height=1024, width=1024, num_inference_steps=2)
