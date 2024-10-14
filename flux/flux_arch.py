import torch
from diffusers import FluxPipeline
# from torch.profiler import profile, record_function, ProfilerActivity

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda:1")
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

print(pipe)
# the arch of the transformer:
print(pipe.transformer)
