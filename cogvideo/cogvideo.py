import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

prompt = "A little girl is riding a bicycle at high speed. Focused, detailed, realistic."
image = load_image(image="input.jpg")
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V",
    torch_dtype=torch.bfloat16
).to("cuda")

# pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

video = pipe(
    prompt=prompt,
    image=image,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(video, "output.mp4", fps=8)

# CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 cogvideo.py
# CUDA_VISIBLE_DEVICES=1 python cogvideo/cogvideo.py