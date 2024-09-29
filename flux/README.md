# Flux

suppose we don't use pipefusion now?

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=1037 flux/flux_profiler.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
--height 1024 --width 1024 --warmup_steps 0 --ulysses_degree 4 --ring_degree 2 \
--pipefusion_parallel_degree 1 --num_pipeline_patch 1 2>&1 | tee logs/flux/log_ngpu8_size1024.log
```

for shape [1024, 1024]
shape of latent_image_ids: torch.Size([1, 4096, 3])
shape of latent: torch.Size([1, 4096, 64])
shape of prompt_embeds: torch.Size([1, 512, 4096])
shape of text_ids: torch.Size([1, 512, 3])

```bash
cat transformer/config.json
{
  "_class_name": "FluxTransformer2DModel",
  "_diffusers_version": "0.30.0.dev0",
  "_name_or_path": "../checkpoints/flux-dev/transformer",
  "attention_head_dim": 128,
  "guidance_embeds": true,
  "in_channels": 64,
  "joint_attention_dim": 4096,
  "num_attention_heads": 24,
  "num_layers": 19,
  "num_single_layers": 38,
  "patch_size": 1,
  "pooled_projection_dim": 768
}

cat vae/config.json
{
  "_class_name": "AutoencoderKL",
  "_diffusers_version": "0.30.0.dev0",
  "_name_or_path": "../checkpoints/flux-dev",
  "act_fn": "silu",
  "block_out_channels": [
    128,
    256,
    512,
    512
  ],
  "down_block_types": [
    "DownEncoderBlock2D",
    "DownEncoderBlock2D",
    "DownEncoderBlock2D",
    "DownEncoderBlock2D"
  ],
  "force_upcast": true,
  "in_channels": 3,
  "latent_channels": 16,
  "latents_mean": null,
  "latents_std": null,
  "layers_per_block": 2,
  "mid_block_add_attention": true,
  "norm_num_groups": 32,
  "out_channels": 3,
  "sample_size": 1024,
  "scaling_factor": 0.3611,
  "shift_factor": 0.1159,
  "up_block_types": [
    "UpDecoderBlock2D",
    "UpDecoderBlock2D",
    "UpDecoderBlock2D",
    "UpDecoderBlock2D"
  ],
  "use_post_quant_conv": false,
  "use_quant_conv": false
}
```

