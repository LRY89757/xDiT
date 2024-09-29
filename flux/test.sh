
CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master_port=1038 flux/flux_profiler.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
--height 4096 --width 4096 --warmup_steps 0 --ulysses_degree 1 --ring_degree 1 \
--pipefusion_parallel_degree 1 --num_pipeline_patch 1 --batch_size 4 2>&1 | tee logs/flux/log_ngpu1_size4096_ulysses1_ring1_bs4.log