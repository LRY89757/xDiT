# # single gpu
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=1037 flux/flux_profiler.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
# --height 1024 --width 1024 --warmup_steps 0 --ulysses_degree 1 --ring_degree 1 \
# --pipefusion_parallel_degree 1 --num_pipeline_patch 1 2>&1 | tee logs/flux/log_ngpu4_size1024_ulysses1_ring1.log

# # 4 gpus
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1037 flux/flux_profiler.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
# --height 1024 --width 1024 --warmup_steps 0 --ulysses_degree 4 --ring_degree 1 \
# --pipefusion_parallel_degree 1 --num_pipeline_patch 1 2>&1 | tee logs/flux/log_ngpu4_size1024_ulysses4_ring1.log

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1037 flux/flux_profiler.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
# --height 1024 --width 1024 --warmup_steps 0 --ulysses_degree 2 --ring_degree 2 \
# --pipefusion_parallel_degree 1 --num_pipeline_patch 1 2>&1 | tee logs/flux/log_ngpu4_size1024_ulysses2_ring2.log


# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1037 flux/flux_profiler.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
# --height 1024 --width 1024 --warmup_steps 0 --ulysses_degree 1 --ring_degree 4 \
# --pipefusion_parallel_degree 1 --num_pipeline_patch 1 2>&1 | tee logs/flux/log_ngpu4_size1024_ulysses1_ring4.log


# single gpu batch size 4
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=1037 flux/flux_profiler.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
# --height 1024 --width 1024 --warmup_steps 0 --ulysses_degree 1 --ring_degree 1 \
# --pipefusion_parallel_degree 1 --num_pipeline_patch 1 --batch_size 4 2>&1 | tee logs/flux/log_ngpu4_size1024_ulysses1_ring1_bs4.log

# # 4 gpus Batch size 4
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1037 flux/flux_profiler.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
# --height 1024 --width 1024 --warmup_steps 0 --ulysses_degree 4 --ring_degree 1 \
# --pipefusion_parallel_degree 1 --num_pipeline_patch 1 --batch_size 4 2>&1 | tee logs/flux/log_ngpu4_size1024_ulysses4_ring1_bs4.log

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1037 flux/flux_profiler.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
# --height 1024 --width 1024 --warmup_steps 0 --ulysses_degree 2 --ring_degree 2 \
# --pipefusion_parallel_degree 1 --num_pipeline_patch 1 --batch_size 4 2>&1 | tee logs/flux/log_ngpu4_size1024_ulysses2_ring2_bs4.log


# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1037 flux/flux_profiler.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
# --height 1024 --width 1024 --warmup_steps 0 --ulysses_degree 1 --ring_degree 4 \
# --pipefusion_parallel_degree 1 --num_pipeline_patch 1 --batch_size 4 2>&1 | tee logs/flux/log_ngpu4_size1024_ulysses1_ring4_bs4.log

# # single gpu batch size 4 height 4096
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=1037 flux/flux_profiler.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
# --height 4096 --width 4096 --warmup_steps 0 --ulysses_degree 1 --ring_degree 1 \
# --pipefusion_parallel_degree 1 --num_pipeline_patch 1 --batch_size 4 2>&1 | tee logs/flux/log_ngpu4_size4096_ulysses1_ring1_bs4.log

# # 4 gpus Batch size 4 height 4096
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1037 flux/flux_profiler.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
# --height 4096 --width 4096 --warmup_steps 0 --ulysses_degree 4 --ring_degree 1 --num_inference_steps 5 \
# --pipefusion_parallel_degree 1 --num_pipeline_patch 1 --batch_size 4 2>&1 | tee logs/flux/log_ngpu4_size4096_ulysses4_ring1_bs4.log

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1037 flux/flux_profiler.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
# --height 4096 --width 4096 --warmup_steps 0 --ulysses_degree 2 --ring_degree 2 --num_inference_steps 5 \
# --pipefusion_parallel_degree 1 --num_pipeline_patch 1 --batch_size 4 2>&1 | tee logs/flux/log_ngpu4_size4096_ulysses2_ring2_bs4.log


# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1037 flux/flux_profiler.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
# --height 4096 --width 4096 --warmup_steps 0 --ulysses_degree 1 --ring_degree 4 --num_inference_steps 5 \
# --pipefusion_parallel_degree 1 --num_pipeline_patch 1 --batch_size 4 2>&1 | tee logs/flux/log_ngpu4_size4096_ulysses1_ring4_bs4.log

#!/bin/bash

# Define the batch sizes, parallel degrees, ulysses degrees, ring degrees, and sizes
batch_sizes=(1 2 4)
parallel_degrees=(4 8)
ulysses_degrees=(1 2 4 8)
ring_degrees=(8 4 2 1)
sizes=(1024 2048 4096)

# # Loop through each combination of batch size, parallel degree, ulysses degree, ring degree, and size
# for bs in "${batch_sizes[@]}"; do
#   for pd in "${parallel_degrees[@]}"; do
#     for ud in "${ulysses_degrees[@]}"; do
#       for rd in "${ring_degrees[@]}"; do
#         for size in "${sizes[@]}"; do
#           product=$((ud * rd))
#           if [ "$product" -eq "$pd" ]; then
#             cmd="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=$pd --master_port=1037 flux/flux_profiler.py --prompt 'hello world' --output_type 'latent' --model 'black-forest-labs/FLUX.1-dev' \
#             --height $size --width $size --warmup_steps 0 --ulysses_degree $ud --ring_degree $rd --num_inference_steps 10 \
#             --pipefusion_parallel_degree 1 --num_pipeline_patch 1 --batch_size $bs 2>&1 | tee logs/flux/log_size${size}_ulysses${ud}_ring${rd}_bs${bs}_pd${pd}.log"
#             echo $cmd
#             eval $cmd
#           fi
#         done
#       done
#     done
#   done
# done

parallel_degrees=(1)
# Loop through each combination of batch size, parallel degree, ulysses degree, ring degree, and size
for bs in "${batch_sizes[@]}"; do
  for pd in "${parallel_degrees[@]}"; do
    for ud in "${ulysses_degrees[@]}"; do
      for rd in "${ring_degrees[@]}"; do
        for size in "${sizes[@]}"; do
          product=$((ud * rd))
          if [ "$product" -eq "$pd" ]; then
            cmd="CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=$pd --master_port=1037 flux/flux_profiler.py --prompt 'hello world' --output_type 'latent' --model 'black-forest-labs/FLUX.1-dev' \
            --height $size --width $size --warmup_steps 0 --ulysses_degree $ud --ring_degree $rd --num_inference_steps 10 \
            --pipefusion_parallel_degree 1 --num_pipeline_patch 1 --batch_size $bs 2>&1 | tee logs/flux/log_size${size}_ulysses${ud}_ring${rd}_bs${bs}_pd${pd}.log"
            echo $cmd
            eval $cmd
          fi
        done
      done
    done
  done
done

