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


# # single gpu batch size 8 height 4096
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=1037 flux/flux_profiler.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
--height 4096 --width 4096 --warmup_steps 0 --ulysses_degree 1 --ring_degree 1 \
--pipefusion_parallel_degree 1 --num_pipeline_patch 1 --batch_size 8 2>&1 | tee logs/flux/log_ngpu1_size4096_ulysses1_ring1_bs8.log

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

# parallel_degrees=(1)
# parallel_degrees=(2)

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

# enable nsys
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nsys profile --force-overwrite true -t cuda,nvtx -o profile torchrun --nproc_per_node=2 --master_port=1037 flux/flux_profiler.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
# --height 1024 --width 1024 --warmup_steps 0 --ulysses_degree 2 --ring_degree 1 --num_inference_steps 10 \
# --pipefusion_parallel_degree 1 --num_pipeline_patch 1 --batch_size 2 --no_profiler 
# # 2>&1 log.log
# # 2>&1 | tee logs/flux/log_size1024_ulysses2_ring1_bs2_pd2.log


# #!/bin/bash

# # Define the batch sizes, parallel degrees, ulysses degrees, ring degrees, and sizes
# batch_sizes=(1 2 4)
# parallel_degrees=(4 8)
# ulysses_degrees=(1 2 4 8)
# ring_degrees=(8 4 2 1)
# sizes=(1024 2048 4096)

# # Loop through each combination of batch size, parallel degree, ulysses degree, ring degree, and size
# for bs in "${batch_sizes[@]}"; do
#   for pd in "${parallel_degrees[@]}"; do
#     for ud in "${ulysses_degrees[@]}"; do
#       for rd in "${ring_degrees[@]}"; do
#         for size in "${sizes[@]}"; do
#           product=$((ud * rd))
#           if [ "$product" -eq "$pd" ]; then
#             # Generate a unique identifier for this run
#             run_id="size${size}_ulysses${ud}_ring${rd}_bs${bs}_pd${pd}"
            
#             # Define the nsys command
#             nsys_cmd="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nsys profile --force-overwrite true -t cuda,nvtx,osrt,cudnn,cublas -o nsys_profile_${run_id} \
#             torchrun --nproc_per_node=$pd --master_port=1037 flux/flux_profiler.py --prompt 'hello world' --output_type 'latent' --model 'black-forest-labs/FLUX.1-dev' \
#             --height $size --width $size --warmup_steps 0 --ulysses_degree $ud --ring_degree $rd --num_inference_steps 10 \
#             --pipefusion_parallel_degree 1 --num_pipeline_patch 1 --batch_size $bs --no_profiler"

#             # Echo and execute the nsys command
#             echo $nsys_cmd
#             eval $nsys_cmd

#             # Generate the SQLite file from the nsys report
#             sqlite_cmd="nsys-exporter --force-overwrite true -s nsys_profile_${run_id}.nsys-rep -d nsys_profile_${run_id}.sqlite"
#             echo $sqlite_cmd
#             eval $sqlite_cmd
#           fi
#         done
#       done
#     done
#   done
# done


# use profiler to profile flops

#!/bin/bash

# Define the batch sizes, parallel degrees, ulysses degrees, ring degrees, and sizes
batch_sizes=(1 2 4)
parallel_degrees=(1 2 4 8)
ulysses_degrees=(1 2 4 8)
ring_degrees=(8 4 2 1)
sizes=(1024 2048 4096)

# Loop through each combination of batch size, parallel degree, ulysses degree, ring degree, and size
for bs in "${batch_sizes[@]}"; do
  for pd in "${parallel_degrees[@]}"; do
    for ud in "${ulysses_degrees[@]}"; do
      for rd in "${ring_degrees[@]}"; do
        for size in "${sizes[@]}"; do
          product=$((ud * rd))
          if [ "$product" -eq "$pd" ]; then
            # Generate the log file name
            log_file="logs/flux/log_flops_size${size}_ulysses${ud}_ring${rd}_bs${bs}_pd${pd}.log"
            
            # Check if the log file already exists
            if [ -f "$log_file" ]; then
              echo "Log file $log_file already exists. Skipping this configuration."
            else
              cmd="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=$pd --master_port=1037 flux/flux_profiler.py --prompt 'hello world' --output_type 'latent' --model 'black-forest-labs/FLUX.1-dev' \
              --height $size --width $size --warmup_steps 0 --ulysses_degree $ud --ring_degree $rd --num_inference_steps 3 \
              --pipefusion_parallel_degree 1 --num_pipeline_patch 1 --batch_size $bs --use_flops_profile --no_profiler 2>&1 | tee $log_file"
              echo $cmd
              eval $cmd
            fi
          fi
        done
      done
    done
  done
done


# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1037 flux/flux_profiler.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
# --height 4096 --width 4096 --warmup_steps 0 --ulysses_degree 1 --ring_degree 4 --num_inference_steps 3 \
# --pipefusion_parallel_degree 1 --num_pipeline_patch 1 --batch_size 4 --use_flops_profile --no_profiler 2>&1 | tee logs/flux/log_flops_size4096_ulysses1_ring4_bs4_pd4.log


