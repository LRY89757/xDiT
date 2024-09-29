import logging
import time
import torch
import torch.distributed
from xfuser import xFuserFluxPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
    is_dp_last_group,
)
from torch.profiler import profile, record_function, ProfilerActivity

def single_run(pipe, input_config, local_rank):
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        max_sequence_length=256,
        guidance_scale=0.0,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    )
    return output

def warmup(pipe, input_config, local_rank, times=3):
    for _ in range(times):
        single_run(pipe, input_config, local_rank)

def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank

    pipe = xFuserFluxPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
    )

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    else:
        pipe = pipe.to(f"cuda:{local_rank}")

    pipe.prepare_run(input_config)

    # Warmup
    warmup(pipe, input_config, local_rank, times=1)

    get_runtime_state().destory_distributed_env()

if __name__ == "__main__":
    main()

'''
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1037 flux/test_wrap.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
--height 1024 --width 1024 --warmup_steps 0 --ulysses_degree 1 --ring_degree 2 \
--pipefusion_parallel_degree 1 --num_pipeline_patch 1 2>&1 | tee logs/flux/test_wrap.log
'''