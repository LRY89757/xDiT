import logging
import os
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

def single_run(pipe, input_config, local_rank, batch_size):

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

def warmup(pipe, input_config, local_rank, batch_size, times=3):
    for _ in range(times):
        single_run(pipe, input_config, local_rank, batch_size)

def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')  # Added argument
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
    
    input_config.batch_size = args.batch_size
    prompt = [""] * args.batch_size
    input_config.prompt = prompt

    pipe.prepare_run(input_config)

    # Warmup
    warmup(pipe, input_config, local_rank, args.batch_size, times=5 if args.batch_size == 1 and args.height == 1024 and args.width == 1024 else 1)

    # Profiling
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True,
                 with_stack=True,
                 with_flops=True,
                 with_modules=True,
                 record_shapes=True,
                #  on_trace_ready=torch.profiler.tensorboard_trace_handler("./tensorboard/xfuser_flux")
                 ) as prof:
        with record_function("xfuser_flux_pipeline"):
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            output = single_run(pipe, input_config, local_rank, args.batch_size)
            end_time = time.time()

    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    logging.info(f"Execution time: {elapsed_time:.2f} seconds")
    logging.info(f"Peak memory usage: {peak_memory / 1024**2:.2f} MB")

    folder = f"flux/profile_data/ulysses_{engine_config.parallel_config.ulysses_degree}_"
    folder += f"ring_{engine_config.parallel_config.ring_degree}/"
    os.makedirs(folder, exist_ok=True)

    # Export Chrome trace
    if local_rank == 0:
        prof.export_chrome_trace(
            folder + f"xfuser_flux_trace_steps_{input_config.num_inference_steps}_rank_{local_rank}_bs{args.batch_size}_size{input_config.height}.json"
        )

    # Print key averages
    if local_rank == 0:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    get_runtime_state().destory_distributed_env()

if __name__ == "__main__":
    main()

