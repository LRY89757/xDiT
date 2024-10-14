import re
import os
import json

def parse_flux_log(log_file_path):
    with open(log_file_path, 'r') as file:
        content = file.read()

    # Regular expressions to match the desired information
    mac_pattern = r"fwd MACs per GPU:\s+([\d.]+)\s+(\w+)"
    flops_pattern = r"fwd flops per GPU:\s+([\d.]+)\s+(\w+)"
    latency_pattern = r"fwd latency:\s+([\d.]+)\s+(\w+)"
    flops_per_gpu_pattern = r"fwd FLOPS per GPU = fwd flops per GPU / fwd latency:\s+([\d.]+)\s+(\w+)"

    # Extract the information
    mac_match = re.search(mac_pattern, content)
    flops_match = re.search(flops_pattern, content)
    latency_match = re.search(latency_pattern, content)
    flops_per_gpu_match = re.search(flops_per_gpu_pattern, content)

    # Extract the values and units
    results = {}
    if mac_match:
        results['fwd_macs_per_gpu'] = f"{mac_match.group(1)} {mac_match.group(2)}"
    if flops_match:
        results['fwd_flops_per_gpu'] = f"{flops_match.group(1)} {flops_match.group(2)}"
    if latency_match:
        latency_value = float(latency_match.group(1))
        latency_unit = latency_match.group(2)
        if latency_unit.lower() == 'ms':
            latency_value /= 1000  # Convert ms to s
        results['fwd_latency'] = f"{latency_value:.6f} s"
    if flops_per_gpu_match:
        results['fwd_flops_per_gpu_per_second'] = f"{flops_per_gpu_match.group(1)} {flops_per_gpu_match.group(2)}"

    return results

if __name__ == "__main__":
    batch_sizes = [1, 2, 4]
    parallel_degrees = [1, 2, 4, 8]
    ulysses_degrees = [1, 2, 4, 8]
    ring_degrees = [8, 4, 2, 1]
    sizes = [1024, 2048, 4096]

    all_results = {}

    for bs in batch_sizes:
        for pd in parallel_degrees:
            for ud in ulysses_degrees:
                for rd in ring_degrees:
                    for size in sizes:
                        if ud * rd != pd:
                            continue
                        log_file_path = f"logs/flux/log_flops_size{size}_ulysses{ud}_ring{rd}_bs{bs}_pd{pd}.log"
                        key = f"bs{bs}_pd{pd}_ud{ud}_rd{rd}_size{size}"
                        if os.path.exists(log_file_path):
                            results = parse_flux_log(log_file_path)
                            all_results[key] = {
                                "batch_size": bs,
                                "parallel_degree": pd,
                                "ulysses_degree": ud,
                                "ring_degree": rd,
                                "size": size,
                                **results
                            }
                        else:
                            print(f"Skipping {ud} {rd} {pd}")
                            all_results[key] = {
                                "batch_size": bs,
                                "parallel_degree": pd,
                                "ulysses_degree": ud,
                                "ring_degree": rd,
                                "size": size,
                                "fwd_macs_per_gpu": None,
                                "fwd_flops_per_gpu": None,
                                "fwd_latency": None,
                                "fwd_flops_per_gpu_per_second": None,
                                "skipped": True
                            }

    # Save results to a JSON file
    output_file = "flux_flops_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Results have been saved to {output_file}")