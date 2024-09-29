import json
from collections import defaultdict


def read_chrome_trace(file_path):
    with open(file_path, 'r') as f:
        trace_data = json.load(f)
    return trace_data['traceEvents']


def analyze_trace(events):
    stream_times = defaultdict(float)
    operator_times = defaultdict(lambda: defaultdict(float))
    operator_counts = defaultdict(lambda: defaultdict(int))

    for event in events:
        if event.get('ph') == 'X':  # Complete events
            duration = event.get('dur', 0) / 1000  # Convert to milliseconds
            name = event.get('name', 'Unknown')
            stream = event.get('args', {}).get('stream', 'Unknown')

            stream_times[stream] += duration
            operator_times[stream][name] += duration
            operator_counts[stream][name] += 1

    return stream_times, operator_times, operator_counts


def print_summary(stream_times, operator_times, operator_counts):
    print("Stream Execution Times:")
    for stream, time in sorted(stream_times.items(),
                               key=lambda x: x[1],
                               reverse=True):
        print(f"  Stream {stream}: {time:.2f} ms")

    print("\nTop 10 Operators by Total Execution Time:")
    all_operators = defaultdict(float)
    for stream_ops in operator_times.values():
        for op, time in stream_ops.items():
            all_operators[op] += time

    for op, time in sorted(all_operators.items(),
                           key=lambda x: x[1],
                           reverse=True)[:10]:
        total_count = sum(operator_counts[s][op] for s in operator_counts)
        avg_time = time / total_count if total_count > 0 else 0
        print(f"  {op}:")
        print(f"    Total Time: {time:.2f} ms")
        print(f"    Count: {total_count}")
        print(f"    Average Time: {avg_time:.2f} ms")

    print("\nDetailed Breakdown by Stream:")
    for stream in stream_times:
        print(f"\nStream {stream}:")
        stream_ops = operator_times[stream]
        for op, time in sorted(stream_ops.items(),
                               key=lambda x: x[1],
                               reverse=True)[:5]:
            count = operator_counts[stream][op]
            avg_time = time / count if count > 0 else 0
            print(f"  {op}:")
            print(f"    Total Time: {time:.2f} ms")
            print(f"    Count: {count}")
            print(f"    Average Time: {avg_time:.2f} ms")


def main(file_path):
    events = read_chrome_trace(file_path)
    stream_times, operator_times, operator_counts = analyze_trace(events)
    print_summary(stream_times, operator_times, operator_counts)


'''
Usage: python read_trace.py <path_to_chrome_trace.json>
python profile/read_trace.py /home/nvme-share/home/lurunyu/projects/xDiT/flux/profile_data/ulysses_2_ring_2/xfuser_flux_trace_steps_20_rank_0.json > log.log

flux/profile_data/ulysses_1_ring_1/xfuser_flux_trace_steps_20_rank_0.json
'''
if __name__ == "__main__":
    # import sys
    # if len(sys.argv) != 2:
    #     print("Usage: python read_trace.py <path_to_chrome_trace.json>")
    #     sys.exit(1)
    # main(sys.argv[1])
    folder = "/home/nvme-share/home/lurunyu/projects/xDiT/flux/profile_data/"
    for u, r in [(1, 1), (2, 2), (4, 1), (1, 4)]:
        print(f"ulysses_{u}_ring_{r}")
        file_path = folder + f"ulysses_{u}_ring_{r}/xfuser_flux_trace_steps_20_rank_0.json"
        main(file_path)

