#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

def parse_args():
    parser = argparse.ArgumentParser(description='Process sorted validPairs file with deduplication')
    parser.add_argument('-i', required=True, help='Sorted validPairs file absolute path')
    parser.add_argument('-o', required=True, help='Output directory')
    parser.add_argument('-t', required=True, help='Temporary directory')
    parser.add_argument('-p', type=int, required=True, help='Sort threads per task')
    parser.add_argument('-s', required=True, help='Sort memory parameter')
    parser.add_argument('-f', type=int, required=True, help='Max parallel jobs')
    return parser.parse_args()

def validate_args(args):
    if not os.path.isfile(args.i):
        sys.exit(f"Error: Input file {args.i} does not exist")
    os.makedirs(args.o, exist_ok=True)
    os.makedirs(args.t, exist_ok=True)

def run_task(line, args, sample_name):
    start_time = time.time()
    sample_tmp_dir = os.path.join(args.t, sample_name, str(line))
    os.makedirs(sample_tmp_dir, exist_ok=True)
    
    fifo_path = os.path.join(sample_tmp_dir, 'data.fifo')
    try:
        os.mkfifo(fifo_path)
    except FileExistsError:
        pass

    head_cmd = f"head -n {line} {args.i} > {fifo_path}"
    head_proc = subprocess.Popen(head_cmd, shell=True)
    
    sort_cmd = [
        'sort',
        f'--parallel={args.p}',
        f'-T', sample_tmp_dir,
        f'-S', args.s,
        '-k2,2V',
        '-k3,3n',
        '-k5,5V',
        '-k6,6n',
        fifo_path
    ]
    
    awk_cmd = [
        'awk',
        '-F\t',
        'BEGIN{c1=0;c2=0;s1=0;s2=0}(c1!=$2 || c2!=$5 || s1!=$3 || s2!=$6){print;c1=$2;c2=$5;s1=$3;s2=$6}'
    ]
    
    wc_cmd = ['wc', '-l']
    
    sort_proc = subprocess.Popen(sort_cmd, stdout=subprocess.PIPE)
    awk_proc = subprocess.Popen(awk_cmd, stdin=sort_proc.stdout, stdout=subprocess.PIPE)
    wc_proc = subprocess.Popen(wc_cmd, stdin=awk_proc.stdout, stdout=subprocess.PIPE)
    
    dedup_lines = wc_proc.communicate()[0].decode().strip()
    
    data_file = os.path.join(args.o, f"{sample_name}_{line}_data.txt")
    with open(data_file, 'w') as f:
        normalized_line = line // 1000000
        f.write(f"{normalized_line}\t{dedup_lines}\n")
    
    runtime = int(time.time() - start_time)
    runtime_log = os.path.join(args.o, f"{sample_name}_runtime_log.txt")
    with open(runtime_log, 'a') as f:
        f.write(f"Task {line} completed in {runtime} seconds\n")
    
    head_proc.wait()
    os.remove(fifo_path)
    os.rmdir(sample_tmp_dir)

def main():
    args = parse_args()
    validate_args(args)
    
    file_name = os.path.basename(args.i)
    sample_name = file_name.split('_sorted')[0]
    
    tasks = [i * 1000000 for i in range(1, 101)]
    
    with ThreadPoolExecutor(max_workers=args.f) as executor:
        futures = []
        for line in tasks:
            futures.append(executor.submit(run_task, line, args, sample_name))
        
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Task failed: {e}")
    
    combined_data = []
    for line in tasks:
        data_file = os.path.join(args.o, f"{sample_name}_{line}_data.txt")
        if os.path.exists(data_file):
            with open(data_file) as f:
                combined_data.append(f.read().strip())
    
    final_output = os.path.join(args.o, f"{sample_name}_final_output.txt")
    sorted_data = sorted(combined_data, key=lambda x: int(x.split('\t')[0]))
    with open(final_output, 'w') as f:
        f.write("\n".join(sorted_data))
    
    for line in tasks:
        data_file = os.path.join(args.o, f"{sample_name}_{line}_data.txt")
        if os.path.exists(data_file):
            os.remove(data_file)
    
    runtime_log = os.path.join(args.o, f"{sample_name}_runtime_log.txt")
    max_runtime = 0
    if os.path.exists(runtime_log):
        with open(runtime_log) as f:
            for line in f:
                if 'completed' in line:
                    rt = int(line.split()[-2])
                    max_runtime = max(max_runtime, rt)
        with open(runtime_log, 'a') as f:
            f.write(f"Longest runtime: {max_runtime} seconds\n")
    
    print(f"Results saved to {final_output}")
    print(f"Longest runtime: {max_runtime} seconds")

if __name__ == '__main__':
    main()
