#!/bin/bash
# Author: Yangyan(also, Yang Xianhui)

# This script is used to perform deduplication processing on the first 50
# million lines of the sorted validPairs file in blocks of each million lines, 
# using mkfifo pipes for parallel processing, with each pipe using independent memory.

# Parse command-line parameters
while getopts "i:o:t:p:s:f:" opt; do
    case $opt in
        i) sorted_file="$OPTARG" ;;  # The absolute path of the sorted validPairs file
        o) output_dir="$OPTARG" ;;   
        t) tmp_dir="$OPTARG" ;;      
        p) sort_threads="$OPTARG" ;; # The number of threads for each sort task
        s) sort_memory="$OPTARG" ;;  # The memory parameters of each sort task
        f) max_jobs="$OPTARG" ;;     # Maximum number of parallel tasks
        *) echo "Usage: $0 -i <sorted_file> -o <output_dir> -t <tmp_dir> -p <sort_threads> -s <sort_memory> -f <max_jobs>"; exit 1 ;;
    esac
done

# Check if the parameters are provided
if [ -z "$sorted_file" ] || [ -z "$output_dir" ] || [ -z "$tmp_dir" ] || [ -z "$sort_threads" ] || [ -z "$sort_memory" ] || [ -z "$max_jobs" ]; then
    echo "Error: The parameters -i, -o, -t, -p, -s and -f must be provided."
    echo "Usage: $0 -i <sorted_file> -o <output_dir> -t <tmp_dir> -p <sort_threads> -s <sort_memory> -f <max_jobs>"
    exit 1
fi

# Check whether the input file exists
if [ ! -f "$sorted_file" ]; then
    echo "Error: The sorted file  $sorted_file does not exist."
    exit 1
fi

# Create the required path
mkdir -p "$output_dir"
mkdir -p "$tmp_dir"

# Extract the file name and sample name
file_name=$(basename "$sorted_file")
sample_name="${file_name%_sorted.*}"

# Generate a task list
total_tasks=50 # This parameter supports customization
extract_lines=()
for ((i=1; i<=total_tasks; i++)); do
    extract_lines+=("$((i * 1000000))")
done

# Define the function for running the task
run_task() {
    local line="$1"
    local sorted_file="$2"
    local output_dir="$3"
    local tmp_dir="$4"
    local sort_threads="$5"
    local sort_memory="$6"
    local sample_name="$7"

    # Task-specific temporary directory
    sample_tmp_dir="$tmp_dir/$sample_name/$line"
    mkdir -p "$sample_tmp_dir"
    
    # Create a FIFO pipeline
    fifo_file="$sample_tmp_dir/data.fifo"
    mkfifo "$fifo_file"
    
    # Output file naming
    log_file="$output_dir/${sample_name}_${line}_dedup_log.txt"
    data_file="$output_dir/${sample_name}_${line}_data.txt"
    runtime_log="$output_dir/${sample_name}_runtime_log.txt"

    # Extract the first N lines to the FIFO 
    head -n "$line" "$sorted_file" > "$fifo_file" &
    
    # Read data from the FIFO and process it
    {
        # Record the start time
        start_time=$(date +%s)
        
        # Process data
        sort --parallel="$sort_threads" -T "$sample_tmp_dir" -S "$sort_memory" -k2,2V -k3,3n -k5,5V -k6,6n "$fifo_file" | \
            awk -F"\t" 'BEGIN{c1=0;c2=0;s1=0;s2=0}(c1!=$2 || c2!=$5 || s1!=$3 || s2!=$6){print;c1=$2;c2=$5;s1=$3;s2=$6}' | \
            wc -l > "$sample_tmp_dir/dedup_count.txt"
        
        # Calculate the number of lines after deduplication
        dedup_lines=$(cat "$sample_tmp_dir/dedup_count.txt")
        echo -e "${line}\t${dedup_lines}" > "$data_file"
        
        # Standardized processing (the first column divided by 1,000,000)
        awk -v line="$line" 'BEGIN{FS=OFS="\t"}{$1 = $1 / 1000000; print}' "$data_file" > "${data_file}.tmp"
        mv "${data_file}.tmp" "$data_file"
        
        # Record the running time (format: Task <line> completed in <seconds> seconds)
        end_time=$(date +%s)
        runtime=$((end_time - start_time))
        echo "Task $line completed in $runtime seconds" >> "$runtime_log"
        
        # Clear temporary documents
        rm -rf "$sample_tmp_dir"
    } &
}

# Dynamic task scheduling logic
current_jobs=0
for line in "${extract_lines[@]}"; do
    # If the current number of tasks reaches the upper limit, 
    # wait for any one of them to complete
    while [ $current_jobs -ge $max_jobs ]; do
        # Wait for any background task to complete
        wait -n
        current_jobs=$((current_jobs - 1))
    done
    
    # Start a new task
    echo "Starting task for $line lines..."
    run_task "$line" "$sorted_file" "$output_dir" "$tmp_dir" "$sort_threads" "$sort_memory" "$sample_name"
    current_jobs=$((current_jobs + 1))
done

# Wait for all remaining tasks to be completed
wait

echo "All tasks completed!"

# Merge all the data files and sort them by the first column
combined_data="$output_dir/${sample_name}_combined_data.txt"

# Merge all data files and sort them
cat "$output_dir/${sample_name}"_*_data.txt | sort -k1,1n >> "$combined_data"

# Remove the header and generate the final file
final_output="$output_dir/${sample_name}_final_output.txt"
tail -n +1 "$combined_data" > "$final_output"

# Delete the intermediate files
rm -f "$combined_data"
rm -f "$output_dir/${sample_name}"_*_data.txt

# Extract the running time and calculate the maximum time
runtime_log="$output_dir/${sample_name}_runtime_log.txt"
max_runtime=$(awk '{print $(NF-1)}' "$runtime_log" | sort -n | tail -1)
echo "Longest runtime: $max_runtime seconds" >> "$runtime_log"

# Clear the temporary directory
find "$tmp_dir" -type d -empty -delete

echo "Results merged, sorted, and header removed. Final output saved to $final_output."
echo "Longest runtime: $max_runtime seconds"