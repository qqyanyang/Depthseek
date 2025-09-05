# Overview of Depthseek
## The pipeline consists of three main steps:

  1. Preprocessing and sorting with HiCpro
  
  2. Downsampling and deduplication with FIFO-based processing

  3. Prediction with Depthseek

# Environment Setup
To set up the required environment, use the provided environment.yml file:

    conda env create -f environment.yml  
    conda activate Depthseek

# Pipeline Steps

## Step 0.1: Preprocess and Sort by Column 1
First, process raw Hi-C data using HiCpro:

    /path/your/HiCpro -c config-hicpro.txt -i raw_data/path -o output

Then sort the valid pairs file by the first column:

    sort -S 50% --parallel 16 -k1,1 /path/validpairs > /output_path/sorted.validpairs

## Step 0.2: Downsampling and Deduplication
### Choose either the bash or Python implementation:
Bash version:

    bash 0_2_Fifo_Downsample_Dedup.sh -i /output_path/sorted.validpairs -o /data_for_predict/data -t ./tmp -p 2 -s 4G -f 20
  
Python version:

    python 0_2_Fifo_Downsample_Dedup_python.py -i /output_path/sorted.validpairs -o /data_for_predict/data -t ./tmp -p 2 -s 4G -f 20
  
Parameters:

    -i: Input sorted validpairs file
  
    -o: Output directory for processed data
  
    -t: Temporary directory
  
    -p: Number of parallel processes
  
    -s: Memory size for sorting (e.g., 4G)
  
    -f: Filtering threshold
  
## Step 0.3: Prediction
  python 0_3_x_Predict*.py -i /data_for_predict/data -o /output/svg_log -n depth_value

Parameters:

    -i: Input directory with processed data
  
    -o: Output directory for results
  
    -n: Depth value for prediction


### Calculating Gbase
  After obtaining validPair counts with Depthseek, calculate Gbase using your in situ Hi-C data's mapping value. Refer to the Formula in the Method section of the associated paper for details.

## Output

The pipeline generates:

* Processed and sorted valid pairs files
  
* Downsampled and deduplicated data
  
* Prediction results in the specified output directory
  
* SVG log files visualization

# Make sure to:

* Adjust all file paths according to your system configuration
  
* Provide appropriate parameters for each step based on your data size and system resources
  
* Refer to the original paper for detailed methodology and formula references

* Demo data for test  https://zenodo.org/records/17054152
