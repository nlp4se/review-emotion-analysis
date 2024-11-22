import sys
import pandas as pd

def merge_csv_files(output_file, *input_files):
    try:
        # Read and merge all CSV files
        merged_data = pd.concat(
            [pd.read_csv(file) for file in input_files], 
            ignore_index=True
        )
        
        # Filter out rows where 'emotion-A' column is 'Reject'
        if 'emotion-A' in merged_data.columns:
            merged_data = merged_data[merged_data['emotion-A'] != 'Reject']
        
        # Save the merged data to the output file
        merged_data.to_csv(output_file, index=False)
        print(f"Successfully merged {len(input_files)} files into {output_file}")
        print(f"Total: {len(merged_data)} reviews")
    except Exception as e:
        print(f"Error merging files: {e}")

if __name__ == "__main__":
    # Ensure proper usage
    if len(sys.argv) < 3:
        print("Usage: python merge_csv.py output_file.csv input1.csv input2.csv ...")
        sys.exit(1)

    # Extract output file and input files from arguments
    output_csv = sys.argv[1]
    input_csv_files = sys.argv[2:]

    merge_csv_files(output_csv, *input_csv_files)
