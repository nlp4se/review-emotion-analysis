import pandas as pd
import os
import argparse
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description='Split annotations file into iteration-specific files')
    parser.add_argument('--input_file', type=str, required=True, 
                       help='Path to the input Excel file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to the output directory')
    return parser.parse_args()

def get_base_filename(input_file):
    # Remove '-annotations.xlsx' from the filename
    base_name = Path(input_file).stem  # Get filename without extension
    return base_name.replace('-annotations', '')

def create_iteration_directory(output_dir, iteration):
    iteration_dir = os.path.join(output_dir, f'{iteration}')
    os.makedirs(iteration_dir, exist_ok=True)
    return iteration_dir

def split_annotations(input_file, output_dir):
    # Read the Excel file - explicitly keep all rows including duplicates
    df = pd.read_excel(input_file)
    
    # Get base filename for output files
    base_filename = get_base_filename(input_file)
    
    # Group by iteration and save to separate files
    for iteration, group in df.groupby('iteration'):
        
        # Create iteration directory
        iteration_dir = create_iteration_directory(output_dir, iteration)
        
        # Create output filename
        output_filename = f'{iteration}_{base_filename}.xlsx'
        output_path = os.path.join(iteration_dir, output_filename)
        
        # Drop duplicates before saving
        #group = group.drop_duplicates()
        
        # Save the group to a new Excel file
        group.to_excel(output_path, index=False)
        print(f"Created {output_path}")

def main():
    args = parse_arguments()
    split_annotations(args.input_file, args.output_dir)

if __name__ == "__main__":
    main()