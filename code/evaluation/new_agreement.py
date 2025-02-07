import os
import sys
import pandas as pd
from pathlib import Path
import csv

def process_agreement(input_folder):
    # Convert input folder to Path object for easier handling
    input_path = Path(input_folder)
    
    # Create a list to store all processed dataframes
    all_processed_dfs = []
    
    # Iterate through all subfolders in the input folder
    for subfolder in input_path.iterdir():
        if not subfolder.is_dir():
            continue
            
        # Skip exactly iterations 0, 1, and 2
        folder_name = subfolder.name.lower()
        if folder_name.endswith('_0') or folder_name.endswith('_1') or folder_name.endswith('_2'):
            continue
            
        # Get all relevant annotation files in the subfolder
        annotation_files = []
        iteration_prefix = None
        
        for file in subfolder.iterdir():
            # Check if filename matches pattern 'iteration_X_AA' where AA is a 2-letter acronym
            parts = file.stem.split('_')
            if len(parts) == 3 and parts[0] == 'iteration' and len(parts[2]) == 2 and parts[2].isalpha():
                annotation_files.append(file)
                iteration_prefix = f"{parts[0]}_{parts[1]}"
        
        if not annotation_files or len(annotation_files) < 2:
            continue
            
        # Read all annotation files
        dataframes = []
        for file in annotation_files:
            print(file)  # Debug print to see which files we're processing
            df = pd.read_excel(file)
            dataframes.append(df)
            
        # Get the base columns (non-emotion columns) from the first file
        base_df = dataframes[0].copy()
        # Find emotion columns by checking which columns contain 'X' in any dataframe
        emotion_columns = [col for col in base_df.columns 
                         if any((df[col] == 'X').any() for df in dataframes)]
        
        # Process agreement for each emotion column
        for col in emotion_columns:
            # Count how many annotators marked 'X' for each row
            agreement_count = sum((df[col] == 'X') for df in dataframes)
            # Mark 'X' if at least 2 annotators agreed
            base_df[col] = ['X' if count >= 2 else '' for count in agreement_count]
        
        # Save the agreement file in the subfolder (both CSV and XLSX)
        output_file_csv = subfolder / f'{iteration_prefix}_agreement.csv'
        output_file_xlsx = subfolder / f'{iteration_prefix}_agreement.xlsx'
        base_df.to_csv(output_file_csv, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
        base_df.to_excel(output_file_xlsx, index=False)
        
        # Add the processed dataframe to our list
        all_processed_dfs.append(base_df)
    
    # After processing all subfolders, merge all dataframes and save to root
    if all_processed_dfs:
        merged_df = pd.concat(all_processed_dfs, ignore_index=True)
        merged_output_csv = input_path / 'all_agreements_merged.csv'
        merged_output_xlsx = input_path / 'all_agreements_merged.xlsx'
        merged_df.to_csv(merged_output_csv, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
        merged_df.to_excel(merged_output_xlsx, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python new_agreement.py <input_folder>")
        sys.exit(1)
        
    input_folder = sys.argv[1]
    process_agreement(input_folder)
