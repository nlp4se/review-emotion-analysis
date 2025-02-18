import os
import sys
import pandas as pd
from pathlib import Path
import csv

def process_agreement(input_folder):
    print(f"Starting agreement processing for folder: {input_folder}")
    # Convert input folder to Path object for easier handling
    input_path = Path(input_folder)
    
    # Create a list to store all processed dataframes
    all_processed_dfs = []
    
    # Iterate through all subfolders in the input folder
    for subfolder in input_path.iterdir():
        if not subfolder.is_dir():
            continue
            
        print(f"\nProcessing subfolder: {subfolder.name}")
        
        # Skip exactly iterations 0, 1, and 2
        folder_name = subfolder.name.lower()
        if folder_name.endswith('_0') or folder_name.endswith('_1') or folder_name.endswith('_2'):
            print(f"Skipping iteration folder: {folder_name}")
            continue
            
        # Get all relevant annotation files in the subfolder
        annotation_files = []
        iteration_prefix = None
        
        # Define allowed annotator acronyms
        #annotators = ['QM', 'MT', 'MO', 'JM', 'XF']
        #annotators = ['gpt-4o','mistral-large-2411','gemini-2-0-flash']
        #annotators = ['gpt-4o-1','gpt-4o-2','gpt-4o-3']
        #annotators = ['mistral-large-2411-1','mistral-large-2411-2','mistral-large-2411-3']
        #annotators = ['gemini-2-0-flash-1','gemini-2-0-flash-2','gemini-2-0-flash-3']
        annotators = ['gpt-4o-1-gpt-4o-2-gpt-4o-3',
                      'mistral-large-2411-1-mistral-large-2411-2-mistral-large-2411-3',
                      'gemini-2-0-flash-1-gemini-2-0-flash-2-gemini-2-0-flash-3']
        #annotators = ['agreement_gemini-2-0-flash-1-gemini-2-0-flash-2-gemini-2-0-flash-3']

        # Create compound name from annotators
        annotator_name = '-'.join(sorted(annotators))  # Sort for consistency
        
        for file in subfolder.iterdir():
            # Check if file is xlsx and filename matches pattern
            if (file.suffix != '.xlsx'):
                continue
                
            parts = file.stem.split('_')
            print(parts)
            if (len(parts) == 3 and 
                parts[0] == 'iteration' and 
                parts[2] in annotators) or (len(parts) == 4 and 
                                            parts[0] == 'iteration' and 
                                            parts[2] == 'agreement' and 
                                            parts[3] in annotators):
                annotation_files.append(file)
                iteration_prefix = f"{parts[0]}_{parts[1]}"
        
        if not annotation_files or len(annotation_files) < 2:
            print(f"Not enough annotation files found in {subfolder.name}, skipping...")
            continue
            
        # Read all annotation files
        dataframes = []
        for file in annotation_files:
            print(f"Reading file: {file}")  # Enhanced debug print
            df = pd.read_excel(file)
            print(f"Shape after reading: {df.shape}")  # Print shape after reading
            dataframes.append(df)
            
        # Get the base columns (non-emotion columns) from the first file
        base_df = dataframes[0].copy()
        print(f"Base DataFrame shape: {base_df.shape}")  # Print shape of base_df
        
        # Define emotion columns as columns K through T
        emotion_columns = base_df.columns[10:20]  # Columns K (index 10) through T (index 19)
        print(f"Emotion columns found: {emotion_columns}")  # Print found emotion columns
        
        # Process agreement for each emotion column
        for col in emotion_columns:
            # Count how many annotators marked '1' for each row
            agreement_count = sum(df[col].reset_index(drop=True) == 1 for df in dataframes)
            # Mark 1 if at least 2 annotators agreed, 0 otherwise
            base_df[col] = [1 if count >= 2 else 0 for count in agreement_count]
            print(f"After processing {col}, shape: {base_df.shape}")  # Print shape after each column
        
        # Save the agreement file in the subfolder (both CSV and XLSX)
        output_file_csv = subfolder / f'{iteration_prefix}_agreement_{annotator_name}.csv'
        output_file_xlsx = subfolder / f'{iteration_prefix}_agreement_{annotator_name}.xlsx'
        base_df.to_csv(output_file_csv, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
        base_df.to_excel(output_file_xlsx, index=False)
        print(f"Saved agreement files:\n  {output_file_csv}\n  {output_file_xlsx}")
        
        # Add the processed dataframe to our list
        all_processed_dfs.append(base_df)
    
    # After processing all subfolders, merge all dataframes and save to root
    if all_processed_dfs:
        print("\nMerging all processed dataframes...")
        merged_df = pd.concat(all_processed_dfs, ignore_index=True)
        
        merged_output_csv = input_path / f'agreement_{annotator_name}.csv'
        merged_output_xlsx = input_path / f'agreement_{annotator_name}.xlsx'
        
        merged_df.to_csv(merged_output_csv, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
        merged_df.to_excel(merged_output_xlsx, index=False)
        print(f"Saved merged agreement files:\n  {merged_output_csv}\n  {merged_output_xlsx}")
    else:
        print("\nNo data to merge - no agreement files were generated.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python agreement.py <input_folder>")
        sys.exit(1)
        
    input_folder = sys.argv[1]
    process_agreement(input_folder)
