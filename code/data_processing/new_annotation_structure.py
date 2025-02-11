import os
import pandas as pd
import argparse
from pathlib import Path

def process_excel_file(input_file, output_file):
    # Read only the first sheet
    df = pd.read_excel(input_file, sheet_name=0)
    
    # Keep columns A to I
    base_columns = df.iloc[:, :9]
    
    # Get emotion columns (J to M)
    emotion_columns = df.iloc[:, 9:13]
    
    # Define all possible emotions
    all_emotions = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 
                    'Disgust', 'Anger', 'Anticipation', 'Neutral', 'Reject']
    
    # Create new columns for each emotion, initialized with empty lists
    new_emotion_columns = {emotion: [] for emotion in all_emotions}
    
    # Process each row to mark emotions with 'X'
    for idx, row in emotion_columns.iterrows():
        emotions_in_row = row.dropna().values
        for emotion in all_emotions:
            if emotion in emotions_in_row:
                new_emotion_columns[emotion].append('1')
            else:
                new_emotion_columns[emotion].append('0')
    
    # Create new dataframe with base columns and emotion columns
    result_df = pd.concat([
        base_columns,
        pd.DataFrame(new_emotion_columns)
    ], axis=1)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to new Excel file
    result_df.to_excel(output_file, index=False)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process annotation Excel files')
    parser.add_argument('input_path', help='Input directory path')
    parser.add_argument('output_path', help='Output directory path')
    
    args = parser.parse_args()
    input_base = Path(args.input_path)
    output_base = Path(args.output_path)
    
    # Walk through all subdirectories
    for root, _, files in os.walk(input_base):
        for file in files:
            # Check if file matches the pattern iteration_X_AA.xlsx
            if file.startswith('iteration_') and file.endswith('.xlsx'):
                parts = file.split('_')
                if len(parts) == 3 and parts[2].endswith('.xlsx'):
                    input_file = os.path.join(root, file)
                    
                    # Create corresponding output path
                    relative_path = os.path.relpath(root, input_base)
                    output_dir = output_base / relative_path
                    output_file = output_dir / file
                    
                    print(f"Processing {input_file}")
                    process_excel_file(input_file, output_file)
                    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
