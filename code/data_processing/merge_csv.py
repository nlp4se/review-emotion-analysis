import sys
import pandas as pd
from pathlib import Path

def merge_csv_files(output_file, *input_files):
    try:
        # Read and merge all CSV files
        merged_data = pd.concat(
            [pd.read_csv(file) for file in input_files], 
            ignore_index=True
        )
        
        # Keep only specified columns and rename agreement columns
        columns_to_keep = [
            'app_name', 'categoryId', 'reviewId', 'sentenceId', 'at', 
            'score', 'feature', 'review', 'sentence', 
            'emotion-A-agreement', 'emotion-B-agreement'
        ]
        
        merged_data = merged_data[columns_to_keep]
        merged_data = merged_data.rename(columns={
            'emotion-A-agreement': 'emotion-A',
            'emotion-B-agreement': 'emotion-B'
        })
        
        # Filter out rows where 'emotion-A' column is 'Reject'
        if 'emotion-A' in merged_data.columns:
            merged_data = merged_data[merged_data['emotion-A'] != 'Reject']

        # Print statistics about emotion frequencies (combined A and B)
        all_emotions = pd.concat([merged_data['emotion-A'], merged_data['emotion-B']]).dropna()
        emotion_counts = all_emotions.value_counts()
        total_emotions = len(all_emotions)
        
        print("\nEmotion Frequencies (combined from A and B):")
        print("-" * 50)
        for emotion, count in emotion_counts.items():
            percentage = (count / total_emotions) * 100
            print(f"{emotion:<20} {count:>5} ({percentage:>6.2f}%)")
        print("-" * 50)
        print(f"Total emotions: {total_emotions}")
        
        # Save the merged data to the output file
        merged_data.to_csv(output_file, index=False)
        print(f"Successfully merged {len(input_files)} files into {output_file}")
        print(f"Total: {len(merged_data)} reviews")
    except Exception as e:
        print(f"Error merging files: {e}")

if __name__ == "__main__":
    # Ensure proper usage
    if len(sys.argv) != 3:
        print("Usage: python merge_csv.py output_file.csv input_folder")
        sys.exit(1)

    # Extract output file and input folder from arguments
    output_csv = sys.argv[1]
    input_folder = sys.argv[2]

    # Get all CSV files from input folder
    input_csv_files = [str(f) for f in Path(input_folder).glob('*.csv')]

    if not input_csv_files:
        print(f"No CSV files found in {input_folder}")
        sys.exit(1)

    merge_csv_files(output_csv, *input_csv_files)
