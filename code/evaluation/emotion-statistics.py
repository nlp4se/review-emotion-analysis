import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import argparse
import numpy as np

def analyze_emotion_annotations(xlsx_path):
    # Read the Excel file
    df = pd.read_excel(xlsx_path, engine='openpyxl')
    
    # Get emotion columns (J to S)
    emotion_cols = df.iloc[:, 9:19]  # 0-based indexing for columns J to S
    
    # Create output directory in same location as input file
    output_dir = Path(xlsx_path).parent / 'analysis_output'
    output_dir.mkdir(exist_ok=True)
    
    # 1. Create pie chart of emotion distribution
    emotion_counts = (emotion_cols == 'X').sum()
    plt.figure(figsize=(10, 10))
    plt.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%')
    plt.title('Distribution of Emotions')
    plt.savefig(output_dir / 'emotion_distribution_pie.png')
    plt.close()
    
    # 2. Create histogram of emotions per review
    emotions_per_review = (emotion_cols == 'X').sum(axis=1)
    plt.figure(figsize=(10, 6))
    plt.hist(emotions_per_review, bins=range(0, max(emotions_per_review) + 2), align='left')
    plt.title('Number of Emotions per Review')
    plt.xlabel('Number of Emotions')
    plt.ylabel('Number of Reviews')
    plt.xticks(range(0, max(emotions_per_review) + 1))
    plt.savefig(output_dir / 'emotions_per_review_hist.png')
    plt.close()
    
    # 3. Create correlation heatmap
    correlation_matrix = (emotion_cols == 'X').corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                vmin=-1, 
                vmax=1, 
                center=0)
    plt.title('Emotion Correlation Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / 'emotion_correlation_heatmap.png')
    plt.close()
    
    # Save numerical statistics to text file
    with open(output_dir / 'statistics_summary.txt', 'w') as f:
        f.write("Emotion Distribution:\n")
        f.write(emotion_counts.to_string())
        f.write("\n\nEmotions per Review Statistics:\n")
        f.write(emotions_per_review.describe().to_string())
        
        # Add histogram data
        hist_counts, hist_bins = np.histogram(emotions_per_review, bins=range(0, max(emotions_per_review) + 2))
        f.write("\n\nHistogram Data (Emotions per Review):\n")
        f.write("Number of Emotions | Count of Reviews\n")
        f.write("-" * 35 + "\n")
        for count, bin_start in zip(hist_counts, hist_bins[:-1]):
            f.write(f"{bin_start:^17d} | {count:^15d}\n")
        
        # Add reviews with zero emotions
        zero_emotion_indices = emotions_per_review[emotions_per_review == 0].index
        zero_emotion_reviews = df.loc[zero_emotion_indices, 'reviewId'].tolist()
        f.write("\n\nReviews with Zero Emotions:\n")
        f.write("------------------------\n")
        for review_id in zero_emotion_reviews:
            f.write(f"{review_id}\n")

        f.write("\n\nCorrelation Matrix:\n")
        f.write(correlation_matrix.to_string())

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Analyze emotion annotations from Excel file.')
    parser.add_argument('xlsx_file', 
                       type=str,
                       help='Path to the Excel file containing emotion annotations')
    
    args = parser.parse_args()
    
    # Verify file exists
    if not os.path.exists(args.xlsx_file):
        print(f"Error: File '{args.xlsx_file}' does not exist.")
        exit(1)
        
    analyze_emotion_annotations(args.xlsx_file)
