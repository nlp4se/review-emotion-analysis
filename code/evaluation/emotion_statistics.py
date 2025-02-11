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
    emotion_counts = (emotion_cols == 1).sum()
    plt.figure(figsize=(10, 10))
    plt.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%')
    plt.title('Distribution of Emotions')
    plt.savefig(output_dir / 'emotion_distribution_pie.png')
    plt.close()
    
    # 2. Create histogram of emotions per review
    emotions_per_review = (emotion_cols == 1).sum(axis=1)
    plt.figure(figsize=(10, 6))
    plt.hist(emotions_per_review, bins=range(0, max(emotions_per_review) + 2), align='left')
    plt.title('Number of Emotions per Review')
    plt.xlabel('Number of Emotions')
    plt.ylabel('Number of Reviews')
    plt.xticks(range(0, max(emotions_per_review) + 1))
    plt.savefig(output_dir / 'emotions_per_review_hist.png')
    plt.close()
    
    # 3. Create correlation heatmaps
    emotion_data = (emotion_cols == 1)
    
    # Calculate both Phi and Yule's Q correlations
    phi_correlation = emotion_data.corr()
    
    # Calculate Yule's Q correlation
    yules_q = pd.DataFrame(0.0, index=emotion_cols.columns, columns=emotion_cols.columns)
    for i in emotion_cols.columns:
        for j in emotion_cols.columns:
            # Create contingency table
            n11 = ((emotion_data[i] & emotion_data[j])).sum()  # both present
            n10 = ((emotion_data[i] & ~emotion_data[j])).sum() # i present, j absent
            n01 = ((~emotion_data[i] & emotion_data[j])).sum() # i absent, j present
            n00 = ((~emotion_data[i] & ~emotion_data[j])).sum() # both absent
            
            # Calculate Yule's Q: (n11*n00 - n10*n01) / (n11*n00 + n10*n01)
            numerator = (n11 * n00) - (n10 * n01)
            denominator = (n11 * n00) + (n10 * n01)
            yules_q.loc[i, j] = numerator / denominator if denominator != 0 else 0

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot Phi correlation
    sns.heatmap(phi_correlation, 
                ax=ax1,
                annot=True, 
                cmap='RdYlGn',
                vmin=-1, 
                vmax=1, 
                center=0,
                annot_kws={'size': 14},
                fmt='.2f',
                square=True,
                cbar_kws={"shrink": .8})
    ax1.set_title('Phi Correlation Matrix', fontsize=16, pad=20)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=10)
    
    # Plot Yule's Q correlation
    sns.heatmap(yules_q, 
                ax=ax2,
                annot=True, 
                cmap='RdYlGn',
                vmin=-1, 
                vmax=1, 
                center=0,
                annot_kws={'size': 14},
                fmt='.2f',
                square=True,
                cbar_kws={"shrink": .8})
    ax2.set_title("Yule's Q Correlation Matrix", fontsize=16, pad=20)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'emotion_correlation_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical statistics to text file
    with open(output_dir / 'statistics_summary.txt', 'w') as f:
        f.write("Emotion Distribution:\n")
        f.write(emotion_counts.sort_values(ascending=False).to_string())
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

        f.write("\n\nPhi Correlation Matrix:\n")
        f.write(phi_correlation.to_string())
        
        f.write("\n\nYule's Q Correlation Matrix:\n")
        f.write(yules_q.to_string())

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
