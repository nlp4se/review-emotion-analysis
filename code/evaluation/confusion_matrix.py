import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate confusion matrix and metrics for emotion annotations')
    parser.add_argument('--ground_truth', required=True, help='Path to ground truth Excel file')
    parser.add_argument('--predictions', required=True, help='Path to predictions Excel file')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    return parser.parse_args()

def load_and_process_data(file_path):
    # Explicitly specify the engine based on file extension
    if file_path.endswith('.xlsx'):
        engine = 'openpyxl'
    elif file_path.endswith('.xls'):
        engine = 'xlrd'
    else:
        raise ValueError("File must be either .xlsx or .xls format")
    
    df = pd.read_excel(file_path, engine=engine)
    # Select columns J to S (emotion annotations)
    return df.iloc[:, 9:19]  # 0-based indexing for columns J to S

def create_confusion_matrices(ground_truth_data, prediction_data, emotion_labels, output_dir):
    # Convert the binary columns into a single column with emotion labels
    y_true = []
    y_pred = []
    
    for idx in range(len(ground_truth_data)):
        # For each row, get the emotions that are marked as 1
        true_emotions = [emotion for emotion in emotion_labels if ground_truth_data[emotion].iloc[idx] == 1]
        pred_emotions = [emotion for emotion in emotion_labels if prediction_data[emotion].iloc[idx] == 1]
        
        # Add all combinations of true and predicted emotions
        for true_emotion in true_emotions:
            for pred_emotion in pred_emotions:
                y_true.append(true_emotion)
                y_pred.append(pred_emotion)
            
            # If no emotion was predicted, count it as "None"
            if not pred_emotions:
                y_true.append(true_emotion)
                y_pred.append("None")
        
        # If true emotions is empty but there are predictions, count them as false positives
        if not true_emotions and pred_emotions:
            for pred_emotion in pred_emotions:
                y_true.append("None")
                y_pred.append(pred_emotion)

    # Create confusion matrix
    labels = list(emotion_labels) + ["None"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Create and save heatmap
    plt.figure(figsize=(15, 12))
    
    # Normalize confusion matrix by row (ground truth), handling zero divisions
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    # Replace zero sums with 1 to avoid division by zero
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_normalized = cm.astype('float') / row_sums
    
    # Create the heatmap with percentages
    ax = sns.heatmap(cm_normalized, 
                     annot=cm,  # Show original numbers
                     fmt='d',   # Format for original numbers
                     cmap='Blues',
                     xticklabels=labels,
                     yticklabels=labels,
                     annot_kws={'size': 20})
    
    # Add borders to diagonal cells
    for i in range(len(labels)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2))
    
    plt.title('Confusion Matrix Across Emotions\n(Colors show row percentages)', fontsize=16)
    plt.xlabel('Generative AI Emotions', fontsize=14)
    plt.ylabel('Ground Truth Emotions', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(rotation=45, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_heatmap.png'))
    plt.close()
    
    # Save numerical data to text file
    with open(os.path.join(output_dir, 'confusion_matrix.txt'), 'w') as f:
        f.write('Confusion Matrix:\n')
        f.write('Rows: Ground Truth, Columns: Predictions\n\n')
        f.write('\t' + '\t'.join(labels) + '\n')
        for i, true_label in enumerate(labels):
            row = [true_label] + [str(x) for x in cm[i]]
            f.write('\t'.join(row) + '\n')

def calculate_metrics(ground_truth_data, prediction_data, emotion_labels):
    results = []
    
    for emotion in emotion_labels:
        y_true = ground_truth_data[emotion].values
        y_pred = prediction_data[emotion].values
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        
        results.append({
            'Emotion': emotion,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        })
    
    # Calculate averages
    metrics_df = pd.DataFrame(results)
    weighted_avg = metrics_df[['Accuracy', 'Precision', 'Recall', 'F1']].mean()
    
    # Calculate non-weighted averages
    y_true_all = ground_truth_data.values.flatten()
    y_pred_all = prediction_data.values.flatten()
    
    overall_accuracy = accuracy_score(y_true_all, y_pred_all)
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, average='macro'
    )
    
    # Add averages to the results
    metrics_df.loc['Weighted Average'] = {
        'Emotion': 'Weighted Average',
        'Accuracy': weighted_avg['Accuracy'],
        'Precision': weighted_avg['Precision'],
        'Recall': weighted_avg['Recall'],
        'F1': weighted_avg['F1']
    }
    
    metrics_df.loc['Non-weighted Average'] = {
        'Emotion': 'Non-weighted Average',
        'Accuracy': overall_accuracy,
        'Precision': overall_precision,
        'Recall': overall_recall,
        'F1': overall_f1
    }
    
    return metrics_df

def main():
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    ground_truth_data = load_and_process_data(args.ground_truth)
    prediction_data = load_and_process_data(args.predictions)
    
    emotion_labels = ground_truth_data.columns
    
    # Generate confusion matrices and save numerical data
    create_confusion_matrices(ground_truth_data, prediction_data, emotion_labels, args.output_dir)
    
    # Calculate and save metrics
    metrics_df = calculate_metrics(ground_truth_data, prediction_data, emotion_labels)
    
    # Save metrics to CSV and text file
    metrics_df.to_csv(os.path.join(args.output_dir, 'metrics.csv'), index=False)
    
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(metrics_df.to_string())

if __name__ == "__main__":
    main()
