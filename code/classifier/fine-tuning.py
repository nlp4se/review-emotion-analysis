import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import os
import logging
import numpy as np

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define label maps
#MULTICLASS_LABEL_MAP = {
#    'Joy': 0, 'Sadness': 1, 'Anger': 2, 'Fear': 3, 'Trust': 4, 'Disgust': 5,
#    'Surprise': 6, 'Anticipation': 7, 'Neutral': 8
#}
MULTICLASS_LABEL_MAP = {
    'Joy': 0, 'Sadness': 1, 'Anticipation': 2, 'Trust': 3, 'Surprise': 4, 'Neutral': 5
}

BINARY_LABEL_MAP = {'Positive': 1, 'Negative': 0}

def remove_empty_folders(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for directory in dirs:
            dir_path = os.path.join(root, directory)
            if not os.listdir(dir_path):  # Check if the directory is empty
                os.rmdir(dir_path)
                logger.info(f"Removed empty folder: {dir_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Hugging Face models for emotion classification.")
    parser.add_argument("--model-id", required=True, help="Hugging Face model ID")
    parser.add_argument("--tokenizer-id", required=True, help="Hugging Face tokenizer ID")
    parser.add_argument("--input-csv", required=True, help="Path to the input CSV file")
    parser.add_argument('--multiclass', action='store_true', help='Set this flag to use multiclass classification')
    parser.add_argument('--k', required=True, help='Cross-validation split')
    parser.add_argument('--use-class-weights', action='store_true', help='Use class weights in the loss function')

    return parser.parse_args()

def load_data(input_csv, multiclass):
    logger.info("Loading data from %s", input_csv)
    df = pd.read_csv(input_csv)
    logger.info("Data loaded. Number of rows: %d", len(df))
    logger.info("Dataset head:\n%s", df.head())
    
    valid_emotions = set(MULTICLASS_LABEL_MAP.keys())

    # Process only valid emotions
    df['emotion-A'] = df['emotion-A'].apply(lambda x: x if x in valid_emotions else None)
    df['emotion-B'] = df['emotion-B'].apply(lambda x: x if x in valid_emotions else None)

    # Remove rows where both 'emotion-A' and 'emotion-B' are invalid
    df = df.dropna(subset=['emotion-A', 'emotion-B'], how='all')

    if multiclass:
        # Use the first valid emotion (priority to 'emotion-A' if both are valid)
        df['labels'] = df[['emotion-A', 'emotion-B']].apply(
            lambda x: MULTICLASS_LABEL_MAP[x['emotion-A']] if pd.notna(x['emotion-A']) 
                      else MULTICLASS_LABEL_MAP[x['emotion-B']], 
            axis=1
        )
    else:
        binary_labels = {}
        for emotion in MULTICLASS_LABEL_MAP.keys():
            df[f"{emotion}_label"] = df[['emotion-A', 'emotion-B']].apply(
                lambda x: BINARY_LABEL_MAP['Positive'] if emotion in [x['emotion-A'], x['emotion-B']] else BINARY_LABEL_MAP['Negative'],
                axis=1
            )
            binary_labels[emotion] = df[f"{emotion}_label"].to_numpy()
        logger.info("Binary labels prepared for emotions.")
        return df, binary_labels
    
    logger.info("Filtered data. Number of rows: %d", len(df))
    return df


def display_emotion_counts(df):
    # Count occurrences of each emotion in emotion-A and emotion-B columns
    emotion_counts = df[['emotion-A', 'emotion-B']].stack().value_counts()
    
    # Display counts
    logger.info("Number of instances per emotion:")
    for emotion, count in emotion_counts.items():
        logger.info(f"{emotion}: {count}")

def display_emotion_counts_binary(df):
    #TODO
    logger.info("Number of instances per emotion:")
    logger.info("\t\tYes\t\tNo")
    for emotion in MULTICLASS_LABEL_MAP.keys():
        logger.info(f'{emotion}\t\t' + str((df[f'{emotion}_label'] == 1).sum()) + '\t\t' + str((df[f'{emotion}_label'] == 0).sum()))
    return

def preprocess_data(texts, labels, tokenizer):
    logger.info("Preprocessing data with tokenizer %s", tokenizer.name_or_path)
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    return encodings, labels

def compute_metrics(predictions):
    preds, labels = predictions
    preds = preds.argmax(axis=1)  # Get predicted class indices
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average=None)

    # Convert per-class metrics to a dictionary with class names
    per_class_metrics = {
        class_name: {
            "precision": precision[class_idx],
            "recall": recall[class_idx],
            "f1": f1[class_idx],
            "support": support[class_idx]
        }
        for class_name, class_idx in MULTICLASS_LABEL_MAP.items()
    }

    # Aggregate average metrics
    average_metrics = {
        "eval_accuracy": accuracy,
        "eval_precision": np.mean(precision),
        "eval_recall": np.mean(recall),
        "eval_f1": np.mean(f1)
    }

    # Return both average and per-class metrics
    return {"average_metrics": average_metrics, "per_class_metrics": per_class_metrics}


def compute_metrics_binary(predictions):
    preds, labels = predictions
    preds = preds.argmax(axis=1)  # For multi-class
    # For binary, ensure pos_label is correctly used
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1
    )
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def cross_validate_binary_models(model_id, tokenizer_id, texts, binary_labels, k=10, output_dir="./output", use_class_weights=False):
    logger.info("Starting binary cross-validation with k=%d", k)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    metrics = {}
    
    for emotion, labels in binary_labels.items():
        logger.info("Processing binary classifier for emotion: %s", emotion)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        
        fold_metrics = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):
            logger.info("Processing fold %d/%d for emotion %s", fold + 1, k, emotion)
            logger.info("Training instances %d", len(train_idx))
            logger.info("Testing instances %d", len(test_idx))
            
            # Split data
            train_texts, test_texts = texts[train_idx], texts[test_idx]
            train_labels, test_labels = labels[train_idx], labels[test_idx]

            # Class weights
            if use_class_weights:
                class_weights = compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(train_labels),
                    y=train_labels
                )
                class_weights = torch.tensor(class_weights, dtype=torch.float).to("cuda" if torch.cuda.is_available() else "cpu")
            else:
                class_weights = None
            
            # Preprocess
            train_encodings, train_labels = preprocess_data(train_texts.tolist(), train_labels.tolist(), tokenizer)
            test_encodings, test_labels = preprocess_data(test_texts.tolist(), test_labels.tolist(), tokenizer)
            
            train_dataset = [
                {key: torch.tensor(val[i]) for key, val in train_encodings.items()} | {"labels": torch.tensor(train_labels[i])}
                for i in range(len(train_labels))
            ]
            test_dataset = [
                {key: torch.tensor(val[i]) for key, val in test_encodings.items()} | {"labels": torch.tensor(test_labels[i])}
                for i in range(len(test_labels))
            ]
            
            # Load model
            model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
            
            # Define training arguments
            training_args = TrainingArguments(
                output_dir=os.path.join(output_dir, f"{emotion}_fold-{fold}"),
                num_train_epochs=10,
                per_device_train_batch_size=16,
                eval_strategy="epoch",
                save_strategy="no",
                #logging_dir=os.path.join(output_dir, f"{emotion}_fold-{fold}/logs"),
                report_to="none"
            )
            
            # Define Trainer
            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                compute_metrics=compute_metrics_binary,
                class_weights=class_weights
            )
            
            # Train and evaluate
            trainer.train()
            eval_metrics = trainer.evaluate()
            logger.info("Fold %d metrics for emotion %s: %s", fold + 1, emotion, eval_metrics)
            
            fold_metrics.append(eval_metrics)

            remove_empty_folders(output_dir)
        
        # Aggregate metrics for this emotion
        avg_metrics = aggregate_metrics(fold_metrics)
        metrics[emotion] = avg_metrics
        logger.info("Average metrics for emotion %s: %s", emotion, avg_metrics)
    
    # Save metrics
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    metrics_csv = os.path.join(output_dir, "binary_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=True)
    logger.info("Binary classifier metrics saved to %s", metrics_csv)
    return metrics

def cross_validate_model(model_id, tokenizer_id, texts, labels, k=10, multiclass=True, output_dir="./output", use_class_weights=False):
    logger.info("Starting cross-validation with k=%d", k)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    fold_metrics = []
    all_per_class_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):
        logger.info("Processing fold %d/%d", fold + 1, k)
        logger.info("Training instances %d", len(train_idx))
        logger.info("Testing instances %d", len(test_idx))
        
        # Split data
        train_texts, test_texts = texts[train_idx], texts[test_idx]
        train_labels, test_labels = labels[train_idx], labels[test_idx]

        # Class weights
        if use_class_weights:
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(train_labels),
                y=train_labels
            )
            class_weights = torch.tensor(class_weights, dtype=torch.float).to("cuda" if torch.cuda.is_available() else "cpu")
        else:
            class_weights = None
        
        # Preprocess
        train_encodings, train_labels = preprocess_data(train_texts.tolist(), train_labels.tolist(), tokenizer)
        test_encodings, test_labels = preprocess_data(test_texts.tolist(), test_labels.tolist(), tokenizer)
        
        # Replace TensorDataset with a list of dictionaries
        train_dataset = [
            {key: torch.tensor(val[i]) for key, val in train_encodings.items()} | {"labels": torch.tensor(train_labels[i])}
            for i in range(len(train_labels))
        ]
        test_dataset = [
            {key: torch.tensor(val[i]) for key, val in test_encodings.items()} | {"labels": torch.tensor(test_labels[i])}
            for i in range(len(test_labels))
        ]

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=len(MULTICLASS_LABEL_MAP) if multiclass else 2
        )
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, f"fold-{fold}"),
            num_train_epochs=10,
            per_device_train_batch_size=16,
            eval_strategy="epoch",
            save_strategy="no",
            report_to="none"
        )
        
        # Define Trainer
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            class_weights=class_weights
        )
        
        # Train and evaluate
        trainer.train()
        eval_metrics = trainer.evaluate()
        logger.info("Fold %d metrics: %s", fold + 1, eval_metrics)

        # Collect fold metrics
        fold_metrics.append(eval_metrics["eval_average_metrics"])
        all_per_class_metrics.append(eval_metrics["eval_per_class_metrics"])

        remove_empty_folders(output_dir)
    
    # Aggregate average metrics
    avg_metrics = aggregate_metrics(fold_metrics)
    logger.info("Average metrics across folds: %s", avg_metrics)

    # Aggregate per-class metrics
    aggregated_per_class_metrics = aggregate_per_class_metrics(all_per_class_metrics)

    # Save all metrics to a single CSV
    metrics_csv = os.path.join(output_dir, "multiclass_metrics.csv")
    save_metrics_to_csv(avg_metrics, aggregated_per_class_metrics, metrics_csv)
    logger.info("Detailed metrics for all folds saved to %s", metrics_csv)

    return avg_metrics

def aggregate_per_class_metrics(all_per_class_metrics):
    aggregated_metrics = {class_name: {"precision": [], "recall": [], "f1": [], "support": []} 
                          for class_name in MULTICLASS_LABEL_MAP.keys()}

    # Collect metrics for each class
    for per_class_metrics in all_per_class_metrics:
        for class_name, metrics in per_class_metrics.items():
            for metric, value in metrics.items():
                aggregated_metrics[class_name][metric].append(value)

    # Average across folds
    for class_name, metrics in aggregated_metrics.items():
        for metric, values in metrics.items():
            aggregated_metrics[class_name][metric] = np.mean(values).item()

    return aggregated_metrics

def save_metrics_to_csv(avg_metrics, per_class_metrics, file_path):
    """
    Save average metrics and per-class metrics to a CSV file.
    Columns: accuracy, precision, recall, f1
    Rows: 1 for each class + 1 for the average metrics.
    """
    # Prepare data for the CSV
    rows = []

    # Add per-class metrics
    for class_name, metrics in per_class_metrics.items():
        rows.append({
            "Class": class_name,
            "Accuracy": None,  # Not relevant for individual classes
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"]
        })

    # Add average metrics
    rows.append({
        "Class": "Average",
        "Accuracy": avg_metrics["accuracy"],
        "Precision": avg_metrics["precision"],
        "Recall": avg_metrics["recall"],
        "F1": avg_metrics["f1"]
    })

    # Create DataFrame and save to CSV
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(file_path, index=False)
    logger.info(f"Metrics saved successfully to {file_path}")

def compute_loss(model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.logits
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    loss = loss_fn(logits, labels)
    return (loss, outputs) if return_outputs else loss

def aggregate_metrics(fold_metrics):
    avg_metrics = {
        "accuracy": np.mean([metrics["eval_accuracy"] for metrics in fold_metrics]),
        "precision": np.mean([np.mean(metrics["eval_precision"]) for metrics in fold_metrics]),
        "recall": np.mean([np.mean(metrics["eval_recall"]) for metrics in fold_metrics]),
        "f1": np.mean([np.mean(metrics["eval_f1"]) for metrics in fold_metrics])
    }
    return avg_metrics

def save_metrics(metrics, output_csv):
    logger.info("Saving metrics to %s", output_csv)
    metrics_df = pd.DataFrame(metrics, index=[0])
    metrics_df.to_csv(output_csv, index=False)
    logger.info("Metrics saved successfully.")

def main():
    args = parse_args()
    logger.info("Arguments received: %s", args)
    
    if args.multiclass:
        df = load_data(args.input_csv, True)
        display_emotion_counts(df)
        texts = df['sentence'].to_numpy()
        labels = df['labels'].to_numpy()
        avg_metrics = cross_validate_model(
            model_id=args.model_id,
            tokenizer_id=args.tokenizer_id,
            texts=texts,
            labels=labels,
            k=int(args.k),
            multiclass=True,
            output_dir="./evaluation",
            use_class_weights=args.use_class_weights
        )
    else:
        df, binary_labels = load_data(args.input_csv, False)
        display_emotion_counts_binary(df)
        texts = df['sentence'].to_numpy()
        avg_metrics = cross_validate_binary_models(
            model_id=args.model_id,
            tokenizer_id=args.tokenizer_id,
            texts=texts,
            binary_labels=binary_labels,
            k=int(args.k),
            output_dir="./evaluation",
            use_class_weights=args.use_class_weights
        )
    
    logger.info("Cross-validation complete. Final metrics: %s", avg_metrics)

if __name__ == "__main__":
    main()
