import argparse
import numpy as np
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import evaluate
from sklearn.model_selection import train_test_split, StratifiedKFold
import logging

logging.basicConfig(level=logging.INFO)

FOLD_QTY = 10
LABEL_QTY = 10
LABEL_MAP = {
    'Joy': 0, 'Sadness': 1, 'Anger': 2, 'Fear': 3, 'Trust': 4, 'Disgust': 5,
    'Surprise': 6, 'Anticipation': 7, 'Neutral': 8#, 'Reject': 9
}
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def evaluate_metrics(trainer):
    return trainer.evaluate()


def load_trainer(model, trainer_args, tokenizer, train_split, test_split):
    return Trainer(
        model=model,
        args=trainer_args,
        train_dataset=train_split,
        eval_dataset=test_split,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )


def load_trainer_args(tag, model_name):
    return TrainingArguments(
        output_dir=f'./models/{model_name}_{tag}',
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=False,
    )


def load_tokenizer(tokenizer_id):
    return BertTokenizer.from_pretrained(tokenizer_id)


def preprocess(examples, tokenizer):
    tokens = tokenizer(
        examples['sentence'],
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors=None
    )

    labels = [LABEL_MAP.get(emotion, -1) for emotion in examples['emotion-primary-agreement']]

    if any(label >= LABEL_QTY for label in labels):
        raise ValueError("Label out of bounds!")

    return {
        'input_ids': tokens['input_ids'],
        'attention_mask': tokens['attention_mask'],
        'labels': labels
    }


def train(model, tokenizer, train_split, test_split, tag, model_name):
    trainer = load_trainer(model,
                        load_trainer_args(tag, model_name),
                        tokenizer,
                        train_split,
                        test_split)
    trainer.train()
    return trainer


def train_model(model, tokenizer, split_datasets, model_name, multiclass):
    all_fold_metrics = []
    if multiclass:
        for i in range(1, FOLD_QTY):
            train_split = split_datasets[f'train_{i}']
            test_split = split_datasets[f'test_{i}']
            logging.info(f'Training {i} split')
            trainer = train(model, tokenizer, train_split, test_split, i, model_name)
            all_fold_metrics.append(evaluate_metrics(trainer))
            #trainer.save_model(f"./{model_name}_{i}")
    else:
        for emotion in LABEL_MAP.keys():
            train_split = split_datasets[f'train_{emotion}']
            test_split = split_datasets[f'test_{emotion}']
            logging.info(f'Training {i} split')
            trainer = train(model, tokenizer, train_split, test_split, emotion, model_name)
            all_fold_metrics.append(evaluate_metrics(trainer))
            #trainer.save_model(f"./{model_name}_{emotion}")
    return all_fold_metrics


def load_hf_model(model_id):
    return BertForSequenceClassification.from_pretrained(model_id, num_labels=LABEL_QTY)


def load_hf_dataset(repository_id):
    return load_dataset(repository_id)


def preprocess_dataset(dataset, tokenizer, multiclass=True):
    def tokenize_and_process(example):
        # Tokenize the 'review' and 'sentence' fields
        review_tokens = tokenizer(
            example['review'], padding='max_length', truncation=True, max_length=128, return_tensors=None
        )
        sentence_tokens = tokenizer(
            example['sentence'], padding='max_length', truncation=True, max_length=128, return_tensors=None
        )
        
        combined_inputs = sentence_tokens['input_ids'] + [tokenizer.sep_token_id] + review_tokens['input_ids']
        combined_attention_mask = sentence_tokens['attention_mask'] + [1] + review_tokens['attention_mask']
        label = LABEL_MAP.get(example['emotion-primary-agreement'], -1)

        return {
            'input_ids': combined_inputs,
            'attention_mask': combined_attention_mask,
            'label': label
        }
    
    # Preprocess the entire dataset once
    processed_dataset = dataset['train'].map(tokenize_and_process, batched=False)

    split_datasets = {}

    if multiclass:
        # Perform k-fold cross-validation with k=10
        kf = StratifiedKFold(n_splits=FOLD_QTY, shuffle=True, random_state=42)
        labels = processed_dataset['label']

        for fold, (train_index, test_index) in enumerate(kf.split(np.zeros(len(labels)), labels)):
            train_split = processed_dataset.select(train_index)
            test_split = processed_dataset.select(test_index)

            split_datasets[f'train_{fold+1}'] = train_split
            split_datasets[f'test_{fold+1}'] = test_split

    else:
        # Create train-test partitions for each emotion category
        for emotion in LABEL_MAP.keys():
            emotion_dataset = processed_dataset.filter(lambda x: x['label'] == LABEL_MAP[emotion])
            train_split, test_split = train_test_split(emotion_dataset, test_size=0.2, stratify=emotion_dataset['label'])

            split_datasets[f'train_{emotion}'] = train_split
            split_datasets[f'test_{emotion}'] = test_split

    return split_datasets

def save_metrics_to_file(metrics, filename):
    with open(filename, 'w') as file:
        file.write("Evaluation Metrics per Fold:\n\n")
        for fold_index, metric in enumerate(metrics, 1):
            file.write(f"Fold {fold_index} Metrics:\n")
            for key, value in metric.items():
                file.write(f"{key}: {value}\n")
            file.write("\n")


def push_model_to_hf(model_id, tokenizer_id):
    model_name = model_id.split('/')[-1]
    for emotion in LABEL_MAP.keys():
        print(f"{emotion} model pushed to Hugging Face Hub.")
        model = BertForSequenceClassification.from_pretrained(f"./{model_name}_{emotion}")
        tokenizer = BertTokenizer.from_pretrained(tokenizer_id)
        model.push_to_hub(f"{model_name}_{emotion}")
        tokenizer.push_to_hub(f"{model_name}_{emotion}")


def main(args):
    # Set arguments
    logging.info("Loading dataset, model and tokenizer")
    dataset = load_hf_dataset(args.repository_id)
    model = load_hf_model(args.model_id)
    tokenizer = load_tokenizer(args.tokenizer_id)
    multiclass = args.multiclass
    
    # Split and tokenize dataset
    logging.info("Preprocessing and splitting data")
    split_datasets = preprocess_dataset(dataset, tokenizer, multiclass)

    # Train model
    logging.info("Training models")
    metrics = train_model(model, tokenizer, split_datasets, args.model_id.split('/')[-1], multiclass)
    
    # Save metrics
    save_metrics_to_file(metrics, 'metrics.txt')
    
    # Push model to HuggingFace
    #push_model_to_hf(args.model_id, args.tokenizer_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Training Parameters")

    parser.add_argument("--model-id", required=True, help="Hugging Face model ID")
    parser.add_argument("--tokenizer-id", required=True, help="Hugging Face tokenizer ID")
    parser.add_argument("--repository-id", required=True, help="Hugging Face dataset repository ID")
    parser.add_argument("--multiclass", required=True, help="Multiclass (True) or binary (False) classification")

    args = parser.parse_args()
    main(args)
