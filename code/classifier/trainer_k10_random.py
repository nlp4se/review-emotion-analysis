import os
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import evaluate

load_dotenv()

# Number of labels and folds
FOLD_QTY = 10
LABEL_QTY = 10
LABEL_MAP = {
    'Joy': 0, 'Sadness': 1, 'Anger': 2, 'Fear': 3, 'Trust': 4, 'Disgust': 5,
    'Surprise': 6, 'Anticipation': 7, 'Neutral': 8, 'Reject': 9
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
        compute_metrics=compute_metrics
    )


def load_trainer_args(fold_index):
    '''return TrainingArguments(
        output_dir=f'./results/fold_{fold_index}',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
    )'''
    return TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")


def load_tokenizer():
    return BertTokenizer.from_pretrained(os.getenv("TOKENIZER_ID"))


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
        'sentence': examples['sentence'],
        'emotion-primary-agreement': examples['emotion-primary-agreement'],
        'input_ids': tokens['input_ids'],
        'attention_mask': tokens['attention_mask'],
        'labels': labels
    }


def train(model, tokenizer, train_split, test_split, fold_index):
    trainer = load_trainer(model,
                           load_trainer_args(fold_index),
                           tokenizer,
                           train_split,
                           test_split)
    trainer.train()
    return trainer


def train_model(model, tokenizer, dataset):
    all_fold_metrics = []
    for fold in range(1, FOLD_QTY + 1):  # Folds go from 1 to 10
        train_split = dataset[f'train_fold_{fold}']
        test_split = dataset[f'test_fold_{fold}']
        trainer = train(model, tokenizer, train_split, test_split, fold)
        all_fold_metrics.append(evaluate_metrics(trainer))
        trainer.save_model(f"./model_fold_{fold}")
    return all_fold_metrics


def load_hf_model():
    return BertForSequenceClassification.from_pretrained(os.getenv("MODEL_ID"))


def load_hf_dataset():
    return load_dataset(os.getenv("REPOSITORY_K10_ID"))


def preprocess_dataset(dataset, tokenizer):
    for fold in range(1, FOLD_QTY + 1):  # Folds go from 1 to 10
        train_split = dataset[f'train_fold_{fold}']
        test_split = dataset[f'test_fold_{fold}']

        train_split = train_split.map(lambda x: preprocess(x, tokenizer), batched=True)
        test_split = test_split.map(lambda x: preprocess(x, tokenizer), batched=True)

        dataset[f'train_fold_{fold}'] = train_split
        dataset[f'test_fold_{fold}'] = test_split


def main():
    dataset = load_hf_dataset()
    model = load_hf_model()
    tokenizer = load_tokenizer()
    preprocess_dataset(dataset, tokenizer)

    metrics = train_model(model, tokenizer, dataset)
    print(metrics)


if __name__ == '__main__':
    main()
