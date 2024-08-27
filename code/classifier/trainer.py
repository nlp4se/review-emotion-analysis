import os
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import evaluate

load_dotenv()


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
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )


def load_trainer_args(emotion, model_name):
    return TrainingArguments(
        output_dir=f'./{model_name}_{emotion}',
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
    )


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
        'input_ids': tokens['input_ids'],
        'attention_mask': tokens['attention_mask'],
        'labels': labels
    }


def train(model, tokenizer, train_split, test_split, emotion, model_name):
    trainer = load_trainer(model,
                           load_trainer_args(emotion, model_name),
                           tokenizer,
                           train_split,
                           test_split)
    trainer.train()
    return trainer


def train_model(model, tokenizer, dataset):
    model_name = os.getenv("MODEL_ID").split('/')[-1]
    all_fold_metrics = []
    for emotion in LABEL_MAP.keys():

        train_split = dataset[f'{emotion}_train']
        test_split = dataset[f'{emotion}_test']
        trainer = train(model, tokenizer, train_split, test_split, emotion, model_name)
        all_fold_metrics.append(evaluate_metrics(trainer))
        trainer.save_model(f"./{model_name}_{emotion}")
    return all_fold_metrics


def load_hf_model():
    return BertForSequenceClassification.from_pretrained(os.getenv("MODEL_ID"), num_labels=LABEL_QTY)


def load_hf_dataset():
    return load_dataset(os.getenv("REPOSITORY_K10_BY_SENTIMENT_ID"))


def preprocess_dataset(dataset, tokenizer):
    for emotion in LABEL_MAP.keys():
        train_split = dataset[f'{emotion}_train']
        test_split = dataset[f'{emotion}_test']

        train_split = train_split.map(lambda x: preprocess(x, tokenizer), batched=True)
        test_split = test_split.map(lambda x: preprocess(x, tokenizer), batched=True)

        dataset[f'{emotion}_train'] = train_split
        dataset[f'{emotion}_test'] = test_split

def save_metrics_to_file(metrics, filename):
    with open(filename, 'w') as file:
        file.write("Evaluation Metrics per Fold:\n\n")
        for fold_index, metric in enumerate(metrics, 1):
            file.write(f"Fold {fold_index} Metrics:\n")
            for key, value in metric.items():
                file.write(f"{key}: {value}\n")
            file.write("\n")


def push_model_to_hf():
    model_name = os.getenv("MODEL_ID").split('/')[-1]
    for emotion in LABEL_MAP.keys():
        print(f"{emotion} model pushed to Hugging Face Hub.")
        model = BertForSequenceClassification.from_pretrained(f"./{model_name}_{emotion}")
        tokenizer = BertTokenizer.from_pretrained(os.getenv("TOKENIZER_ID"))
        model.push_to_hub(f"{model_name}_{emotion}")
        tokenizer.push_to_hub(f"{model_name}_{emotion}")


def main():
    dataset = load_hf_dataset()
    model = load_hf_model()
    tokenizer = load_tokenizer()
    preprocess_dataset(dataset, tokenizer)

    metrics = train_model(model, tokenizer, dataset)
    save_metrics_to_file(metrics, 'metrics.txt')
    push_model_to_hf()


if __name__ == '__main__':
    main()
