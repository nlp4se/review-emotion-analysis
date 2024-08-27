import os
import pandas as pd
import datasets
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()
CSV_PATH = 'datasets/csv'
K10_SPLIT_PATH = 'datasets/splits/k10'
COLUMNS = ['app_name', 'categoryId', 'reviewId', 'sentenceId', 'feature', 'review', 'sentence',
           'emotion-primary-agreement']

LABEL_MAP = {
    'Joy': 0, 'Sadness': 1, 'Anger': 2, 'Fear': 3, 'Trust': 4, 'Disgust': 5,
    'Surprise': 6, 'Anticipation': 7, 'Neutral': 8, 'Reject': 9
}


def read_csv_file(csv_file):
    csv = pd.read_csv(CSV_PATH + '/' + csv_file)
    cleaned_csv = csv[COLUMNS]
    stringified_csv = cleaned_csv.applymap(str)
    return stringified_csv


def merge_csv_files(csv_files):
    return pd.concat(csv_files)


def generate_train_test_split_by_emotion(file):
    for emotion in LABEL_MAP.keys():
        emotion_data = file[file['emotion-primary-agreement'] == emotion]

        if len(emotion_data) < 2:
            print(f"Skipping emotion {emotion} due to insufficient data.")
            continue

        train_data, test_data = train_test_split(
            emotion_data,
            test_size=0.2,
            random_state=42
        )

        yield {'emotion': emotion, 'train': train_data, 'test': test_data}


def save_csv_datasets_by_emotion(train_test_splits):
    for split in train_test_splits:
        emotion = split['emotion']
        train_path = os.path.join(K10_SPLIT_PATH, f'{emotion}/training')
        test_path = os.path.join(K10_SPLIT_PATH, f'{emotion}/test')

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        split['train'].to_csv(os.path.join(train_path, 'training_dataset.csv'), index=False)
        split['test'].to_csv(os.path.join(test_path, 'test_dataset.csv'), index=False)


def generate_and_push_hf_datasets():
    data_files = {}

    for emotion in LABEL_MAP.keys():
        train_path = os.path.join(K10_SPLIT_PATH, f'{emotion}/training/training_dataset.csv')
        test_path = os.path.join(K10_SPLIT_PATH, f'{emotion}/test/test_dataset.csv')

        data_files[f'{emotion}_train'] = train_path
        data_files[f'{emotion}_test'] = test_path

    hf_datasets = datasets.load_dataset('csv', data_files=data_files)

    hf_datasets.push_to_hub(
        repo_id=os.getenv("REPOSITORY_K10_BY_SENTIMENT_ID"),
        token=os.getenv("HF_TOKEN"))


def main():
    csv_files = os.listdir(CSV_PATH)
    read_csv_files = []

    for csv_file in csv_files:
        read_csv_files.append(read_csv_file(csv_file))

    merged_csv = merge_csv_files(read_csv_files)

    train_test_splits = list(generate_train_test_split_by_emotion(merged_csv))

    save_csv_datasets_by_emotion(train_test_splits)

    generate_and_push_hf_datasets()


if __name__ == '__main__':
    main()
