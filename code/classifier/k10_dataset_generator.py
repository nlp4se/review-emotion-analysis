import os
import pandas as pd
import datasets
from sklearn.model_selection import KFold
from dotenv import load_dotenv

load_dotenv()
CSV_PATH = 'datasets/csv'
K10_SPLIT_PATH = 'datasets/splits/k10'
COLUMNS = ['app_name', 'categoryId', 'reviewId', 'sentenceId', 'feature', 'review', 'sentence',
           'emotion-primary-agreement']


def read_csv_file(csv_file):
    csv = pd.read_csv(CSV_PATH + '/' + csv_file)
    cleaned_csv = csv[COLUMNS]
    stringified_csv = cleaned_csv.applymap(str)
    return stringified_csv


def merge_csv_files(csv_files):
    return pd.concat(csv_files)


def generate_k_fold_datasets(file, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold = 1
    for train_index, test_index in kf.split(file):
        train_data = file.iloc[train_index]
        test_data = file.iloc[test_index]
        yield {'fold': fold, 'train': train_data, 'test': test_data}
        fold += 1


def save_csv_datasets(fold_datasets):
    for fold_dataset in fold_datasets:
        fold = fold_dataset['fold']
        train_path = os.path.join(K10_SPLIT_PATH, f'fold_{fold}/training')
        test_path = os.path.join(K10_SPLIT_PATH, f'fold_{fold}/test')

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        fold_dataset['train'].to_csv(os.path.join(train_path, 'training_dataset.csv'), index=False)
        fold_dataset['test'].to_csv(os.path.join(test_path, 'test_dataset.csv'), index=False)


def generate_and_push_hf_datasets(k=10):
    data_files = {}

    for fold in range(1, k + 1):
        data_files[f'train_fold_{fold}'] = os.path.join(K10_SPLIT_PATH, f'fold_{fold}/training/training_dataset.csv')
        data_files[f'test_fold_{fold}'] = os.path.join(K10_SPLIT_PATH, f'fold_{fold}/test/test_dataset.csv')

    hf_datasets = datasets.load_dataset('csv', data_files=data_files)

    hf_datasets.push_to_hub(
        repo_id=os.getenv("REPOSITORY_K10_ID"),
        token=os.getenv("HF_TOKEN"))


def main():
    csv_files = os.listdir(CSV_PATH)
    read_csv_files = []

    for csv_file in csv_files:
        read_csv_files.append(read_csv_file(csv_file))

    merged_csv = merge_csv_files(read_csv_files)

    k_fold_datasets = list(generate_k_fold_datasets(merged_csv))

    save_csv_datasets(k_fold_datasets)

    generate_and_push_hf_datasets()


if __name__ == '__main__':
    main()
