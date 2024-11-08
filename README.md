# review-emotion-analysis

Example for fine-tuning model:

```python.exe .\code\classifier\trainer.py --model-id 'google-bert/bert-base-uncased' --tokenizer-id 'google-bert/bert-base-uncased' --repository-id 'quim-motger/review-emotions' --multiclass True```

Example of random selector:

```python.exe .\code\random_selector.py .\data\output\reviews-15.csv .\data\output\iterations\iteration_4.csv 100 --exclude_files .\data\output\iterations\iteration_0.csv .\data\output\iterations\iteration_1.csv .\data\output\iterations\iteration_2.csv .\data\output\iterations\iteration_3.csv```