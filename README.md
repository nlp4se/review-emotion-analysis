# review-emotion-analysis

Example for fine-tuning model:

```python.exe .\code\classifier\fine-tuning.py --model-id bert-base-uncased --tokenizer-id bert-base-uncased --input-csv .\data\ground-truth\ground-truth.csv --multiclass --k 10```

Example of random selector:

```python.exe .\code\random_selector.py .\data\output\reviews-15.csv .\data\output\iterations\iteration_4.csv 100 --exclude_files .\data\output\iterations\iteration_0.csv .\data\output\iterations\iteration_1.csv .\data\output\iterations\iteration_2.csv .\data\output\iterations\iteration_3.csv```

Example of GPT-4o annotation:

```python .\code\generative_annotation\annotate-gpt-4o.py .\data\ground-truth\ground-truth.xlsx .\data\instructions\guidelines.txt .\data\output\generative_annotation\```
