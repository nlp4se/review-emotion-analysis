import sys
import csv
from collections import defaultdict

def compute_fleiss_kappa(data):
    num_items = len(data)
    num_raters = len(data[0]) - 2

    # Count the number of times each category is chosen for each item
    category_counts = defaultdict(lambda: defaultdict(int))
    for item in data:
        for i in range(num_raters):
            category = item[i + 2]
            category_counts[item[0]][category] += 1

    # Compute the observed agreement
    observed_agreement = sum((sum((category_counts[item[0]][category] ** 2 for category in category_counts[item[0]])) - num_raters) / (num_raters * (num_raters - 1)) for item in data) / num_items

    # Compute the expected agreement
    total_counts = defaultdict(int)
    for item in data:
        for category in category_counts[item[0]]:
            total_counts[category] += category_counts[item[0]][category]
    expected_agreement = sum((total_counts[category] ** 2 - num_raters) / (num_raters * (num_raters - 1)) for category in total_counts) / (num_items * num_raters)

    # Compute Fleiss kappa
    fleiss_kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)

    return fleiss_kappa

def main():
    if len(sys.argv) != 2:
        print("Usage: python fleiss_kappa.py <csv_file>")
        return

    csv_file = sys.argv[1]

    # Read the CSV file
    data = []
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)

    # Compute Fleiss kappa
    fleiss_kappa = compute_fleiss_kappa(data)
    print("Fleiss kappa:", fleiss_kappa)

if __name__ == "__main__":
    main()