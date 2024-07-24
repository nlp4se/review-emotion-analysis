import csv

# Read the replacement dictionary
replacements = {}
with open('data/input/features.csv', mode='r', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    replacements = {rows[0]: rows[1] for rows in reader}

# Read the input CSV, replace features, and write to a new CSV
with open('data/output/reviews-15.csv', mode='r', encoding='utf-8') as infile, open('data/output/reviews-15-replaced.csv', mode='w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    header = next(reader)
    writer.writerow(header)
    for row in reader:
        row[6] = replacements.get(row[6], row[6])
        writer.writerow(row)
