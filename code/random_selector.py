import csv
import random
import argparse
import pandas as pd

def select_random_rows(input_file, output_file, n, exclude_files=None):
    # Read the input CSV
    with open(input_file, mode='r', encoding='utf-8') as infile:
        reader = list(csv.reader(infile))
        header, rows = reader[0], reader[1:]

    if exclude_files:
        # Read the main dataframe
        df1 = pd.read_csv(input_file)

        # Read and merge all exclude CSVs into a single dataframe
        exclude_dfs = [pd.read_csv(file) for file in exclude_files]
        df2 = pd.concat(exclude_dfs).drop_duplicates().reset_index(drop=True)

        print(len(df1))
        print(len(df2))

        # Merge and remove the common rows
        filtered_df = df1.merge(df2, indicator=True, how='left').loc[lambda x: x['_merge'] != 'both']
        filtered_df = filtered_df.drop(columns=['_merge'])
        
        # Remove any potential duplicates in the filtered dataframe
        filtered_df = filtered_df.drop_duplicates().reset_index(drop=True)

        # If N is greater than the number of available rows, adjust N
        n = min(n, len(filtered_df))

        # Convert the filtered dataframe to a list of rows
        rows = filtered_df.values.tolist()

    # Select N random rows
    if n > len(rows):
        n = len(rows)
    selected_rows = random.sample(rows, n)

    # Write to the output CSV
    with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)  # Write the header
        writer.writerows(selected_rows)  # Write the selected rows

def main():
    parser = argparse.ArgumentParser(description='Select random rows from a CSV file.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file')
    parser.add_argument('output_file', type=str, help='Path to the output CSV file')
    parser.add_argument('n', type=int, help='Number of random rows to select')
    parser.add_argument('--exclude_files', type=str, nargs='*', default=None, help='Paths to the optional exclude CSV files')
    args = parser.parse_args()

    select_random_rows(args.input_file, args.output_file, args.n, args.exclude_files)

if __name__ == '__main__':
    main()
