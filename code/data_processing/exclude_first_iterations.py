import os
import pandas as pd

# Parameters
input_file = 'data/output/reviews-15.csv'
exclude_file = 'data/output/iterations/already_annotated.csv'
output_folder = 'data/output/iterations/'  # Output folder path
start_iteration = 3  # Starting iteration number
rows_per_file = 100  # Number of rows per batch

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Read the CSV files
input_df = pd.read_csv(input_file)
exclude_df = pd.read_csv(exclude_file)

# Initial size of the input file
initial_input_size = len(input_df)

# Merge based only on 'reviewId' and 'sentenceId'
filtered_df = pd.merge(input_df, exclude_df, on=['reviewId', 'sentenceId'], how='outer', indicator=True)

# Count how many rows were found in both files (and thus removed)
rows_removed = len(filtered_df.query('_merge == "both"'))

# Filter the rows only in the input file (those not present in exclude)
filtered_df = filtered_df.query('_merge == "left_only"').drop(columns=['_merge'])

# Final size after filtering
final_input_size = len(filtered_df)

# Report the number of rows removed and the size after filtering
print(f"Initial size of input: {initial_input_size} rows")
print(f"Rows removed (found in exclude based on reviewId and sentenceId): {rows_removed} rows")
print(f"Final size of input after filtering: {final_input_size} rows")

# Splitting into batches of 100 rows and saving each batch to a separate CSV
for i in range(0, final_input_size, rows_per_file):
    batch_df = filtered_df.iloc[i:i + rows_per_file]
    iteration_number = start_iteration + (i // rows_per_file)
    output_filename = f'iteration_{iteration_number}.csv'
    output_path = os.path.join(output_folder, output_filename)  # Create the full output path
    batch_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
