import pandas as pd
import os
import sys

def add_iteration_descriptor(folder_path: str, xlsx_file_path: str):
    # Read the xlsx file
    df_xlsx = pd.read_excel(xlsx_file_path)
    
    # Get all CSV files from the folder that match the pattern
    csv_files = [f for f in os.listdir(folder_path) if f.startswith('iteration_') and f.endswith('.csv')]
    
    # Create a dictionary to store all CSV data for faster lookup
    csv_data = {}
    for csv_file in csv_files:
        iteration_name = csv_file.split('.')[0]  # Get filename without extension
        df_csv = pd.read_csv(os.path.join(folder_path, csv_file))
        # Create tuples of reviewId and feature for each row
        csv_data[iteration_name] = set(zip(df_csv['reviewId'], df_csv['feature']))
    
    # Create new column for iteration names
    iteration_column = []
    
    # Check each row in xlsx file
    for _, row in df_xlsx.iterrows():
        review_id = row['reviewId']
        feature = row['feature']
        found_iteration = None
        
        # Look for matching pair in CSV files
        for iteration_name, pairs in csv_data.items():
            if (review_id, feature) in pairs:
                found_iteration = iteration_name
                break
        
        iteration_column.append(found_iteration)
    
    # Insert the new column at the beginning
    df_xlsx.insert(0, 'iteration', iteration_column)
    
    # Generate output filename
    output_file = xlsx_file_path.rsplit('.', 1)[0] + '_with_iterations.xlsx'
    
    # Save the modified file
    df_xlsx.to_excel(output_file, index=False)
    print(f"Processed file saved as: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python add_iteration_descriptor.py <folder_path> <xlsx_file_path>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    xlsx_file_path = sys.argv[2]
    add_iteration_descriptor(folder_path, xlsx_file_path)
