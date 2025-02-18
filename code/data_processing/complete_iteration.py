import pandas as pd
import argparse
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Compare and merge Excel files based on compound key')
    parser.add_argument('file_a', help='Path to the first Excel file (file to be updated)')
    parser.add_argument('file_b', help='Path to the second Excel file (reference file)')
    return parser.parse_args()

def load_excel_files(file_a_path, file_b_path):
    logger = logging.getLogger(__name__)
    logger.info(f"Loading file A from: {file_a_path}")
    df_a = pd.read_excel(file_a_path)
    logger.info(f"File A loaded with {len(df_a)} rows")
    
    logger.info(f"Loading file B from: {file_b_path}")
    df_b = pd.read_excel(file_b_path)
    logger.info(f"File B loaded with {len(df_b)} rows")
    return df_a, df_b

def merge_files(df_a, df_b):
    logger = logging.getLogger(__name__)
    logger.info("Starting merge process")
    
    # Create a list to store rows in the final order
    result_rows = []
    
    # Keep track of rows we've already added from df_a
    used_a_indices = set()
    
    # Add counters
    used_b_count = 0
    used_a_count = 0
    
    total_rows = len(df_b)
    logger.info(f"Processing {total_rows} rows from file B")
    
    # Iterate through df_b to maintain its row order for insertions
    for idx, row_b in df_b.iterrows():
        if idx % 1000 == 0:  # Log progress every 1000 rows
            logger.info(f"Processing row {idx}/{total_rows}")
            
        # Try to find matching row in df_a
        mask_a = (df_a['reviewId'] == row_b['reviewId']) & (df_a['feature'] == row_b['feature'])
        matching_a = df_a[mask_a]
        
        if len(matching_a) > 0:
            idx_a = matching_a.index[0]
            if idx_a not in used_a_indices:
                row_a = df_a.loc[idx_a]
                # Check columns 10 through 19 (K through T in Excel)
                cols_to_check = list(df_a.columns[10:20])
                if all(int(row_a[col]) == 0 for col in cols_to_check):
                    logger.debug(f"Using B's version for reviewId={row_b['reviewId']}, feature={row_b['feature']} (A has all zeros)")
                    result_rows.append(row_b)
                    used_b_count += 1
                else:
                    logger.debug(f"Using A's version for reviewId={row_a['reviewId']}, feature={row_a['feature']} (A has non-zero values)")
                    result_rows.append(row_a)
                    used_a_count += 1
                used_a_indices.add(idx_a)
        else:
            logger.debug(f"Using B's version for reviewId={row_b['reviewId']}, feature={row_b['feature']} (not found in A)")
            result_rows.append(row_b)
            used_b_count += 1
    
    logger.info(f"Used file A's version: {used_a_count} times")
    logger.info(f"Used file B's version: {used_b_count} times")
    
    return pd.DataFrame(result_rows)

def main():
    logger = setup_logging()
    logger.info("Starting processing")
    
    args = parse_arguments()
    
    # Load both Excel files
    df_a, df_b = load_excel_files(args.file_a, args.file_b)
    
    # Merge files and maintain order
    result = merge_files(df_a, df_b)
    
    # Save the result back to file_a
    logger.info(f"Saving merged result to: {args.file_a}")
    result.to_excel(args.file_a, index=False)
    logger.info("Processing completed successfully")

if __name__ == "__main__":
    main()
