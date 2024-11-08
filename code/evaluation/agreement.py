import pandas as pd
from collections import Counter

# Step 1: Read the CSV file
file_path = 'data/annotations/iteration_4.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Step 2: Define a function to find the most common and second most common values
def get_common_annotations(row):
    # Gather all annotations
    annotations = [
        row[9], row[10],  # Annotator 1
        row[12], row[13],  # Annotator 2
        row[15], row[16]   # Annotator 3
    ]
    
    # Filter out NaN values
    annotations = [ann for ann in annotations if pd.notna(ann)]
    
    # Count occurrences of each annotation
    counter = Counter(annotations)
    
    # Filter annotations appearing at least twice
    common_annotations = [key for key, count in counter.items() if count >= 2]
    
    # Sort by frequency (descending) and then by value (ascending)
    common_annotations.sort(key=lambda x: (-counter[x], x))
    
    # Return most common and second most common values (or None if not available)
    return (
        common_annotations[0] if len(common_annotations) > 0 else None,
        common_annotations[1] if len(common_annotations) > 1 else None
    )


# Step 3: Apply the function to each row and store results in columns 12 and 13
df['emotion-A-agreement'] = None
df['emotion-B-agreement'] = None

for index, row in df.iterrows():
    most_common, second_most_common = get_common_annotations(row)
    print(most_common, second_most_common)
    df.at[index, 'emotion-A-agreement'] = most_common
    df.at[index, 'emotion-B-agreement'] = second_most_common

print(df)

# Step 4: Save the resulting DataFrame to a new CSV (optional)
output_file = 'data/annotations/iteration_4_agreement.csv'  # Replace with your desired output file name
df.to_csv(output_file, index=False)

print("Processing complete. Results saved to:", output_file)
