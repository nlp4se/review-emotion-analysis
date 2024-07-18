import json
import random
import csv
import sys
import os

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def save_csv(data, file_path):
    with open(file_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['app_name', 'categoryId', 'reviewId', 'at', 'score', 'review','feature'])
        for row in data:
            writer.writerow(row)

def filter_reviews(app_data, review_ids):
    for app in app_data:
        app['reviews'] = [review for review in app['reviews'] if review['reviewId'] in review_ids]
    return app_data

def select_random_reviews(app_data, review_count_dict, n):
    selected_reviews = []
    
    for feature in review_count_dict.keys():
        review_set = review_count_dict[feature]
        selected_items = random.sample(review_set, min(len(review_set), n))
        for app in app_data:
            for review in app['reviews']:
                if any(review['reviewId'] in item for item in selected_items):
                    selected_reviews.append({'app_name': app['app_name'], 'categoryId': app['categoryId'], 'reviewId': review['reviewId'], 'at': review['at'], 'score': review['score'], 'review': review['review'], 'feature': feature})
    
    print(len(selected_reviews))
    return selected_reviews

def transform_to_csv_format(app_data):
    csv_data = []
    for review in app_data:
        csv_data.append([review['app_name'], review['categoryId'], review['reviewId'], review['at'], review['score'], review['review'], review['feature']])
    return csv_data

def main(apps_reviews_path, review_ids_path, output_path, n):
    # Step 1: Load the JSON files
    app_data = load_json(apps_reviews_path)
    review_ids_dict = load_json(review_ids_path)
    
    # Step 2: Extract all review ids from the dictionary
    all_review_ids = set()
    for ids in review_ids_dict.values():
        all_review_ids.update(ids)
        
    all_review_ids_c = set()
    for review_id in all_review_ids:
        all_review_ids_c.add(review_id.split("_")[0])
            
    # Step 3: Filter reviews in the app data
    filtered_app_data = filter_reviews(app_data, all_review_ids_c)
    
    # Step 4: Save the filtered JSON
    save_json(filtered_app_data, os.path.join(output_path, 'reviews.json'))
    
    # Step 5: Select random reviews
    selected_reviews = select_random_reviews(filtered_app_data, review_ids_dict, n)
    
    # Step 7: Transform the JSON data to CSV format
    csv_data = transform_to_csv_format(selected_reviews)
    
    # Save the CSV output
    save_csv(csv_data, os.path.join(output_path, f'reviews-{n}.csv'))

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <apps_reviews_path> <review_ids_path> <output_path> <n>")
        sys.exit(1)

    apps_reviews_path = sys.argv[1]
    review_ids_path = sys.argv[2]
    output_path = sys.argv[3]
    n = int(sys.argv[4])

    main(apps_reviews_path, review_ids_path, output_path, n)
