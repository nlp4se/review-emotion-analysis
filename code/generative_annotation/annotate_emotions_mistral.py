import mistralai
import pandas as pd
import json
import os
import time
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Mistral client
client = mistralai.MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))

def load_assistant_id(model: str) -> str:
    """Load the assistant ID from a model-specific file if it exists."""
    assistant_id_file = f"assistant_id_{model}.txt"
    if os.path.exists(assistant_id_file):
        with open(assistant_id_file, 'r') as file:
            return file.read().strip()
    return None

def get_assistant(model: str) -> str:
    """Check for an existing assistant ID or raise an error if not found."""
    assistant_id = load_assistant_id(model)
    
    if assistant_id:
        print(f"Reusing existing assistant ID for {model}: {assistant_id}")
        return assistant_id
    
    raise ValueError(f"Assistant ID not found for {model}. Run create_assistant.py first.")

def get_annotation(assistant_id: str, reviews_batch: list) -> list:
    """Classify emotions for a batch of reviews."""
    user_prompt = json.dumps(reviews_batch, ensure_ascii=False)
    
    try:
        response = client.generate(
            model=assistant_id,
            prompt=user_prompt,
            temperature=0.0
        )
        
        annotations_list = validate_json(response['choices'][0]['message']['content'])
        
        # Extract token usage if available
        prompt_tokens = response.get('usage', {}).get('prompt_tokens', 0)
        completion_tokens = response.get('usage', {}).get('completion_tokens', 0)
        total_tokens = response.get('usage', {}).get('total_tokens', 0)

        for annotation in annotations_list:
            annotation["prompt_tokens"] = prompt_tokens // len(annotations_list)
            annotation["completion_tokens"] = completion_tokens // len(annotations_list)
            annotation["total_tokens"] = total_tokens // len(annotations_list)

        return annotations_list

    except Exception as e:
        print(f"Error processing batch: {e}")
        default_annotation = {emotion: 0 for emotion in ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 
                                           'Disgust', 'Anger', 'Anticipation', 'Neutral', 'Reject']}
        return [default_annotation] * len(reviews_batch)

def validate_json(response_text: str) -> list:
    """Ensure the response is a valid JSON object with a 'reviews' array matching the expected schema."""
    expected_emotions = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 
                         'Disgust', 'Anger', 'Anticipation', 'Neutral', 'Reject']
    
    try:
        response_json = json.loads(response_text.strip())
        
        if not isinstance(response_json, list):
            raise ValueError("Response is not a list")
        
        for annotation in response_json:
            for emotion in expected_emotions:
                if emotion not in annotation or not isinstance(annotation[emotion], int):
                    annotation[emotion] = 0

        return response_json

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Warning: Failed to parse JSON: {e}")
        return [{emotion: 0 for emotion in expected_emotions}]

def main(input_file: str, output_folder: str, batch_size: int = 5, n: int = None, model: str = "mistral-large-latest"):
    start_time = datetime.now()
    print(f"Starting annotation process at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    df = pd.read_excel(input_file)
    if n is not None:
        df = df.head(n)

    assistant_id = get_assistant(model)

    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f'{model}-annotations.xlsx')

    results_df = pd.DataFrame()
    total_tokens_used = 0
    total_prompt_tokens_used = 0
    total_completion_tokens_used = 0

    for i in range(0, len(df), batch_size):
        batch_start_time = time.time()
        batch = df.iloc[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")

        batch_data = [{"review": row['review'], "sentence": row['sentence']} for _, row in batch.iterrows()]
        
        annotations_list = get_annotation(assistant_id, batch_data)

        for j, (_, row) in enumerate(batch.iterrows()):
            if j < len(annotations_list):
                row.update(annotations_list[j])
                total_tokens_used += annotations_list[j].get("total_tokens", 0)
                total_prompt_tokens_used += annotations_list[j].get("prompt_tokens", 0)
                total_completion_tokens_used += annotations_list[j].get("completion_tokens", 0)
                results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
                results_df.to_excel(output_file, index=False, engine='openpyxl')
        
        batch_time = time.time() - batch_start_time
        print(f"Batch {i//batch_size + 1} processed in {batch_time:.2f}s")
        time.sleep(10)

    end_time = datetime.now()
    print(f"Total tokens used: {total_tokens_used}")
    print(f"Total prompt tokens used: {total_prompt_tokens_used}")
    print(f"Total completion tokens used: {total_completion_tokens_used}")
    print(f"Annotation process completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')} (Duration: {(end_time - start_time).total_seconds():.2f}s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run emotion annotation with Mistral AI')
    parser.add_argument('--input', required=True, help='Input Excel file')
    parser.add_argument('--output', required=True, help='Output folder')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of reviews to process in each batch')
    parser.add_argument('--n', type=int, help='Number of rows to process (optional)')
    parser.add_argument('--model', type=str, default='mistral-large-latest', help='Model name for output file')
    args = parser.parse_args()
    main(args.input, args.output, args.batch_size, args.n, args.model)
