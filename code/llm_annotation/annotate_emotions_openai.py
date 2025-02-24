import pandas as pd
from openai import OpenAI
import json
import os
import time
import argparse
from datetime import datetime
from dotenv import load_dotenv
from annotate_emotions import EmotionAnnotator

# Load environment variables
load_dotenv()

class OpenAIEmotionAnnotator(EmotionAnnotator):
    def _initialize_client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key)

    def create_thread(self) -> str:
        """Create a new thread for conversation persistence."""
        return self.client.beta.threads.create().id

    def get_annotation(self, reviews_batch: list, **kwargs) -> tuple:
        """Get emotion annotations for a batch of reviews using OpenAI's Assistant API."""
        assistant_id = kwargs.get('assistant_id')
        temperature = kwargs.get('temperature', 0)
        user_prompt = json.dumps(reviews_batch, ensure_ascii=False)
        thread_id = self.create_thread()

        try:
            # Create message in thread
            self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=user_prompt
            )

            # Create and monitor run
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id,
                temperature=temperature
            )

            while True:
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id, 
                    run_id=run.id
                )
                if run_status.status == "completed":
                    break
                elif run_status.status not in ["in_progress", "queued"]:
                    print(f"Run status: {run_status.status}")
                time.sleep(2)

            # Get response
            messages = self.client.beta.threads.messages.list(thread_id=thread_id)
            last_message = messages.data[0].content[0].text.value
            annotations_list = self.validate_json(last_message)

            # Collect usage statistics
            usage_metadata = {
                'prompt_tokens': getattr(run_status.usage, "prompt_tokens", 0),
                'completion_tokens': getattr(run_status.usage, "completion_tokens", 0),
                'total_tokens': getattr(run_status.usage, "total_tokens", 0)
            }

            return annotations_list, usage_metadata

        except Exception as e:
            print(f"Error processing batch: {e}")
            default_annotation = {emotion: 0 for emotion in ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 
                                               'Disgust', 'Anger', 'Anticipation', 'Neutral', 'Reject']}
            default_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
            return [default_annotation] * len(reviews_batch), default_usage

def validate_json(response_text: str) -> list:
    """Ensure the response is a valid JSON object with a 'reviews' array matching the expected schema."""
    expected_emotions = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 
                         'Disgust', 'Anger', 'Anticipation', 'Neutral', 'Reject']
    
    try:
        cleaned_text = response_text.strip()
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith('```'):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3]

        response_json = json.loads(cleaned_text.strip())
        
        # Check if response has the expected structure
        if not isinstance(response_json, dict) or 'reviews' not in response_json:
            raise ValueError("Response is missing 'reviews' key")
            
        reviews = response_json['reviews']
        if not isinstance(reviews, list):
            raise ValueError("'reviews' is not a list")

        # Validate each annotation object
        for i, annotation in enumerate(reviews):
            for emotion in expected_emotions:
                if emotion not in annotation or not isinstance(annotation[emotion], int):
                    annotation[emotion] = 0

        return reviews

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Warning: Failed to parse JSON: {e}")
        print(response_text)
        return [{emotion: 0 for emotion in expected_emotions}]

def main(input_file: str, output_folder: str, batch_size: int = 5, n: int = None, model: str = "GPT-4o", temperature: float = 0):
    print(f"Starting annotation process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df = pd.read_excel(input_file)
    if n is not None:
        df = df.head(n)

    annotator = OpenAIEmotionAnnotator(api_key=os.getenv("OPENAI_API_KEY"))
    assistant_id = annotator.get_assistant(model)

    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f'{model}-annotations.xlsx')

    results_df = pd.DataFrame()
    total_tokens_used = 0
    total_prompt_tokens_used = 0
    total_completion_tokens_used = 0

    # Process reviews in batches
    for i in range(0, len(df), batch_size):
        batch_start_time = time.time()
        batch = df.iloc[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")

        # Prepare batch data
        batch_data = [{"review": row['review'], "sentence": row['sentence']} for _, row in batch.iterrows()]
        
        annotations_list, usage_metadata = annotator.get_annotation(batch_data, assistant_id=assistant_id, temperature=temperature)

        batch_time = time.time() - batch_start_time

        # Update running totals
        total_tokens_used += usage_metadata['total_tokens']
        total_prompt_tokens_used += usage_metadata['prompt_tokens']
        total_completion_tokens_used += usage_metadata['completion_tokens']

        # Update results for each review in the batch
        for j, (_, row) in enumerate(batch.iterrows()):
            if j < len(annotations_list):
                annotations = annotations_list[j]
                annotations["annotation_time"] = batch_time / len(batch)
                row.update(annotations)
                results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
                results_df.to_excel(output_file, index=False, engine='openpyxl')

        print(f"Batch {i//batch_size + 1} processed (Time: {batch_time:.2f}s, Batch Tokens: {usage_metadata['total_tokens']}, Batch Prompt Tokens: {usage_metadata['prompt_tokens']}, Batch Completion Tokens: {usage_metadata['completion_tokens']})")
        time.sleep(10)  # Rate limiting between batches

    print(f"Total tokens used: {total_tokens_used}")
    print(f"Total prompt tokens used: {total_prompt_tokens_used}")
    print(f"Total completion tokens used: {total_completion_tokens_used}")
    print("Annotation process complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run emotion annotation with OpenAI Assistants API')
    parser.add_argument('--input', required=True, help='Input Excel file')
    parser.add_argument('--output', required=True, help='Output folder')
    parser.add_argument('--batch_size', type=int, default=10, 
                       help='Number of reviews to process in each batch')
    parser.add_argument('--n', type=int, help='Number of rows to process (optional)')
    parser.add_argument('--model', type=str, default='gpt-4', 
                       help='Model name for assistant lookup')
    parser.add_argument('--temperature', type=float, default=0, 
                       help='Temperature for model responses (0-2)')
    parser.add_argument('--sleep_time', type=float, default=10, 
                       help='Sleep time between batches')
    
    args = parser.parse_args()
    main(args.input, args.output, args.batch_size, args.n, args.model, args.temperature)
