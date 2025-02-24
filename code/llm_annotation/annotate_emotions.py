from abc import ABC, abstractmethod
import pandas as pd
import json
import os
import time
from datetime import datetime
from typing import Any, List, Dict, Tuple
from dotenv import load_dotenv

class EmotionAnnotator(ABC):
    """Abstract base class for emotion annotation across different AI providers."""
    
    def __init__(self, api_key: str):
        """Initialize the annotator with API key and default values."""
        self.api_key = api_key
        self.client = self._initialize_client()
        self.expected_emotions = [
            'Joy', 'Trust', 'Fear', 'Surprise', 'Sadness',
            'Disgust', 'Anger', 'Anticipation', 'Neutral', 'Reject'
        ]

    @abstractmethod
    def _initialize_client(self) -> Any:
        """Initialize the specific AI provider's client."""
        pass

    def load_assistant_id(self, model: str) -> str:
        """Load the assistant ID from a model-specific file if it exists."""
        assistant_id_file = f"data/assistants/assistant_id_{model}.txt"
        if os.path.exists(assistant_id_file):
            with open(assistant_id_file, 'r') as file:
                return file.read().strip()
        return None

    def get_assistant(self, model: str) -> str:
        """Check for an existing assistant ID or raise an error if not found."""
        assistant_id = self.load_assistant_id(model)
        if assistant_id:
            print(f"Reusing existing assistant ID for {model}: {assistant_id}")
            return assistant_id
        raise ValueError(f"Assistant ID not found for {model}. Run create_assistant.py first.")

    @abstractmethod
    def get_annotation(self, reviews_batch: List[Dict], **kwargs) -> Tuple[List[Dict], Dict]:
        """Get emotion annotations for a batch of reviews.
        
        Returns:
            Tuple containing:
            - List of annotation dictionaries
            - Dictionary with token usage information
        """
        pass

    def validate_json(self, content: str) -> List[Dict]:
        """Validate and parse the JSON content."""
        try:
            # Clean the content if it contains markdown code blocks
            cleaned_text = content.strip()
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]
            elif cleaned_text.startswith('```'):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]

            annotations = json.loads(cleaned_text.strip())
            
            # Handle both list and dict with 'reviews' key formats
            if isinstance(annotations, dict) and 'reviews' in annotations:
                annotations = annotations['reviews']
            elif not isinstance(annotations, list):
                raise ValueError("Response is not a list or dict with 'reviews' key")

            # Validate each annotation object
            for annotation in annotations:
                for emotion in self.expected_emotions:
                    if emotion not in annotation or not isinstance(annotation[emotion], int):
                        annotation[emotion] = 0

            return annotations

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to parse JSON: {e}")
            return [{emotion: 0 for emotion in self.expected_emotions}]

    def annotate(self, input_file: str, output_folder: str, batch_size: int = 5, 
                n: int = None, model: str = None, **kwargs) -> None:
        """Main annotation process."""
        start_time = datetime.now()
        print(f"Starting annotation process at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Prepare input/output
        df = pd.read_excel(input_file)
        if n is not None:
            df = df.head(n)

        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f'{model}-annotations.xlsx')
        metrics_file = os.path.join(output_folder, f'{model}-metrics.json')

        results_df = pd.DataFrame()
        metrics = {
            'start_time': start_time.isoformat(),
            'batches': [],
            'total_tokens': 0,
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0
        }

        # Process batches
        for i in range(0, len(df), batch_size):
            batch_start_time = time.time()
            batch = df.iloc[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")

            batch_data = [{"review": row['review'], "sentence": row['sentence']} 
                         for _, row in batch.iterrows()]
            
            annotations_list, usage_metadata = self.get_annotation(batch_data, **kwargs)

            # Update metrics
            batch_metrics = {
                'batch_number': i//batch_size + 1,
                'batch_size': len(batch),
                'processing_time': time.time() - batch_start_time,
                'tokens': usage_metadata
            }
            metrics['batches'].append(batch_metrics)
            metrics['total_tokens'] += usage_metadata.get('total_tokens', 0)
            metrics['total_prompt_tokens'] += usage_metadata.get('prompt_tokens', 0)
            metrics['total_completion_tokens'] += usage_metadata.get('completion_tokens', 0)

            # Update results
            for j, (_, row) in enumerate(batch.iterrows()):
                if j < len(annotations_list):
                    row_data = row.to_dict()
                    row_data.update(annotations_list[j])
                    results_df = pd.concat([results_df, pd.DataFrame([row_data])], 
                                         ignore_index=True)

            # Save progress
            results_df.to_excel(output_file, index=False, engine='openpyxl')

            batch_time = time.time() - batch_start_time
            print(f"Batch {i//batch_size + 1} processed in {batch_time:.2f}s "
                  f"(Tokens: {usage_metadata.get('total_tokens', 0)})")
            
            time.sleep(kwargs.get('sleep_time', 1))  # Configurable rate limiting

        # Record final metrics
        end_time = datetime.now()
        metrics['end_time'] = end_time.isoformat()
        metrics['total_duration'] = (end_time - start_time).total_seconds()

        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Annotation process completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')} "
              f"(Duration: {metrics['total_duration']:.2f}s)")
        print(f"Total tokens used: {metrics['total_tokens']}") 