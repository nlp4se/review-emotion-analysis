from abc import ABC, abstractmethod
from typing import Any
from dotenv import load_dotenv

class AssistantCreator(ABC):
    """Abstract base class for creating AI assistants."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = self._initialize_client()
    
    @abstractmethod
    def _initialize_client(self) -> Any:
        """Initialize the specific AI provider's client."""
        pass
    
    def load_guidelines(self, guidelines_path: str) -> str:
        """Load annotation guidelines from a text file."""
        with open(guidelines_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def save_assistant_id(self, assistant_id: str, model: str):
        """Save the assistant ID to a file for reuse."""
        filename = f"data/assistants/assistant_id_{model}.txt"
        with open(filename, 'w') as file:
            file.write(assistant_id)
    
    def get_base_instructions(self, guidelines: str) -> str:
        """Get the base instructions template."""
        return f"""
        You are an assistant that annotates emotions in sentences from app reviews. You have been provided detailed guidelines below that you MUST follow at all times.

        **Annotation Guidelines:**
        {guidelines}

        **Annotation Input Format:**
        The input is a JSON list of objects following this schema:
        [
          {{
              "review": "The full text of the review",
              "sentence": "The sentence to annotate"
          }}
        ]

        **Annotation Output Format:**
        Return a JSON list of objects where each item corresponds to the input sentence in the same order, following this schema:
        [
          {{
              "Joy": 0 or 1,
              "Trust": 0 or 1,
              "Fear": 0 or 1,
              "Surprise": 0 or 1,
              "Sadness": 0 or 1,
              "Disgust": 0 or 1,
              "Anger": 0 or 1,
              "Anticipation": 0 or 1,
              "Neutral": 0 or 1,
              "Reject": 0 or 1
          }}
        ]

        **Important Rules:**
        - Each output object must directly correspond to an input sentence.
        - Ensure JSON validity and maintain correct key-value formatting.
        - Do NOT include explanations, additional text, or formatting.
        """
    
    @abstractmethod
    def create_assistant(self, guidelines_file: str, model: str) -> Any:
        """Create an AI assistant with the specified guidelines and model."""
        pass 