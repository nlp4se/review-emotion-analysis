import os
import argparse
import mistralai
from typing import Any
from dotenv import load_dotenv
from create_assistant import AssistantCreator

class MistralAssistantCreator(AssistantCreator):
    def _initialize_client(self) -> mistralai.MistralClient:
        return mistralai.MistralClient(api_key=self.api_key)
    
    def create_assistant(self, guidelines_file: str, model: str = "mistral-large-latest") -> Any:
        guidelines = self.load_guidelines(guidelines_file)
        instructions = self.get_base_instructions(guidelines)
        
        print("Creating a new Mistral assistant...")
        
        assistant = self.client.create_assistant(
            name="Emotion Annotation Assistant",
            instructions=instructions,
            model=model,
        )
        
        self.save_assistant_id(assistant['id'], model)
        print(f"Assistant created successfully. Assistant ID: {assistant['id']}")
        return assistant

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description='Create a Mistral AI assistant for emotion annotation.')
    parser.add_argument('--guidelines', required=True, help='Path to the guidelines file')
    parser.add_argument('--model', default="mistral-large-latest", help='Mistral AI model to use')
    
    args = parser.parse_args()
    
    creator = MistralAssistantCreator(api_key=os.getenv("MISTRAL_API_KEY"))
    creator.create_assistant(args.guidelines, args.model)

if __name__ == "__main__":
    main() 