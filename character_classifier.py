from openai import OpenAI
from transformers import pipeline
import torch
from tqdm import tqdm

class CharacterClassifier:
    def __init__(self, character_types, model_name=None):
        self.character_types = character_types
        self.prompt = "Classify the character '%s' into one of these types: %s. Only return the theme and nothing else. Based on the following script, determine the character type of '%s':\n\n%%s\n\nCharacter Type:" % ('%s', ', '.join(self.character_types), '%s')
        
        if self.model_name:
            self.classifier = pipeline("text-generation", model=self.model_name,  model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
        else:
            self.classifier = OpenAI()


    def classify_sentence_openai(self, sentence):
        response = self.classifier.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": self.prompt % sentence}
            ],
            max_tokens=10
        )
        answer = response.choices[0].message.content.strip()
        for theme in self.themes:
            if theme.lower() in answer.lower():
                return theme
    
    def classify_sentence_transformers(self, sentence):
        result = self.classifier(self.prompt % sentence, max_new_tokens=10, num_return_sequences=1)[0]['generated_text']
        for theme in self.themes:
            if theme.lower() in result.lower():
                return theme

    def classify_all_characters(self, characters, script):
        classifications = {}
        for character in tqdm(characters):
            character_type = self.classify_character(character, script)
            classifications[character] = character_type
        return classifications

    def save_classifications_to_csv(self, classifications, output_file):
        import pandas as pd
        data = [(character, character_type) for character, character_type in classifications.items()]
        df = pd.DataFrame(data, columns=['Character', 'Character Type'])
        df.to_csv(output_file, index=False, encoding='utf-8')
