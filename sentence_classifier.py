from openai import OpenAI
from tqdm import tqdm
from transformers import pipeline
import torch

class SentenceClassifier:
    def __init__(self,themes, model_name=None):
        self.themes = themes
        self.model_name = model_name
        
        self.prompt = "Classify the following sentence into one of the themes: %s.\n\n. Only return the theme and nothing else. Sentence: %%s\n\nTheme:" % ', '.join(self.themes)
            
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
        prompt = self.prompt % sentence
        result = self.classifier(prompt, max_new_tokens=10, num_return_sequences=1)[0]['generated_text']
        result = result[len(prompt):]
        for theme in self.themes:
            if theme.lower() in result.lower():
                return theme

    def classify_sentences(self, sentences):
        classifications = []
        for sentence in tqdm(sentences):
            if self.model_name:
                theme = self.classify_sentence_transformers(sentence)
            else:
                theme = self.classify_sentence_openai(sentence)
            classifications.append((sentence, theme))
        return classifications

    def save_classifications_to_csv(self, classifications, output_file):
        import pandas as pd
        df = pd.DataFrame(classifications, columns=['Sentence', 'Theme'])
        df.to_csv(output_file, index=False, encoding='utf-8')
