import openai

class SentenceClassifier:
    def __init__(self, api_key, themes):
        self.api_key = api_key
        self.themes = themes
        openai.api_key = self.api_key

    def classify_sentence(self, sentence):
        response = openai.ChatCompletion.create(
            model="gpt-4-0",
            messages=[
                {"role": "system", "content": "Classify the following sentence into one of the themes: " + ", ".join(self.themes)},
                {"role": "user", "content": sentence}
            ]
        )
        return response['choices'][0]['message']['content'].strip()

    def classify_sentences(self, sentences):
        classifications = []
        for sentence in sentences:
            theme = self.classify_sentence(sentence)
            classifications.append((sentence, theme))
        return classifications

    def save_classifications_to_csv(self, classifications, output_file):
        import pandas as pd
        df = pd.DataFrame(classifications, columns=['Sentence', 'Theme'])
        df.to_csv(output_file, index=False, encoding='utf-8')
