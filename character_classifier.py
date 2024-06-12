import openai

class CharacterClassifier:
    def __init__(self, api_key, character_types):
        self.api_key = api_key
        self.character_types = character_types
        openai.api_key = self.api_key

    def classify_character(self, character, script):
        prompt = f"Classify the character '{character}' into one of these types: {', '.join(self.character_types)}. Based on the following script, determine the character type of '{character}':\n\n{script}"
        response = openai.ChatCompletion.create(
            model="gpt-4-0",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()

    def classify_all_characters(self, characters, script):
        classifications = {}
        for character in characters:
            character_type = self.classify_character(character, script)
            classifications[character] = character_type
        return classifications

    def save_classifications_to_csv(self, classifications, output_file):
        import pandas as pd
        data = [(character, character_type) for character, character_type in classifications.items()]
        df = pd.DataFrame(data, columns=['Character', 'Character Type'])
        df.to_csv(output_file, index=False, encoding='utf-8')
