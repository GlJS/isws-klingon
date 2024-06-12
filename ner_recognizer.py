from transformers import pipeline

class NERRecognizer:
    def __init__(self, model_name='dbmdz/bert-large-cased-finetuned-conll03-english'):
        self.model_name = model_name
        self.ner_pipeline = pipeline('ner', model=self.model_name, aggregation_strategy="simple", device_map="auto")

    def recognize_entities(self, sentences):
        ner_results = []
        for sentence in sentences:
            entities = self.ner_pipeline(sentence)
            ner_results.append((sentence, entities))
        return ner_results

    def save_entities_to_csv(self, ner_results, output_file):
        import pandas as pd
        data = []
        for sentence, entities in ner_results:
            for entity in entities:
                data.append({
                    'Sentence': sentence,
                    'Entity': entity['word'],
                    'Entity Type': entity['entity_group'],
                    'Start': entity['start'],
                    'End': entity['end']
                })
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False, encoding='utf-8')
