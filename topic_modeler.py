import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

class TopicModeler:
    def __init__(self, sentences):
        self.sentences = sentences
        self.topic_model = None
        self.topics = None

    def perform_topic_modeling(self):
        vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
        self.topic_model = BERTopic(vectorizer_model=vectorizer_model)
        self.topics, _ = self.topic_model.fit_transform(self.sentences)
    
    def save_to_csv(self, author_text_pairs, output_file):
        # Prepare data for CSV
        data = []
        for (author, text), topic in zip(author_text_pairs, self.topics):
            data.append([author, text, topic])
        
        # Save to CSV
        df = pd.DataFrame(data, columns=['Author', 'Text', 'Topic'])
        df.to_csv(output_file, index=False, encoding='utf-8')

    def get_topic_info(self):
        return self.topic_model.get_topic_info()
