import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
sns.set_theme()

class TopicModeler:
    def __init__(self, sentences):
        self.sentences = sentences
        self.topic_model = None
        self.topics = None

    def perform_topic_modeling(self):
        vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
        self.topic_model = BERTopic(vectorizer_model=vectorizer_model, nr_topics=20)
        self.topics, _ = self.topic_model.fit_transform(self.sentences)
    
    def save_to_csv(self, author_text_pairs, output_file):
        # Prepare data for CSV
        data = []
        for (author, text), topic in zip(author_text_pairs, self.topics):
            data.append([author, text, topic])
        
        # Save to CSV
        df = pd.DataFrame(data, columns=['Author', 'Text', 'Topic'])
        df.to_csv(output_file, index=False, encoding='utf-8')

    def plot_topic_scatter(self, output_file='topic_scatter.png', top_n_words=5):
        # Get embeddings using SBERT
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        embeddings = model.encode(self.sentences, show_progress_bar=True)
        
        # Reduce dimensionality using PCA
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        # Get the top n words for each topic
        topic_words = {topic: ', '.join([word for word, _ in self.topic_model.get_topic(topic)[:top_n_words]]) 
                       for topic in set(self.topics) if topic != -1}
        
        # Use seaborn for the scatter plot
        plt.figure(figsize=(14, 10))
        palette = sns.color_palette("tab20", len(topic_words))
        sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=self.topics, palette=palette, legend=None, alpha=0.7)
        
        # Create a legend with topic names and top words
        topic_colors = {topic: palette[i] for i, topic in enumerate(topic_words.keys())}
        legend_labels = [f"Topic {topic}: {words}" for topic, words in topic_words.items()]
        handles = [mlines.Line2D([], [], color=topic_colors[topic], marker='o', linestyle='None', markersize=10, label=label) 
                   for topic, label in zip(topic_words.keys(), legend_labels)]

        # Calculate the bbox_to_anchor values based on the legend width
        max_label_length = max(len(label) for label in legend_labels)
        legend_width = max_label_length * 0.015  # Adjust factor as needed
        plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., frameon=False)
        
        plt.title("Topic Scatter Plot with PCA")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.subplots_adjust(right=0.75)  # Adjust right to make space for legend
        plt.savefig(output_file, bbox_inches='tight')
        plt.show()
