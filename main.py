import os
import argparse
import numpy as np
import random
import torch
from data_cleaner import DataCleaner
from topic_modeler import TopicModeler
from interaction_graph import InteractionGraph
from sentence_classifier import SentenceClassifier
from character_classifier import CharacterClassifier
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Function to set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(input_file, output_folder, seed, algorithms, use_openai):
    set_seed(seed)
    os.makedirs(output_folder, exist_ok=True)

    # Define output file paths
    output_file = os.path.join(output_folder, 'Script_Topics.csv')
    graph_output_file = os.path.join(output_folder, 'interaction_graph.png')
    graph_gml_file = os.path.join(output_folder, 'interaction_graph.gml')
    classification_output_file = os.path.join(output_folder, 'Sentence_Classifications.csv')
    character_classification_output_file = os.path.join(output_folder, 'Character_Classifications.csv')
    scatter_plot_file = os.path.join(output_folder, 'topic_scatter.png')

    # List of themes
    themes = [
        "Human nature", "Free will", "Self-awareness", "Coming of age", "Hope", "Jealousy",
        "Justice", "Fear", "Freedom", "Friendship", "Bravery", "Happiness", "Passion",
        "Kindness", "Trust", "War"
    ]

    # List of character types
    character_types = [
        "Protagonist", "Antagonist", "Love interest", "Confidant",
        "Deuteragonists", "Tertiary characters", "Foil"
    ]

    # Step 1: Clean the movie script and get author-text pairs
    if 'data_cleaner' in algorithms or 'all' in algorithms:
        data_cleaner = DataCleaner(input_file)
        author_text_pairs = data_cleaner.clean_movie_script()
    else:
        author_text_pairs = []

    # Step 2: Split text into sentences
    if 'topic_modeler' in algorithms or 'sentence_classifier' in algorithms or 'character_classifier' in algorithms or 'all' in algorithms:
        sentences = [text for _, text in author_text_pairs]
        all_sentences = [sentence for text in sentences for sentence in DataCleaner.split_into_sentences(text)]
    else:
        all_sentences = []

    # Step 3: Perform topic modeling on the sentences
    if 'topic_modeler' in algorithms or 'all' in algorithms:
        topic_modeler = TopicModeler(all_sentences)
        topic_modeler.perform_topic_modeling()
        topic_modeler.save_to_csv(author_text_pairs, output_file)
        topic_modeler.plot_topic_scatter(scatter_plot_file)

    # Step 4: Create and draw the interaction graph
    if 'interaction_graph' in algorithms or 'all' in algorithms:
        interaction_graph = InteractionGraph(author_text_pairs)
        interaction_graph.build_graph()
        interaction_graph.draw_graph(graph_output_file)
        interaction_graph.save_graph(graph_gml_file)

    # Step 5: Classify sentences based on themes
    if 'sentence_classifier' in algorithms or 'all' in algorithms:
        if use_openai:
            sentence_classifier = SentenceClassifier(themes=themes)
        else:
            sentence_classifier = SentenceClassifier(model_name="meta-llama/Meta-Llama-3-8B-Instruct", themes=themes)
        classifications = sentence_classifier.classify_sentences(all_sentences)
        sentence_classifier.save_classifications_to_csv(classifications, classification_output_file)

    # Step 6: Classify characters based on their types
    if 'character_classifier' in algorithms or 'all' in algorithms:
        characters = set(author for author, _ in author_text_pairs if author != 'None')
        script_text = '\n'.join([f'{author}: {text}' for author, text in author_text_pairs])
        if use_openai:
            character_classifier = CharacterClassifier(character_types=character_types)
        else:
            character_classifier = CharacterClassifier(model_name="meta-llama/Meta-Llama-3-8B-Instruct", character_types=character_types)
        character_classifications = character_classifier.classify_all_characters(characters, script_text)
        character_classifier.save_classifications_to_csv(character_classifications, character_classification_output_file)

    # Output a summary of the topics
    if 'topic_modeler' in algorithms or 'all' in algorithms:
        print(topic_modeler.get_topic_info())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a movie script and perform various analyses.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input movie script text file.')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save the output files.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility.')
    parser.add_argument('--algorithms', nargs='+', default=['all'], choices=['all', 'data_cleaner', 'topic_modeler', 'interaction_graph', 'sentence_classifier', 'character_classifier'], help='Algorithms to run.')
    parser.add_argument('--use_openai',  action=argparse.BooleanOptionalAction, default=False, help='Use OpenAI API for classification.')

    args = parser.parse_args()
    main(args.input_file, args.output_folder, args.seed, args.algorithms, args.use_openai)