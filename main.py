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
from ner_recognizer import NERRecognizer
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
    os.makedirs(os.path.join(output_folder, 'graphs'), exist_ok=True)

    # Define output file paths
    input_file_basename = os.path.basename(input_file)
    graph_output_file = os.path.join(output_folder, "graphs", f'{input_file_basename}.png')
    # graph_gml_file = os.path.join(output_folder, 'interaction_graph.gml')
    # classification_output_file = os.path.join(output_folder, 'Sentence_Classifications.csv')
    # character_classification_output_file = os.path.join(output_folder, 'Character_Classifications.csv')
    # scatter_plot_file = os.path.join(output_folder, 'topic_scatter.png')

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
    with open(input_file, 'r', encoding='utf-8') as file:
        texts = file.read()
    data_cleaner = DataCleaner(texts)
    if 'data_cleaner' in algorithms:
        texts = data_cleaner.clean_movie_script()
        with open(os.path.join("data/parsed", f'{input_file}_parsed.txt'), 'w', encoding='utf-8') as file:
            file.write(texts)
    author_text_pairs = data_cleaner.extract_author_text_pairs(content=texts)

    # Step 3: Create and draw the interaction graph
    interaction_graph = InteractionGraph(author_text_pairs)
    interaction_graph.build_graph()
    interaction_graph.draw_graph(graph_output_file)
    # interaction_graph.save_graph(graph_gml_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a movie script and perform various analyses.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input movie script text file.')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save the output files.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility.')
    parser.add_argument('--algorithms', nargs='+', default=['none'], choices=['all', 'data_cleaner', 'topic_modeler', 'interaction_graph', 'sentence_classifier', 'character_classifier', 'ner_recognizer', 'none'], help='Algorithms to run.')
    parser.add_argument('--use_openai',  action=argparse.BooleanOptionalAction, default=False, help='Use OpenAI API for classification.')

    args = parser.parse_args()
    main(args.input_file, args.output_folder, args.seed, args.algorithms, args.use_openai)