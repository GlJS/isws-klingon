import os
import argparse
from data_cleaner import DataCleaner
from topic_modeler import TopicModeler
from interaction_graph import InteractionGraph
from theme_classifier import SentenceClassifier
from character_classifier import CharacterClassifier
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

def main(input_file, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Define output file paths
    output_file = os.path.join(output_folder, 'Script_Topics.csv')
    graph_output_file = os.path.join(output_folder, 'interaction_graph.png')
    graph_gml_file = os.path.join(output_folder, 'interaction_graph.gml')
    classification_output_file = os.path.join(output_folder, 'Sentence_Classifications.csv')
    character_classification_output_file = os.path.join(output_folder, 'Character_Classifications.csv')

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
    data_cleaner = DataCleaner(input_file)
    author_text_pairs = data_cleaner.clean_movie_script()

    # Step 2: Split text into sentences
    sentences = [text for _, text in author_text_pairs]
    all_sentences = [sentence for text in sentences for sentence in DataCleaner.split_into_sentences(text)]

    # Step 3: Perform topic modeling on the sentences
    topic_modeler = TopicModeler(all_sentences)
    topic_modeler.perform_topic_modeling()

    # Step 4: Save the results to CSV
    topic_modeler.save_to_csv(author_text_pairs, output_file)

    # Step 5: Create and draw the interaction graph
    interaction_graph = InteractionGraph(author_text_pairs)
    interaction_graph.build_graph()
    interaction_graph.draw_graph(graph_output_file)
    interaction_graph.save_graph(graph_gml_file)

    # Step 6: Classify sentences based on themes using OpenAI API
    sentence_classifier = SentenceClassifier(api_key, themes)
    classifications = sentence_classifier.classify_sentences(all_sentences)

    # Step 7: Save classifications to CSV
    sentence_classifier.save_classifications_to_csv(classifications, classification_output_file)

    # Step 8: Classify characters based on their types using OpenAI API
    characters = set(author for author, _ in author_text_pairs if author != 'None')
    script_text = '\n'.join([f'{author}: {text}' for author, text in author_text_pairs])
    character_classifier = CharacterClassifier(api_key, character_types)
    character_classifications = character_classifier.classify_all_characters(characters, script_text)

    # Step 9: Save character classifications to CSV
    character_classifier.save_classifications_to_csv(character_classifications, character_classification_output_file)

    # Output a summary of the topics
    print(topic_modeler.get_topic_info())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a movie script and perform various analyses.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input movie script text file.')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save the output files.')

    args = parser.parse_args()
    main(args.input_file, args.output_folder)
