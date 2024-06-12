import re
from tqdm.contrib.concurrent import process_map

class DataCleaner:
    def __init__(self, input_file):
        self.input_file = input_file

    @staticmethod
    def clean_script_line(line):
        # Remove character descriptions and annotations
        cleaned_line = re.sub(r'\{.*?\}', '', line)  # Remove {...}
        cleaned_line = re.sub(r'\[.*?\]', '', cleaned_line)  # Remove [...]
        cleaned_line = re.sub(r'\(.*?\)', '', cleaned_line)  # Remove (...)
        cleaned_line = re.sub(r'\n', ' ', cleaned_line) # Remove \n 
        cleaned_line = re.sub(r'\s+', ' ', cleaned_line) # Remove extra whitespace
        
        # Remove extra whitespace
        cleaned_line = cleaned_line.strip()
        
        return cleaned_line

    @staticmethod
    def extract_author_and_text(paragraph):
        match = re.match(r'^(\w+):\s*(.*)', paragraph, re.DOTALL)
        if match:
            author = match.group(1)
            text = match.group(2).strip()
        else:
            author = 'None'
            text = paragraph.strip()
        
        return author, text

    @staticmethod
    def split_into_sentences(text):
        # Split text into sentences
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        return sentences

    def clean_movie_script(self):
        with open(self.input_file, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Split content based on double newlines (paragraphs)
        paragraphs = content.split('\n\n')
        
        # Use process_map to clean each paragraph concurrently
        cleaned_paragraphs = process_map(self.clean_script_line, paragraphs, max_workers=4)
        
        # Extract author and text for each cleaned paragraph
        author_text_pairs = [self.extract_author_and_text(para) for para in cleaned_paragraphs if para]
        
        return author_text_pairs
