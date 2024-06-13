import re
from tqdm.contrib.concurrent import process_map

class DataCleaner:
    def __init__(self, input_file):
        self.input_file = input_file

    @staticmethod
    def clean_script_line(line):
        # Define the patterns for different parts
        patterns = {
            'M': r'^[A-Z\s\'&]+$',
            'S': r'^(EXT\.|INT\.)',
            'N': r'^[A-Z\s\.,\?!:]+$',
            'C': r'^[A-Z]+\s*$',
            'D': r'^[a-zA-Z,\';!\?\s\.\-]+$',
            'E': r'^\([a-z\s]+\)$'
        }

        # Define a function to identify the type of each line
        def identify_line_type(line):
            for key, pattern in patterns.items():
                if re.match(pattern, line):
                    return key
            return None

        # Split the text into lines and identify their types
        line_type = identify_line_type(line)
        if line_type:
            return f'{line_type}: {line}'
        else:
            return line

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
        
        # Split content based on newlines
        lines = content.split('\n')
        
        # Use process_map to clean each line concurrently
        cleaned_lines = process_map(self.clean_script_line, lines, max_workers=4)
        
        # Combine the cleaned lines into a single string
        cleaned_script = '\n'.join(cleaned_lines)
        
        return cleaned_script

# Usage example:
input_file = '/mnt/data/to_parse.txt'
cleaner = DataCleaner(input_file)
parsed_script = cleaner.clean_movie_script()

with open('/mnt/data/parsed_output.txt', 'w', encoding='utf-8') as file:
    file.write(parsed_script)
