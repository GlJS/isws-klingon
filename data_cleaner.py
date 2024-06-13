import re
from tqdm.contrib.concurrent import process_map

class DataCleaner:
    def __init__(self, data):
        self.data = data
        self.patterns = {
            'M': r'(^[A-Z\s\'&]+$)|SCENE ',
            'S': r'^(EXT\.|INT\.)',
            'N': r'^([A-Z ]+)(?=\[|$)',
            'C': r'^[A-Z]+\s*$',
            'T': r'^\[.*?\]$',
            'E': r'^\([a-z\s]+\)$',
            'D': r'^([a-zA-Z ]+)$',
        }


    def clean_script_line(self, line):
        # Define the patterns for different parts, including transitions

        # Define a function to identify the type of each line
        def identify_line_type(line):
            for key, pattern in self.patterns.items():
                if re.match(pattern, line):
                    return key
            return None

        # Identify the type of the line
        line_type = identify_line_type(line)
        if line_type:
            return f'{line_type}: {line}'
        else:
            return line
        
    
    def clean_movie_script(self):
        
        # Split content based on newlines
        lines = self.data.split('\n')
        
        # Use process_map to clean each line concurrently
        cleaned_lines = map(self.clean_script_line, lines)
        
        cleaned_lines = [line for line in cleaned_lines if len(line) > 0]
        
        new_cleaned_lines = []
        for line in cleaned_lines:
            for p in self.patterns.keys():
                if line.startswith(f'{p}:'):
                    new_cleaned_lines.append(line)
        
        # Combine the cleaned lines into a single string
        cleaned_script = '\n'.join(new_cleaned_lines)
        
        
        return cleaned_script


    @staticmethod
    def extract_author_and_text(paragraph):
        match = re.match(r'C: ([A-Z\s]+)', paragraph)
        if match:
            return match.group(1)
        return None
    
    def extract_author_text_pairs(self, content):
        lines = content.split('\n')
        author = None
        pairs = []
        text_lines = []

        for line in lines:
            if line.startswith('C: '):
                if author:
                    pairs.append((author, ' '.join(text_lines)))
                author = self.extract_author_and_text(line)
                text_lines = []
            elif line.startswith('D: '):
                if author:
                    text_lines.append(line[3:].strip())
            else:
                if author:
                    pairs.append((author, ' '.join(text_lines)))
                    author = None
                    text_lines = []

        if author:
            pairs.append((author, ' '.join(text_lines)))

        return pairs