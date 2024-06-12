# Movie Script Analysis
This GitHub repository represents the code for the Klingon group at the International Semantic Web Research Summer School 2024.
This project performs various analyses on a movie script, including data cleaning, topic modeling, character interaction graph creation, sentence theme classification, and character type classification.

## Setup

### Prerequisites

- Python 3.7 or higher
- `pip` for package management

### Install Dependencies

1. Clone this repository:
    ```bash
    git clone https://github.com/your-repo/movie-script-analysis.git
    cd movie-script-analysis
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file in the root directory and add your OpenAI API key:
    ```plaintext
    OPENAI_API_KEY=your_openai_api_key_here
    ```

### Run the Analysis

1. Place your movie script text file in a desired location.

2. Run the analysis with the following command:
    ```bash
    python main.py --input_file path/to/your/script.txt --output_folder path/to/output_folder
    ```

### File Descriptions

- **data_cleaner.py**: Contains the `DataCleaner` class for cleaning and processing the script.
- **topic_modeler.py**: Contains the `TopicModeler` class for performing topic modeling on the sentences.
- **interaction_graph.py**: Contains the `InteractionGraph` class for creating and visualizing a network graph of character interactions.
- **sentence_classifier.py**: Contains the `SentenceClassifier` class for classifying sentences based on themes using OpenAI API.
- **character_classifier.py**: Contains the `CharacterClassifier` class for classifying characters based on their types using OpenAI API.
- **main.py**: Orchestrates the entire workflow.

### Dependencies

Ensure you have the following packages installed:

- openai
- python-dotenv
- tqdm
- pandas
- networkx
- matplotlib
- bertopic

These can be installed via the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
