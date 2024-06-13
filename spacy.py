import spacy
import networkx as nx
from tqdm.contrib.concurrent import process_map

# Load SpaCy model
nlp = spacy.load("en_core_web_trf")

# Define the main function to process text
def process_text(text):
    doc = nlp(text)

    # Word Segmentation and POS Tagging
    tokens = [(token.text, token.pos_) for token in doc]

    # Dependency Parsing
    dependencies = [(token.text, token.dep_, token.head.text) for token in doc]

    # Event Extraction and Relation Extraction (simplified for illustration)
    events = []
    relations = []
    for ent in doc.ents:
        events.append((ent.text, ent.label_))
        for token in doc:
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                relations.append((token.text, token.head.text))
    
    # Coreference Resolution
    coreferences = []
    if doc._.has_coref:
        for cluster in doc._.coref_clusters:
            coreferences.append(cluster.main.text)

    return tokens, dependencies, events, relations, coreferences

# Build the knowledge graph
def build_knowledge_graph(processed_data):
    G = nx.DiGraph()

    for tokens, dependencies, events, relations, coreferences in processed_data:
        for token, pos in tokens:
            G.add_node(token, pos=pos)
        
        for word, dep, head in dependencies:
            G.add_edge(word, head, dependency=dep)
        
        for event, label in events:
            G.add_node(event, label=label)
        
        for subj, verb in relations:
            G.add_edge(subj, verb, relation="subject_of")
        
        for coref in coreferences:
            G.add_node(coref, coreference=True)
    
    return G

# Example usage
text = "President Obama gave a speech at the White House. He discussed the economy."
processed_data = [process_text(text)]
G = build_knowledge_graph(processed_data)

# Display the knowledge graph
print(nx.info(G))
