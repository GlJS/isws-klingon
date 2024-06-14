import networkx as nx
import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map
import json

class InteractionGraph:
    def __init__(self, author_text_pairs):
        self.author_text_pairs = author_text_pairs
        self.graph = nx.Graph()

    def build_graph(self, window_size=10):
        for i, (author, _) in enumerate(self.author_text_pairs):
            if author == 'None':
                continue
            if author not in self.graph:
                self.graph.add_node(author)
            
            # Check the past and future window for interactions
            for j in range(max(0, i - window_size), min(len(self.author_text_pairs), i + window_size + 1)):
                if j != i:
                    co_author = self.author_text_pairs[j][0]
                    if co_author != 'None' and co_author != author:
                        if self.graph.has_edge(author, co_author):
                            self.graph[author][co_author]['weight'] += 1
                        else:
                            self.graph.add_edge(author, co_author, weight=1)
        
        self.filter_nodes_by_interaction(threshold=30)

    def filter_nodes_by_interaction(self, threshold):
        nodes_to_remove = [node for node in self.graph.nodes if self.graph.degree(node, weight='weight') <= threshold]
        self.graph.remove_nodes_from(nodes_to_remove)

    def draw_graph(self, output_file='interaction_graph.png', scale_down=50):
        pos = nx.spring_layout(self.graph, k=10, iterations=50)
        plt.figure(figsize=(12, 12))
        
        # Calculate node sizes based on the sum of the weights of incoming and outgoing edges
        node_sizes = [self.graph.degree(node, weight='weight')  for node in self.graph.nodes]

        edges = self.graph.edges(data=True)
        weights = [edge[2]['weight'] for edge in edges]
        
        # Scale down the weights for visualization
        scaled_weights = [weight / scale_down for weight in weights]  # Adjust the multiplier as needed

        nx.draw(self.graph, pos, with_labels=True, node_size=node_sizes, node_color='skyblue', font_size=10, width=scaled_weights)
        plt.savefig(output_file)
        plt.show()

    def save_graph(self, output_file='interaction_graph.gml'):
        nx.write_gml(self.graph, output_file)
    
    def dump_to_json(self, output_file='interaction_graph.json', top_n=50):
        edges = sorted(self.graph.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:top_n]
        triples = [(u, v, d['weight']) for u, v, d in edges]
        
        with open(output_file, 'w') as f:
            json.dump(triples, f, indent=2)

    def get_top_10_nodes(self, output_file='top_10_nodes.json'):
        degree_dict = dict(self.graph.degree(weight='weight'))
        top_10_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        
        with open(output_file, 'w') as f:
            json.dump(top_10_nodes, f, indent=2)
