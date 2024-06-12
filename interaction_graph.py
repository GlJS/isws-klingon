import networkx as nx
import matplotlib.pyplot as plt

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

    def draw_graph(self, output_file='interaction_graph.png'):
        pos = nx.spring_layout(self.graph)
        plt.figure(figsize=(12, 12))
        
        edges = self.graph.edges(data=True)
        weights = [edge[2]['weight'] for edge in edges]

        nx.draw(self.graph, pos, edges=edges, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, width=weights)
        plt.savefig(output_file)
        plt.show()

    def save_graph(self, output_file='interaction_graph.gml'):
        nx.write_gml(self.graph, output_file)
