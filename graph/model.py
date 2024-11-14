import networkx as nx

class GraphEngine(nx.Graph):
    def __init__(self):
        super().__init__()
        self.nodes = []
        self.edges = []
        self.node_map = {}
        self.edge_map = {}

    def get_connected_components(self):
        return list(nx.connected_components(self))
    
    def __str__(self):
        return f"Graph(nodes={list(self.nodes)}, edges={list(self.edges)})"

    def __repr__(self):
        return str(self)