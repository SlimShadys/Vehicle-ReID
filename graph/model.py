class GraphEngine:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.node_map = {}
        self.edge_map = {}

    def add_node(self, node):
        self.nodes.append(node)
        self.node_map[node.id] = node

    def add_edge(self, edge):
        self.edges.append(edge)
        self.edge_map[edge.id] = edge

    def get_node(self, node_id):
        return self.node_map.get(node_id)

    def get_edge(self, edge_id):
        return self.edge_map.get(edge_id)

    def __str__(self):
        return f"Graph(nodes={self.nodes}, edges={self.edges})"

    def __repr__(self):
        return str(self)