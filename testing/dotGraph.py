class DotGraph():
    def __init__(self, name="Unnamed Graph"):
        self.name = name
        self.nodes = {}
        self.edges = []

    def addNode(self, id: str = "", attributes: dict = None):
        """
        Note: Don't pass an id to have the graph assign one automatically. The assigned id will be returned.
        Avoid manually naming nodes 'NodeX' (where X is an integer) to avoid naming conflicts.
        """
        if not id:
            id = 'Node' + str(len(self.nodes))
        nodeString = f'"{id}"'
        if attributes is not None:
            nodeString += " ["
            for key, value in attributes.items():
                nodeString += f'{key}="{value}", '
            nodeString += "];"
        self.nodes[id] = nodeString

        return id

    def addEdge(self, source: str, dest: str, attributes: dict = None):
        if source not in self.nodes:
            print(f"ERROR: Unknown source node for graph: '{source}'.")
            source = "Unknown Node"
        if dest not in self.nodes:
            print(f"ERROR: Unknown destination node for graph: '{dest}'.")
            dest = "Unknown Node"

        edgeString = f'"{source}" -> "{dest}"'
        if attributes is not None:
            edgeString += " ["
            for key, value in attributes.items():
                edgeString += f'{key}="{value}", '
            edgeString += "];"

        self.edges.append(edgeString)

    def fullDotString(self, attributes: dict = None):
        graphString = 'digraph '+self.name+' {\n\t'
        if attributes is not None:
            for key, value in attributes.items():
                graphString += f'{key}={value};\n\t'
        # Nodes:
        graphString += "\n\t".join(self.nodes.values())
        graphString += "\n\t"

        # Edges:
        graphString += "\n\t".join(self.edges)
        graphString += "\n"

        graphString += "}"
        return graphString

    def dotURL(self, attributes: dict = None):
        import plantuml  # This is non-standard, which is why I import it only if actually needed.
        PlantUML = plantuml.PlantUML(url='http://www.plantuml.com/plantuml/svg/')
        return PlantUML.get_url(self.fullDotString(attributes))