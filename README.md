# SentenceGraph
A Data Structure suitable for a graph above text spans, including logic for visualization, comparisons, etc.

## Visualizers
There are two optional visualizers for SentenceGraph:
1. `displacy`: requires installation of the SpaCy package - run:

 ```pip install spacy```

 Generates an SVG file.

2. `brat`: requires installation of brat-visualizer, which is a repository for using Brat
to visualize a networkx DiGraph object. To install it, run:

 ```sh install_brat_visualizer.sh```

 Generates an HTML file.


 You can specify the desired visualizer as a `visualizer=` argument to the `visualize` method.

