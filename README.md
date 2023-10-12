# Frameworks to Estimate Motif Counts in Directed Networks
This is our code base, fully developed by us in Python to evaluate the performance of our three frameworks (namely RW-IS, RW-RJ and RW-BB). The files provided in this  repository can be leveraged to create graph objects that are compatible with the three techniques. We have predefined exact counting functions for each class that can count _wedges, cyclic-C3, cyclic-C4, source-S3, leaf-C3, butterfly, clique-K4_ graphlets. Below is the pictorial representation of each graphlet. These exact counts form the ground truths for our error comparisons across the techniques introduced. 

## Creating a Graph:
The dataset should be in the form of text file with node labels ranging from $0$ to $n-1$, where $n$ is the number of nodes in the graphical dataset. Each line in the text file is formatted as `a b` where `a` and `b` are nodes in the graph. 

[load_graphs.py](load_graphs.py) creates graph objects 
