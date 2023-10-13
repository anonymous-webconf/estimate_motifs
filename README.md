# Frameworks to Estimate Motif Counts in Directed Networks
This is our code base, fully developed by us in Python to evaluate the performance of our three frameworks (namely RW-IS, RW-RJ and RW-BB). The files provided in this  repository can be leveraged to create graph objects that are compatible with the three techniques. We have predefined exact counting functions for each class that can count _wedges, cyclic-C3, cyclic-C4, source-S3, leaf-C3, butterfly, clique-K4_ graphlets. Below is the pictorial representation of each graphlet. These exact counts form the ground truths for our error comparisons across the techniques introduced. 

## About the Classes:
The file [`class_graphs.py`](class_graphs.py) contains three graph classes: `Graph_RWIS`, `Graph_RWRJ` and `Graph_RWBB`. Each class supports the corresponding estimation framework. For all the three classes, there are pre-defined counting functions for our graphlets of interest. To count _cyclic-C3_ for example on a RW-IS graph object named `graph_obj`, run the following:
```
num_C3 = count_cyclic_C3(graph_obj)
```
Note that the adjacency matrices representing the graphlets are defined on top of file [`class_graphs.py`](class_graphs.py) 
The file [`get_accurate_count.py`](get_accurate_count.py) leverages multiprocessing to extract exact graphlet counts. The user can define his own counting function in `class_graphs.py` and use `get_accurate_count.py` to get the exact count.

Walk operations are strictly class specific. For instance, a RW-BB graph object will not support emulating a walk on $\mathcal{L}(\tilde{G})$. To do so, consider creating a RW-IS graph object. Similarly, random walk that supports random jumps are supported by RW-RJ only. The tunable parameter $\alpha$ is passed to RW-RJ and RW-BB objects for random walks on respective graphs.

## Creating a Graph:
The dataset should be in the form of text file with node labels ranging from $0$ to $n-1$, where $n$ is the number of nodes in the graphical dataset. Each line in the text file is formatted as `a b` where `a` and `b` are nodes in the graph. 

[`load_graphs.py`](load_graphs.py) creates graph objects that support either RW-IS, RW-RJ or RW-BB framework. The function `create_graph_from_file()` takes the dataset text file as an input. To allow bi-directional edges in the graph, set flag `allow_multi_edges=True`. Else, the first tuple `a b` that occurs in the dataset will be considered and subsequent tuples of type `b a` will be ignored, if any. `line_skips` takes an integer that skips the specific number of lines in the dataset text file (you may want to skip the starting 4 lines in the dataset if they contain contextual information about the dataset, and no edge tuples).

After a graph has been loaded by `create_graph_from_file()', it is advised to extract a strongly/ weakly connected component from the graph. Use [`extract_maximal_scc.py`](extract_maximal_scc.py) for the former and `graph_obj.BFS(seed)` for the latter. 

**IMPORRANT:** Any graph 
