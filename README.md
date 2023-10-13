# Frameworks to Estimate Motif Counts in Directed Networks
This is our code base, fully developed by us in Python to evaluate the performance of our three frameworks (namely RW-IS, RW-RJ and RW-BB). The files provided in this  repository can be leveraged to create graph objects that are compatible with the three techniques. We have predefined exact counting functions for each class that can count _wedges, cyclic-C3, cyclic-C4, source-S3, leaf-C3 (G24), butterfly, clique-K4_ graphlets. Below is the pictorial representation of each graphlet. These exact counts form the ground truths for our error comparisons across the techniques introduced. 
![image](https://github.com/anonymous-webconf/estimate_motifs/assets/147694163/81f88fa6-f71c-4097-b29b-783c137b6d8f)


## About the Classes:
The file [`class_graphs.py`](class_graphs.py) contains three graph classes: `Graph_RWIS`, `Graph_RWRJ` and `Graph_RWBB`. Each class supports the corresponding estimation framework. When loading graphs from dataset, it is mandatory to select one mode among `rwis`, `rwrj` or `rwbb`. 

Walk operations are strictly class specific. For instance, a RW-BB graph object will not support emulating a walk on $\mathcal{L}(\tilde{G})$. To do so, consider creating a RW-IS graph object. Similarly, random walk that supports random jumps are supported by RW-RJ only. The tunable parameter $\alpha$ is passed to RW-RJ and RW-BB objects for random walks on respective graphs.

## Creating a Graph:
The dataset should be in the form of text file with node labels ranging from $0$ to $n-1$, where $n$ is the number of nodes in the graphical dataset. Each line in the text file is formatted as `a b` where `a` and `b` are nodes in the graph. 

[`load_graphs.py`](load_graphs.py) creates graph objects that support either RW-IS, RW-RJ or RW-BB framework. The function `create_graph_from_file()` takes the dataset text file as an input. To allow bi-directional edges in the graph, set flag `allow_multi_edges=True`. Else, the first tuple `a b` that occurs in the dataset will be considered and subsequent tuples of type `b a` will be ignored, if any. `line_skips` takes an integer that skips the specific number of lines in the dataset text file (you may want to skip the starting 4 lines in the dataset if they contain contextual information about the dataset, and no edge tuples).

After a graph has been loaded by `create_graph_from_file()`, it is advised to extract a strongly/ weakly connected component from the graph. Use [`extract_maximal_scc.py`](extract_maximal_scc.py) for the former and `graph_obj.BFS(seed)` for the latter. 

>[!IMPORTANT]
>Any counting or walk subroutine necessitates a _cleaned_ graph. Once the maximal connected component has been extracted from the dataset, use `write_to_file()` in `load_graphs.py` to create a text file graph dataset. Then, load your graph using `create_graph_cleaned_file()` with the appropriate mode. Datasets are given in the [`datasets/complete_datasets`](datasets/complete_datasets) directory. We have extracted the maximal strongly connected component from some of these large datasets and stored them in the directory [`datasets/cleaned_datasets_new`](datasets/cleaned_datasets_new). If the user wishes to use RW-RJ counting framework on [Twitter weakly connected dataset](datasets/cleaned_datasets_new/twitter-weak_cln.txt), use the following command:
```
graph_twitter_weak = create_graph_cleaned_file('datasets/cleaned_datasets_new/twitter-weak_cln.txt', 'rwrj', False)
```

The created graph object is now compatible for all exact counting subroutines and RW-RJ based random walks.

## Exact Counting:
For all the three classes, there are pre-defined counting functions for our graphlets of interest. To count _cyclic-C3_ for example on a RW-IS graph object named `graph_obj`, run the following:
```
num_C3 = count_cyclic_C3(graph_obj)
```
Note that the adjacency matrices representing the graphlets are defined on top of file [`class_graphs.py`](class_graphs.py) 
The file [`get_accurate_count.py`](get_accurate_count.py) leverages multiprocessing to extract exact graphlet counts. The user can define his own counting function in `class_graphs.py` and use `get_accurate_count.py` to get the exact count.

## Getting the Stationary Distribution:
According to the shared literature, RW-RJ and RW-BB require the values for stationary distribution of nodes for graphlet count estimation. For RW-IS, the stationary distribution is known without any pre-computation. There are two ways to compute the stationary distribution of underlying Markov chain:
1. Compute the stationary distribution exactly (recommended for smaller graphs with $|V|\leq \mathcal{O}(10^5)$)
2. Estimate the stationary distribution for large datasets

We use left eigen vector as $\vec{\pi} = \vec{\pi}\cdot \mathbf{P}$ where $\mathbf{P}$ is the transition matrix for the graph. The function `get_stationary_distribution(transition_marix)` in file [`get_stationary.py`](get_stationary.py) returns the exact stationary distribution.

### Graph_RWRJ
For objects of this class, the pre-defined functionality `graph_obj.load_walk_distribution()`. Flag `accurate=True` returns an exact stationary distribution. It might be beneficial to store the result by writing the returned dictionary as a string in a returned file. Make sure to store the corresponding $\alpha$ value in a new line following the dictionary. To read the distribution from file, pass the data file as an argument in the function. For example, the file [`datasets/cleaned_datasets_new/stationary_dist_epinions_RWRJ.txt`](datasets/cleaned_datasets_new/stationary_dist_epinions_RWRJ.txt) stores the computed stationary distribution for [soc-Epinions cleaned dataset](datasets/cleaned_datasets_new/epinions_cln.txt) for $\alpha=0.8$. To load the stored stationary distribution, use the following command:
```
graph_obj.load_walk_distribution(iterations=0, alpha=0.8, file= "datasets/cleaned_datasets_new/stationary_dist_epinions_RWRJ.txt", accurate=False)
```
This populates the stationary distribution values in the RW-RJ graph object.

To estimate the stationary distribution over a walk length of, say, $10^6$ steps for $\alpha=0.8$, pass the following arguments:
```
graph_obj.load_walk_distribution(iterations=int(1e6), alpha=0.8, file= None, accurate=False)
```

### Graph_RWBB
A similar menthod named `load_walk_distribution()` exists for RW-BB class based graphs. Assign `True` to the `accurate` parameter for accurate distribution computation. Assigning a valid file path to `file` parameter reads the stored distribution dictionary from the file and loads it with the graph object. Note that for these objects, the stationary distribution is independent of $\alpha$ and depends on the underlying Markov chain (or the graphical structure). To estimate the stationary distribution for a graph object of this model over $10^8$ walk steps, use the following arguments:
```
graph_obj.load_walk_distribution(iterations=int(1e8), alpha=0.8, file= None, accurate=False)
```
Note that the $\alpha$ parameter is only used to generate a random walk sequence. the resulting stationary distribution is analytically independent of $\alpha$ for $\alpha\in[0.5,1]$.

>[!IMPORTANT]
>For RW-BB, the stationary distribution exists if and only if alpha>= 0.5. Do NOT use a lower alpha value while estimating stationary distribution in RW-BB based objects as they will not yield a correct stationary distribution estimate.

## Estimating Graphlet Count
Any sized graphlet can be counted using the three frameworks. Unlike exact counting, where a counting function is required to count the number of graphlets in the graph, our random walk based implementaion only requires the adjacency matrix of the graphlet to estimate its count. Below are the methods to count graphlets across the three classes:
1. **Graph_RWIS**: In `class_graphs.py`, use the class method `rw_count_motif_back_button()` to count any graphlet that is discoverable and countable under back button model constraints. Refer to our literature for details. For example, to estimate _cyclic-C4_ count under this framework for Epinions dataset, over $10^4$ walk steps and $\alpha=0.8$, pass the following arguments:
   ```
   rw_count_motif_back_button(self, seed, walk_len: int, motif_adj_mat: list, k: int, alpha: int)
   ```
2. **Graph_RWRJ**: In `class_graphs.py`, use the class method `rw_count_motif_back_button()` to count any graphlet that is discoverable and countable under back button model constraints. Refer to our literature for details. For example, to estimate _cyclic-C4_ count under this framework for Epinions dataset, over $10^4$ walk steps and $\alpha=0.8$, pass the following arguments:
   ```
   rw_count_motif_back_button(self, seed, walk_len: int, motif_adj_mat: list, k: int, alpha: int)
   ```
3. **Graph_RWBB**: In `class_graphs.py`, use the class method `rw_count_motif_back_button()` to count any graphlet that is discoverable and countable under back button model constraints. Refer to our literature for details. For example, to estimate _cyclic-C4_ count under this framework for Epinions dataset, over $10^4$ walk steps and $\alpha=0.8$, pass the following arguments:
   ```
   rw_count_motif_back_button(self, seed, walk_len: int, motif_adj_mat: list, k: int, alpha: int)
   ```
