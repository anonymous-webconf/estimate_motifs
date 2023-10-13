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
1. **Graph_RWIS**: In `class_graphs.py`, use the class method `rw_count_motif_Gd()` to count any connected graphlet that is discoverable and countable for $G$ and $\mathcal{L}(\tilde{G})$. Refer to our literature for details. For example, _source-S3_ count can only be estimated on $\mathcal{L}(\tilde{G})$. To estimate its count under this framework for Epinions dataset, over $10^4$ walk steps and , pass the following arguments:
   ```
   source_S3_epinions_estimate = graph_epinions_rwis.rw_count_motif_Gd(seed = random.choice(list(graph_epinions.edges.keys())), walk_len = int(1e4), motif_adj_mat = source_S3_adj_mat, k = len(source_S3_adj_mat), d=2)
    ```
   >[!NOTE]
   >The type of seed depends upon the value of parameter `d`. For `d=1`, the seed should be among the vertices in the extracted component. When `d=2`, the seed should be among the edges of the extracted graph component.

2. **Graph_RWRJ**: In `class_graphs.py`, use the class method `rw_count_motif_rwrj()` to count any sized graphlet under the RW-RJ model. There are no constraints on the _type_ of graphlet that can be counted. Refer to our literature for details. For example, to estimate _butterfly_ count under this framework for Epinions dataset, over $10^4$ walk steps and $\alpha=0.6$, pass the following arguments:
   ```
   butterfly_epinions_estimate = graph_epinions_rwrj.rw_count_motif_rwrj(seed = random.choice(graph_epinions.valid_vertices), walk_len = int(1e4), motif_adj_mat = butterfly_adj_mat, k = len(butterfly_adj_mat), alpha=0.6)
   ```
   >[!NOTE]
   >Be sure to load the stationary distribution values corresponding to the given alpha for the graph object before estimating counts!
   
3. **Graph_RWBB**: In `class_graphs.py`, use the class method `rw_count_motif_back_button()` to count any graphlet that is discoverable and countable under back button model constraints. Refer to our literature for details. For example, to estimate _cyclic-C4_ count under this framework for Epinions dataset, over $10^4$ walk steps and $\alpha=0.8$, pass the following arguments:
   ```
   cyclic_C4_epinions_estimate = graph_epinions.rw_count_motif_back_button(seed = random.choice(graph_epinions.valid_vertices), walk_len = int(1e4), motif_adj_mat = cyclic_C4_adj_mat, k = len(cyclic_C4_adj_mat), alpha = 0.8)
   ```

## Testing Frameworks' Accuracy
There are two types of tests that we conduct for our frameworks: estimation accuracy against ground truths and estimation accuracy for different $\alpha$ values across all mentioned graphlets.

### Estimation Accuracy
First, we generate a CSV file named `dataset_walk_data.csv` that stores a defined length random walk using the three frameworks (and pre-decided $\alpha$ for RW-RJ and RW-BB). We use the function `populate_walk_csv()` in file [`populate_walk_data.py`](populate_walk_data.py) to generate walk sequences for the three methods. 

We use NRMSE to account for accuracy and variance in the returned data points. We generate another CSV file named `dataset_error_details.csv` with error data points for a fixed length walks taken from `dataset_walk_data.csv`. The function `populate_error()` in file [`populate_error_details.py`](populate_error_details.py) generate these error data points. We use multiprocessing library to speed the computation.

### Alpha Testing
Our code base supports $\alpha$ testing. Specifically for RW-RJ and RW-BB, the choice of $\alpha$ for a given graphlet impacts the NRMSE. We generate fixed length random walks for the two frameworks across a range of $\alpha$ values using `populate_alpha_walk_csv()` and `populate_alpha_BB_walk_csv()` in file [`populate_walk_data.py`](populate_walk_data.py)
```
populate_alpha_walk_csv(epinions_rwrj, 30000, "epinions")
```
The above command populates `dataset_alpha_walk_details.csv` with 30000 length walks on Epinions dataset for $\alpha\in [0.1,0.9]$. `populate_alpha_BB_walk_csv()` follows similar syntax but with $\alpha\geq 0.5$ due to back button model constraints.
```
populate_alpha_BB_walk_csv(wiki_vote_rwbb, 10000, "wiki-vote")
```
The above code populates `dataset_alpha_BB_walk_details.csv` with 10000 length random walks using RW-BB model over Wiki-Vote dataset.

### Plotting
To show the convergence of our estimates, we use `create_error_plot()` function in [`error_create_plots.py`](error_create_plots.py) to create line plots, showing convergence at given `step size` over the total walk length.
```
create_error_plot(df = pd.read_csv(csv_file, index_col=0), save_loc = "test_fresh/count_error_new")
```
The above command will create a plot figure for each column of error_details file. Therefore, one plot for a dataset, graphlet tuple is created and saved in directory `test_fresh/count_error_new`

The final NRMSE values are compared for each graphlet across the three frameworks. A figure is generated for each dataset. To create an error bar plot given a populated `dataset_error_details.csv` file, use the following command:
```
create_error_bar("dataset_error_details.csv", ["twitter-weak"], ["cyclic-C3", "wedge", "leaf-C3", "butterfly"], 44000, "test_fresh", num_fw = 3)
```
This populates the directory `test_fresh` with a bar plot of Twitter dataset (weakly connected). The NRMSE for graphlets are compared in the plot for the walk length of 44000 steps. Since the dataset is weakly connected, we will skip the RW-BB results, hence `num_fw=3` (number of bar plots per graphlet: RW-IS on $G$, RW-IS on $\mathcal{L}(\tilde{G})$ and RW-RJ on $G$)

Similar syntax follows to create $\alpha$ testing plots for RW-RJ and RW-BB. The function `create_alpha_plots()` creates $\alpha$ convergence comparison plots. A figure is generated for each dataset, graphlet tuple.

To compare the final NRMSE values among different $\alpha$ across all graphlets, we provide `create_alpha_error_bar()` function. A figure is generated for each dataset.
```
create_alpha_error_bar("dataset_alpha_BB_error_details.csv", ["wiki-vote"], 
                 ["cyclic-C3", "wedge", "leaf-C3", "cyclic-C4", "clique-K4"], [0.5,0.6,0.7,0.8,0.9], "test_fresh", "RWBB")
```
The above command creates a single plot for Wiki-Vote dataset saved in the `test_fresh` directory. For each graphlet, there are 5 bars that indicate final NRMSE values for corresponding $\alpha$ values at the complete walk length indicated by `dataset_alpha_BB_error_details.csv` file for RW-BB framework.
