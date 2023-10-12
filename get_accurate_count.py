from load_graphs import *
from class_graphs import count_acyclic_C3, count_cyclic_C3, count_cyclic_C4, \
    count_source_S3, count_butterfly, count_leaf_C3, count_K4_clique, count_wedge
import multiprocessing 

valid_graphlets = ['cyclic C3', 'acyclic C3', 'source S3', 'cyclic C4', 'butterfly', 'leaf-C3', 'clique-K4', 'wedge']

def exact_count_graph (graphlet_type: str, graph: Graph_RWRJ, dataset_name: str):
    print("---------------------------------------------------------")
    print("COUNTING {} in {}".format(graphlet_type, dataset_name))
    if graphlet_type == 'cyclic C3':
        print("{} count for {} : {}".format(graphlet_type, dataset_name, count_cyclic_C3(graph)))
    elif graphlet_type == 'acyclic C3':
        print("{} count for {} : {}".format(graphlet_type, dataset_name, count_acyclic_C3(graph)))
    elif graphlet_type == 'source S3':
        print("{} count for {} : {}".format(graphlet_type, dataset_name, count_source_S3(graph)))
    elif graphlet_type == 'cyclic C4':
        print("{} count for {} : {}".format(graphlet_type, dataset_name, count_cyclic_C4(graph)))
    elif graphlet_type == 'butterfly':
        print("{} count for {} : {}".format(graphlet_type, dataset_name, count_butterfly(graph)))
    elif graphlet_type == 'leaf-C3':
        print("{} count for {} : {}".format(graphlet_type, dataset_name, count_leaf_C3(graph)))
    elif graphlet_type == 'clique-K4':
        print("{} count for {} : {}".format(graphlet_type, dataset_name, count_K4_clique(graph)))
    elif graphlet_type == 'wedge':
        print("{} count for {} : {}".format(graphlet_type, dataset_name, count_wedge(graph)))
    else:
        print("No graphlet type")
    print("---------------------------------------------------------")


def count_all_graphlets(graphlets: list, graph, dataset: str):
    processes = []
    for g in graphlets:
        assert g in valid_graphlets
        processes.append(multiprocessing.Process(target = exact_count_graph,
                                args = (g, graph, dataset)))
    for p in processes:
        p.start()
    for p in processes:
        p.join()

# ---------------------------------------------------------------------------------

graph_epi = create_graph_cleaned_file('datasets/cleaned_datasets_new/epinions_cln.txt', 'rwrj', False)
graph_wiki = create_graph_cleaned_file('datasets/cleaned_datasets_new/wiki-vote_cln.txt', 'rwrj', False)
graph_twitter_weak = create_graph_cleaned_file('datasets/cleaned_datasets_new/twitter-weak_cln.txt', 'rwis', False)

graphlets = ['cyclic C3', 'cyclic C4', 'butterfly', 'leaf-C3', 'clique-K4', 'wedge']

subset_graphlets = ['cyclic C3', 'butterfly', 'leaf-C3', 'wedge']

# p1 = multiprocessing.Process(target = count_all_graphlets
#                                 , args = (graphlets[-1:], graph_epi, 'soc-Epinions'))
# p2 = multiprocessing.Process(target = count_all_graphlets
#                                 , args = (graphlets[-1:], graph_wiki, 'wiki-Vote'))
# p1.start()
# p2.start()
# p1.join()
# p2.join()
p1 = multiprocessing.Process(target = count_all_graphlets
                                , args = (subset_graphlets[2:3], graph_twitter_weak, 'Twitter Weak Connected'))
p1.start()
p1.join()