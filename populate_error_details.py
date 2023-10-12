from class_graphs import Graph_RWIS, Graph_RWRJ, Graph_RWBB, \
    cyclic_C3_adj_mat, cyclic_C4_adj_mat, butterfly_adj_mat, leaf_C3_adj_mat, clique_K4_mat, wedge_adj_mat
from load_graphs import create_graph_cleaned_file
import pandas as pd
from multiprocessing import Process
from multiprocessing import Lock

walk_data = pd.read_csv("dataset_walk_data.csv", index_col=0)
alpha_walk_data = pd.read_csv("dataset_alpha_walk_details.csv", index_col=0)
alpha_BB_walk_data = pd.read_csv("dataset_alpha_BB_walk_details.csv", index_col=0)
actual_counts = pd.read_csv("dataset_actual_counts.csv", index_col=0)

step_size = 1000

def populate_error(lock, dataset_name: str, graphlet_name : str, mode : str):
    file_path = "datasets/cleaned_datasets_new/{}_cln.txt".format(dataset_name)
    print("Counting {} in {} with mode {}".format(graphlet_name, dataset_name, mode))
    k = -1
    motif_adj_mat = None
    if graphlet_name == "cyclic-C3":
        motif_adj_mat = cyclic_C3_adj_mat
    elif graphlet_name == "cyclic-C4":
        motif_adj_mat = cyclic_C4_adj_mat
    elif graphlet_name == "butterfly":
        motif_adj_mat = butterfly_adj_mat
    elif graphlet_name == "leaf-C3":
        motif_adj_mat = leaf_C3_adj_mat
    elif graphlet_name == "clique-K4":
        motif_adj_mat = clique_K4_mat
    elif graphlet_name == "wedge":
        motif_adj_mat = wedge_adj_mat
    assert motif_adj_mat is not None
    k = len(motif_adj_mat)
    assert k != -1

    a_ct = actual_counts[dataset_name][graphlet_name]

    assert mode in ["rwis", "rwrj", "rwbb"]

    if mode == "rwis":

        graph_is = create_graph_cleaned_file(file_path, "rwis")
        r_G1, error_G1 = graph_is.plot_error_motif(
                        list_walk_list= eval(walk_data[dataset_name]["RWIS_G1"]), step_size=step_size,
                        motif_adj_mat=motif_adj_mat, k=k, actual_count=a_ct, d=1)
        print("Counting {} in {} as RWIS-G1 COMPLETED".format(graphlet_name, dataset_name))

        r_G2, error_G2 = graph_is.plot_error_motif(
            list_walk_list= eval(walk_data[dataset_name]["RWIS_G2"]), step_size=step_size,
            motif_adj_mat=motif_adj_mat, k=k, actual_count=a_ct, d=2)
        print("Counting {} in {} as RWIS-G2 COMPLETED".format(graphlet_name, dataset_name))

        with lock:
            error_details = pd.read_csv("dataset_error_details.csv", index_col=0)
            col_name = dataset_name+"_"+graphlet_name
            if col_name not in error_details.columns:
                error_details[col_name] = [-1]*len(error_details.index)
            error_details.loc["RWIS_G1", col_name] = str({'r': r_G1, 'error': error_G1})
            error_details.loc["RWIS_G2", col_name] = str({'r': r_G2, 'error': error_G2})
            error_details.to_csv("dataset_error_details.csv")

    elif mode == "rwrj":
        graph_rj = create_graph_cleaned_file(file_path, "rwrj")
        graph_rj.load_walk_distribution(iterations=0,alpha=0.8, file = "datasets/cleaned_datasets_new/stationary_dist_{}_RWRJ.txt".format(dataset_name))
        r_RJ, error_RJ = graph_rj.plot_error_motif(
            list_walk_list= eval(walk_data[dataset_name]["RWRJ"]), step_size=step_size,
            motif_adj_mat=motif_adj_mat, k=k, actual_count=a_ct, alpha=0.8)
        print("Counting {} in {} as RWRJ COMPLETED".format(graphlet_name, dataset_name))

        with lock:
            error_details = pd.read_csv("dataset_error_details.csv", index_col=0)
            col_name = dataset_name+"_"+graphlet_name
            if col_name not in error_details.columns:
                error_details[col_name] = [-1]*len(error_details.index)
            error_details.loc["RWRJ", col_name] = str({'r': r_RJ, 'error': error_RJ})
            error_details.to_csv("dataset_error_details.csv")
    
    elif mode == "rwbb":
        graph_bb = create_graph_cleaned_file(file_path, "rwbb")
        graph_bb.load_walk_distribution(iterations=0,alpha=0.8, file = "datasets/cleaned_datasets_new/stationary_dist_{}_RWBB.txt".format(dataset_name))
        r_BB, error_BB = graph_bb.plot_error_motif(
            list_walk_list= eval(walk_data[dataset_name]["RWBB"]), step_size=step_size,
            motif_adj_mat=motif_adj_mat, k=k, actual_count=a_ct, alpha=0.8)
        print("Counting {} in {} as RWBB COMPLETED".format(graphlet_name, dataset_name))

        with lock:
            error_details = pd.read_csv("dataset_error_details.csv", index_col=0)
            col_name = dataset_name+"_"+graphlet_name
            if col_name not in error_details.columns:
                error_details[col_name] = [-1]*len(error_details.index)
            error_details.loc["RWBB", col_name] = str({'r': r_BB, 'error': error_BB})
            error_details.to_csv("dataset_error_details.csv")

# -----------------------------------------------------------------------------------

def populate_alpha_error(dataset_name: str, graphlets: list):
    step_size = 200
    file_path = "datasets/cleaned_datasets_new/{}_cln.txt".format(dataset_name)
    error_details = pd.read_csv("dataset_alpha_error_details.csv", index_col=0)
    graph_rj = create_graph_cleaned_file(file_path, "rwrj")

    for alpha in list(error_details.index):
        list_walk_list = eval(alpha_walk_data.loc[alpha, dataset_name])
        graph_rj.load_walk_distribution(iterations=0,alpha=alpha, file = None, accurate=True)

        for graphlet_name in graphlets:
            a_ct = actual_counts[dataset_name][graphlet_name]
            col_name = dataset_name+"_"+graphlet_name
            if col_name not in error_details.columns:
                error_details[col_name] = [-1]*len(error_details.index)
            print("Current Computation:\talpha {}, dataset {}, graphlet {}".format(alpha, dataset_name, graphlet_name))
            k = -1
            if graphlet_name == "cyclic-C3":
                motif_adj_mat = cyclic_C3_adj_mat
            elif graphlet_name == "cyclic-C4":
                motif_adj_mat = cyclic_C4_adj_mat
            elif graphlet_name == "butterfly":
                motif_adj_mat = butterfly_adj_mat
            elif graphlet_name == "leaf-C3":
                motif_adj_mat = leaf_C3_adj_mat
            elif graphlet_name == "clique-K4":
                motif_adj_mat = clique_K4_mat
            elif graphlet_name == "wedge":
                motif_adj_mat = wedge_adj_mat
            assert motif_adj_mat is not None
            k = len(motif_adj_mat)
            assert k != -1
            r_RJ, error_RJ = graph_rj.plot_error_motif(
                                list_walk_list= list_walk_list, step_size=step_size,
                                motif_adj_mat=motif_adj_mat, k=k, actual_count=a_ct, alpha=alpha)
            error_details.loc[alpha, col_name] = str({'r': r_RJ, 'error': error_RJ})

    error_details.to_csv("dataset_alpha_error_details.csv")
        
# -----------------------------------------------------------------------------------  

def populate_alpha_BB_error(dataset_name: str, graphlets: list):
    step_size = 200
    file_path = "datasets/cleaned_datasets_new/{}_cln.txt".format(dataset_name)
    error_details = pd.read_csv("dataset_alpha_BB_error_details.csv", index_col=0)
    graph_bb = create_graph_cleaned_file(file_path, "rwbb")

    for alpha in list(error_details.index):
        list_walk_list = eval(alpha_BB_walk_data.loc[alpha, dataset_name])
        graph_bb.load_walk_distribution(iterations=0,alpha=alpha, 
                                        file = "datasets/cleaned_datasets_new/stationary_dist_{}_RWBB.txt".format(dataset_name))

        for graphlet_name in graphlets:
            a_ct = actual_counts[dataset_name][graphlet_name]
            col_name = dataset_name+"_"+graphlet_name
            if col_name not in error_details.columns:
                error_details[col_name] = [-1]*len(error_details.index)
            print("Current Computation BB:\talpha {}, dataset {}, graphlet {}".format(alpha, dataset_name, graphlet_name))
            k = -1
            if graphlet_name == "cyclic-C3":
                motif_adj_mat = cyclic_C3_adj_mat
            elif graphlet_name == "cyclic-C4":
                motif_adj_mat = cyclic_C4_adj_mat
            elif graphlet_name == "butterfly":
                motif_adj_mat = butterfly_adj_mat
            elif graphlet_name == "leaf-C3":
                motif_adj_mat = leaf_C3_adj_mat
            elif graphlet_name == "clique-K4":
                motif_adj_mat = clique_K4_mat
            elif graphlet_name == "wedge":
                motif_adj_mat = wedge_adj_mat
            assert motif_adj_mat is not None
            k = len(motif_adj_mat)
            assert k != -1
            r_BB, error_BB = graph_bb.plot_error_motif(
                                list_walk_list= list_walk_list, step_size=step_size,
                                motif_adj_mat=motif_adj_mat, k=k, actual_count=a_ct, alpha=alpha)
            error_details.loc[alpha, col_name] = str({'r': r_BB, 'error': error_BB})

    error_details.to_csv("dataset_alpha_BB_error_details.csv")
        
# -----------------------------------------------------------------------------------  

# processes = []

# lock = Lock()

# datasets = ["wiki-vote", "epinions", "twitter-weak"]
# graphlets = ["cyclic-C3", "cyclic-C4", "butterfly", "leaf-C3", "clique-K4", "wedge"]
# modes = ["rwis", "rwrj", "rwbb"]

# subset_graphlets = ['cyclic-C3', 'wedge', 'butterfly', 'leaf-C3']

# for d in datasets[-1:]:
#     for g in subset_graphlets[3:4]:
#         for m in modes[:-1]:
#             processes.append(Process(target=populate_error, args = (lock, d,g,m)))
   
# for p in processes:
#     p.start()

# for p in processes:
#     p.join()
    

# # -----------------------------------------------------------------------------------

processes = []

datasets = ["wiki-vote"]

graphlets = ["cyclic-C3", "cyclic-C4", "butterfly", "leaf-C3", "clique-K4", "wedge"]
graphlets = ["cyclic-C3", "cyclic-C4", "leaf-C3", "clique-K4", "wedge"]

for d in datasets:
    processes.append(Process(target=populate_alpha_BB_error, args = (d,graphlets)))
   
for p in processes:
    p.start()
    p.join()

# # -----------------------------------------------------------------------------------

# processes = []

# datasets = ["epinions", "wiki-vote"]
# graphlets = ["cyclic-C3", "cyclic-C4", "butterfly", "leaf-C3", "clique-K4"]

# for d in datasets:
#     for g in graphlets:
#         processes.append(Process(target=load_from_files_to_csv, args = (d,g)))
   
# for p in processes:
#     p.start()
#     p.join()