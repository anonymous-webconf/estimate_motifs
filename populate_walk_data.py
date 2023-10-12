import pandas as pd
from class_graphs import Graph_RWIS, Graph_RWBB, Graph_RWRJ
from load_graphs import create_graph_cleaned_file
from typing import Union
import random
from multiprocessing import Process

# wiki_vote_rwis = create_graph_cleaned_file("datasets/cleaned_datasets_new/wiki-vote_cln.txt", "rwis", False)
# wiki_vote_rwrj = create_graph_cleaned_file("datasets/cleaned_datasets_new/wiki-vote_cln.txt", "rwrj", False)
# wiki_vote_rwbb = create_graph_cleaned_file("datasets/cleaned_datasets_new/wiki-vote_cln.txt", "rwbb", False)

# epinions_rwis = create_graph_cleaned_file("datasets/cleaned_datasets_new/epinions_cln.txt", "rwis", False)
# epinions_rwrj = create_graph_cleaned_file("datasets/cleaned_datasets_new/epinions_cln.txt", "rwrj", False)
# epinions_rwbb = create_graph_cleaned_file("datasets/cleaned_datasets_new/epinions_cln.txt", "rwbb", False)

twitter_weak_rwis = create_graph_cleaned_file("datasets/cleaned_datasets_new/twitter_weak_cln.txt", "rwis", False)
twitter_weak_rwrj = create_graph_cleaned_file("datasets/cleaned_datasets_new/twitter_weak_cln.txt", "rwrj", False)

iterations = 200
walk_len = 50000
alpha = 0.8

def populate_walk_csv (graph_is: Graph_RWIS, graph_rj: Graph_RWRJ, graph_bb: Graph_RWBB,
                       dataset_name : str):
    walk_data = pd.read_csv("dataset_walk_data.csv", index_col=0)
    if dataset_name not in walk_data.columns:
        walk_data[dataset_name] = [-1]*len(walk_data.index)
    list_g1 =[]
    list_g2 = []
    list_bb = []
    list_rj = []
    last_print = -1

    for i in range(iterations):
        curr_per = (i+1)*100/iterations
        if curr_per%10 == 0 and last_print != curr_per:
            print("{}% progress in populating {}".format(curr_per, dataset_name))

        list_g1.append(graph_is.random_walk_on_Gd
                       (seed = random.choice(graph_is.valid_vertices) , walk_length=walk_len, d=1))
        list_g2.append(graph_is.random_walk_on_Gd
                       (seed = random.choice(list(graph_is.edges.keys())) , walk_length=walk_len, d=2))
        list_rj.append(graph_rj.random_walk_rwrj
                       (seed = random.choice(graph_rj.valid_vertices), walk_length= walk_len, alpha=alpha))
        if graph_bb is not None:
            list_bb.append(graph_bb.random_walk_rwbb
                       (seed = random.choice(graph_bb.valid_vertices), walk_length=walk_len, alpha=alpha))
        
    walk_data.loc["RWIS_G1",dataset_name] = str(list_g1)
    print("{} G1 done".format(dataset_name))

    walk_data.loc["RWIS_G2",dataset_name] = str(list_g2)
    print("{} G2 done".format(dataset_name))

    walk_data.loc["RWRJ",dataset_name] = str(list_rj)
    print("{} RJ done".format(dataset_name))

    if graph_bb is not None:
        walk_data.loc["RWBB",dataset_name] = str(list_bb)
        print("{} BB done".format(dataset_name))

    walk_data.to_csv("dataset_walk_data.csv")

def populate_alpha_walk_csv(graph_rj: Graph_RWRJ, walk_length:int, dataset_name: str):
    walk_data = pd.read_csv("dataset_alpha_walk_details.csv", index_col=0)
    last_print = -1
    for alpha in list(walk_data.index):
        list_rj = []
        for i in range(iterations):
            curr_per = (i+1)*100/iterations
            if curr_per%20 == 0 and last_print != curr_per:
                print("{}% progress in populating alpha {} walk for {}".format(curr_per, alpha, dataset_name))
                last_print = curr_per
            list_rj.append(graph_rj.random_walk_rwrj
                        (seed = random.choice(graph_rj.valid_vertices), walk_length= walk_length, alpha=alpha))
        print("Completed adding walk for alpha: ",alpha)

        walk_data.loc[alpha,dataset_name] = str(list_rj)

    walk_data.to_csv("dataset_alpha_walk_details.csv")

def populate_alpha_BB_walk_csv(graph_bb: Graph_RWBB, walk_length:int, dataset_name: str):
    walk_data = pd.read_csv("dataset_alpha_BB_walk_details.csv", index_col=0)
    last_print = -1
    for alpha in list(walk_data.index):
        list_bb = []
        for i in range(iterations):
            curr_per = (i+1)*100/iterations
            if curr_per%20 == 0 and last_print != curr_per:
                print("{}% progress in populating alpha {} walk for {}".format(curr_per, alpha, dataset_name))
                last_print = curr_per
            list_bb.append(graph_bb.random_walk_rwbb
                        (seed = random.choice(graph_bb.valid_vertices), walk_length= walk_length, alpha=alpha))
        print("Completed adding walk for alpha: ",alpha)

        walk_data.loc[alpha,dataset_name] = str(list_bb)

    walk_data.to_csv("dataset_alpha_BB_walk_details.csv")


# populate_walk_csv(wiki_vote_rwis, wiki_vote_rwrj, wiki_vote_rwbb, "wiki-vote")
# populate_walk_csv(epinions_rwis, epinions_rwrj, epinions_rwbb, "epinions")

# populate_alpha_walk_csv(epinions_rwrj, 30000, "epinions")
# populate_alpha_walk_csv(wiki_vote_rwrj, 10000, "wiki-vote")

# populate_alpha_BB_walk_csv(wiki_vote_rwbb, 10000, "wiki-vote")
populate_walk_csv(twitter_weak_rwis, twitter_weak_rwrj, None, "twitter-weak")