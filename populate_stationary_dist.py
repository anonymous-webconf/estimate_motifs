from class_graphs import Graph_RWIS, Graph_RWBB
from load_graphs import create_graph_cleaned_file
import random

# graph_wiki_bb = create_graph_cleaned_file("datasets/cleaned_datasets_new/wiki-vote_cln.txt", "rwbb")
# graph_wiki_rj = create_graph_cleaned_file("datasets/cleaned_datasets_new/wiki-vote_cln.txt", "rwrj")

# graph_soc_bb = create_graph_cleaned_file("datasets/cleaned_datasets_new/epinions_cln.txt", "rwbb")
# graph_soc_rj = create_graph_cleaned_file("datasets/cleaned_datasets_new/epinions_cln.txt", "rwrj")

# graph_wiki_bb.load_walk_distribution(0,0,accurate=True)
# f = open("datasets/cleaned_datasets_new/stationary_dist_wiki-vote_RWBB.txt", "w")
# f.write(str(graph_wiki_bb.stationary_dist))

# graph_soc_bb.load_walk_distribution(0,0,accurate=True)
# f = open("datasets/cleaned_datasets_new/stationary_dist_epinions_RWBB.txt", "w")
# f.write(str(graph_soc_bb.stationary_dist))

# alpha = 0.8
# graph_wiki_rj.load_walk_distribution(0,0.8,accurate=True)
# f = open("datasets/cleaned_datasets_new/stationary_dist_wiki-vote_RWRJ.txt", "w")
# f.write(str(graph_wiki_rj.stationary_dist))
# f.write("\n")
# f.write("Alpha: {}".format(alpha))

# alpha = 0.8
# graph_soc_rj.load_walk_distribution(0,0.8,accurate=True)
# f = open("datasets/cleaned_datasets_new/stationary_dist_epinions_RWRJ.txt", "w")
# f.write(str(graph_soc_rj.stationary_dist))
# f.write("\n")
# f.write("Alpha: {}".format(alpha))

graph_twitter_weak_rj = create_graph_cleaned_file('datasets/cleaned_datasets_new/twitter-weak_cln.txt', 'rwrj')
alpha = 0.8
graph_twitter_weak_rj.load_walk_distribution(0,0.8,accurate=True)
f = open("datasets/cleaned_datasets_new/stationary_dist_twitter-weak_RWRJ.txt", "w")
f.write(str(graph_twitter_weak_rj.stationary_dist))
f.write("\n")
f.write("Alpha: {}".format(alpha))