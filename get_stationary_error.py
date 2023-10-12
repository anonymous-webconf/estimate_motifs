from load_graphs import create_graph_cleaned_file
from class_graphs import Graph_RWRJ, Graph_RWBB
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
import random
import numpy as np
from typing import Union

epinions_rwrj = create_graph_cleaned_file("datasets/cleaned_datasets_new/epinions_cln.txt", "rwrj")
wiki_vote_rwrj = create_graph_cleaned_file("datasets/cleaned_datasets_new/wiki-Vote_cln.txt", "rwrj")

epinions_rwbb = create_graph_cleaned_file("datasets/cleaned_datasets_new/epinions_cln.txt", "rwbb")
wiki_vote_rwbb = create_graph_cleaned_file("datasets/cleaned_datasets_new/wiki-Vote_cln.txt", "rwbb")

alpha_default = 0.8

def get_actual_dist (file: str, is_rwrj = True):
    f = open(file, 'r')
    dist=  eval(f.readline())
    if is_rwrj:
        alpha = float(f.readline().split(" ")[1])
    else:
        alpha = alpha_default
    return dist, alpha

# error is calculated as sqrt(sum(psi(i) -psi_actual(i))^2)/n

def prob_from_num (num_dict: dict):
    tot_itr = sum(num_dict.values())
    min_val = min((num_dict.values()))
    prob = num_dict.copy()
    for i in num_dict.keys():
        if num_dict[i] == 0:
            prob[i] = min_val
            tot_itr += min_val
    for i in prob.keys():
        prob[i] = prob[i]/tot_itr
    # assert sum(prob.values()) >= 0.98 and sum(prob.values()) <= 1.02
    return prob

def update_num_dict (walk:list, num_dict: dict):
    for i in walk:
        num_dict[i] += 1
    return num_dict

def calculate_error(actual_dist: dict, prob_dist: dict):
    error = 0
    for i in actual_dist.keys():
        # error += (abs(actual_dist[i] - prob_dist[i])/actual_dist[i])
        error += abs(actual_dist[i] - prob_dist[i])
    error = (error)
    return error

def get_stationary_error(graph: Union[Graph_RWRJ, Graph_RWBB], actual_dist: dict, start:int, step : int,
                         total_len: int, alpha: float, is_rwrj=True):
    assert len(actual_dist) == graph.V
    if is_rwrj:
        large_walk = graph.random_walk_rwrj(random.choice(graph.valid_vertices), total_len, alpha)
    else:
        large_walk = graph.random_walk_rwbb(random.choice(graph.valid_vertices), total_len, alpha)
    num_dict = {}
    for i in graph.valid_vertices:
        num_dict[i] = 0
    r = range(start,total_len, step)
    assert max(r) <= len(large_walk)

    error_list = []
    start_ind = 0
    for i in range(0, len(r)):
        end_ind = r[i]
        print("Complete for iterations: {}".format(end_ind))
        print(len(large_walk[start_ind : end_ind]))
        num_dict = update_num_dict(large_walk[start_ind : end_ind], num_dict)
        start_ind = end_ind
        prob = prob_from_num(num_dict)
        e = calculate_error(actual_dist=actual_dist, prob_dist=prob)
        error_list.append(e)
    # for i in graph.valid_vertices:
    #     error = max(error, abs(actual_dist[i] - graph.stationary_dist[i])/actual_dist[i])
    return r, error_list

# r = [i for i in range(100, 30000, 1000)]

actual_epi_rwrj, alpha_epi_rwrj = get_actual_dist("datasets/cleaned_datasets_new/stationary_dist_epinions_RWRJ.txt")
actual_wiki_rwrj, alpha_wiki_rwrj = get_actual_dist("datasets/cleaned_datasets_new/stationary_dist_wiki-vote_RWRJ.txt")

actual_epi_rwbb, alpha_epi_rwbb = get_actual_dist("datasets/cleaned_datasets_new/stationary_dist_epinions_RWBB.txt", False)
actual_wiki_rwbb, alpha_wiki_rwbb = get_actual_dist("datasets/cleaned_datasets_new/stationary_dist_wiki-vote_RWBB.txt", False)

# error_epi = [get_stationary_error(epinions_rwrj, actual_epi, i, alpha_epi) for i in r]
# error_wiki= [get_stationary_error(wiki_vote_rwrj, actual_wiki, i, alpha_wiki) for i in r]

# r_ = range(min(r),max(r)+1,20)

# error_epi_smooth = (make_interp_spline(r,error_epi))(r_)
# error_wiki_smooth = make_interp_spline(r,error_wiki)(r_)

r_epi_rwrj, error_epi_rwrj = get_stationary_error(epinions_rwrj, actual_epi_rwrj, 1000, 2000, 100000, alpha_epi_rwrj)
r_wiki_rwrj, error_wiki_rwrj = get_stationary_error(wiki_vote_rwrj, actual_wiki_rwrj, 1000, 2000, 100000, alpha_wiki_rwrj)

f = plt.figure(figsize=(4,3))
plt.plot(r_epi_rwrj, error_epi_rwrj, label = "Epinions", marker=".")
plt.plot(r_wiki_rwrj, error_wiki_rwrj, label = "Wiki-Vote", marker=".")
# plt.plot(r_, error_wiki_smooth, label = "Wiki-Vote")
# plt.plot(r_, error_smooth, label = "smooth")
# plt.title("Estimating Stationary Distribution for RWBB")
plt.ylabel("Error "+ r"$\sum_i | \widehat{\pi_i} - \pi_i|$")
plt.xlabel("Steps "+r"$(10^4)$")
xlim_min, xlim_max = plt.xlim(1000,100000)
plt.xticks(ticks=list(np.arange(xlim_min, xlim_max+10000, 10000)), labels=np.arange(xlim_min/10000, xlim_max/10000+1, 1).round(2).tolist() )
plt.grid(True)
plt.margins(0)
plt.legend(loc="upper right", fontsize=9)
plt.savefig("stationary_error_RWRJ.png", bbox_inches='tight', pad_inches=0)

r_epi_rwbb, error_epi_rwbb = get_stationary_error(epinions_rwbb, actual_epi_rwbb, 1000, 2000, 100000, alpha_epi_rwbb, False)
r_wiki_rwbb, error_wiki_rwbb = get_stationary_error(wiki_vote_rwbb, actual_wiki_rwbb, 1000, 2000, 100000, alpha_wiki_rwbb, False)

f = plt.figure(figsize=(4,3))
plt.plot(r_epi_rwbb, error_epi_rwbb, label = "Epinions", marker=".")
plt.plot(r_wiki_rwbb, error_wiki_rwbb, label = "Wiki-Vote", marker=".")
# plt.plot(r_, error_wiki_smooth, label = "Wiki-Vote")
# plt.plot(r_, error_smooth, label = "smooth")
# plt.title("Estimating Stationary Distribution for RWBB")
plt.ylabel("Error "+ r"$\sum_i | \widehat{\pi_i} - \pi_i|$")
plt.xlabel("Steps "+r"$(10^4)$")
xlim_min, xlim_max = plt.xlim(1000,100000)
plt.xticks(ticks=list(np.arange(xlim_min, xlim_max+10000, 10000)), labels=np.arange(xlim_min/10000, xlim_max/10000+1, 1).round(2).tolist() )
plt.grid(True)
plt.margins(0)
plt.legend(loc="upper right", fontsize=9)
plt.savefig("stationary_error_RWBB.png", bbox_inches='tight', pad_inches=0)


    