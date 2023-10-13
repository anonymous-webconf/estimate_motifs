import random
import numpy as np
from matplotlib import pyplot as plt
from itertools import permutations, combinations
import pynauty as pn
import threading

from typing import Union

from get_stationary import get_stationary_distribution
from math import ceil

cyclic_C3_adj_mat = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
# cyclic_C3_adj_mat = [[0,1,0], [0,0,1], [1,1,0]] # this is cyclic (with additional edge)
acyclic_C3_adj_mat = [[0, 1, 0], [0, 0, 0], [1, 1, 0]]
cyclic_C4_adj_mat = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
source_S3_adj_mat = [[0, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
butterfly_adj_mat = [[0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0]]
leaf_C3_adj_mat = [[0, 1, 1, 0], [0, 0, 0 ,0], [0, 1, 0, 1], [0, 0, 0, 0]] # G24
clique_K4_mat = [[0, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 1], [1, 0, 0, 0]]
wedge_adj_mat = [[0,1,1], [0,0,0], [0,0,0]]

'''
****************************************************************************************************
Graph class RW-IS begins here
****************************************************************************************************
'''

class Graph_RWIS():
    def __init__(self, V: int):
        self.V = V
        self.in_neigh = [[] for _ in range(V)]
        self.out_neigh = [[] for _ in range(V)]
        self.in_deg = [0 for _ in range(V)]
        self.out_deg = [0 for _ in range(V)]
        self.edges = {}
        # vertices in the largest Connected Component of graph
        self.valid_vertices = []
        self.in_neigh_valid = {}
        self.out_neigh_valid = {}
        self.in_deg_valid = {}
        self.out_deg_valid = {}

        self.num_edges_in_G2 = 0
        self.is_clean = False

    def add_edge(self, u: int, v: int):  # add edge u -> v
        # u is in neighbor of v
        # v is out neighbor of u
        self.in_neigh[v].append(u)
        self.out_neigh[u].append(v)
        self.in_deg[v] += 1
        self.out_deg[u] += 1
        self.edges[(u, v)] = True

    def is_edge(self, u: int, v: int):
        # is there an edge u -> v
        return (u, v) in self.edges

    def tot_deg(self, v: int, d=1):
        if d == 2:
            return len(set(self.in_neigh_valid[v] + self.out_neigh_valid[v]))
        return self.in_deg_valid[v] + self.out_deg_valid[v]

    def random_vertex(self):
        return random.choice(self.valid_vertices)

    def random_out_neigh(self, v: int):
        return random.sample(self.out_neigh_valid[v])

    def random_in_neigh(self, v: int):
        return random.sample(self.in_neigh_valid[v])

    def BFS(self, v: int):
        q = []
        q.append(v)
        visited = [False]*self.V
        visited[v] = True
        size = 1
        while(len(q) >= 1):
            # print(q)
            node = q.pop()
            neigh_lst = list(set(self.out_neigh[node] + self.in_neigh[node]))
            # print(node, neigh_lst)
            assert len(set(neigh_lst)) == len(neigh_lst)
            for neigh in neigh_lst:
                if visited[neigh] == False:
                    size += 1
                    q.append(neigh)
                    visited[neigh] = True
            # if size >= 0.5*self.V:
            #       print("broken because got a size match!")
            #       break
        return size, visited

    def clean_graph(self, largest_SCC):
        self.valid_vertices = largest_SCC.copy()
        # assert ((u,v) not in self.edges for (v,u) in self.edges)
        assert self.valid_vertices != []
        # print("start vertex is: ", self.start_vertex)
        # print("max size is: ", max_size)
        self.clean_neighbours()

    def clean_neighbours(self):
        for node in self.valid_vertices:
            self.in_neigh_valid[node] = list(
                set(self.in_neigh[node]) & set(self.valid_vertices))
            self.out_neigh_valid[node] = list(
                set(self.out_neigh[node]) & set(self.valid_vertices))
            self.in_deg_valid[node] = len(self.in_neigh_valid[node])
            self.out_deg_valid[node] = len(self.out_neigh_valid[node])
        self.num_edges_in_G2 = self.calculate_G2_edges()
        self.is_clean = True

    def get_neighbour(self, v: int):
        if self.in_deg[v]==0:
          return random.choice(self.out_neigh[v])
        elif self.out_deg[v]==0:
          return random.choice(self.in_neigh[v])
        din_dout = self.in_deg[v] + self.out_deg[v]
        # choose either the in neighbour or out neighbour
        prob_out = self.out_deg[v]/din_dout
        prob_in = self.in_deg[v]/din_dout
        choice = random.choices( ['in_n', 'out_n'], weights= [prob_in, prob_out], k=1)[0]
        if choice == 'in_n':
              return random.choice(self.in_neigh[v])
        else:
              return random.choice(self.out_neigh[v])
        return np.random.choice(sampled_vertices,1,p=[sample_prob_out, sample_prob_in])[0]
        # vertex = random.choice(list(set(self.in_neigh[v]+self.out_neigh[v])))
        # return vertex

    def edge_degree(self, e: tuple):
        u, v = e[0], e[1]
        return self.tot_deg(u, d=2) + self.tot_deg(v, d=2) - 2

    def calculate_G2_edges(self):
        if self.is_clean == False:
            ct = 0
            for v in self.valid_vertices:
                ct += self.tot_deg(v, d=2)*(self.tot_deg(v, d=2) - 1)/2
            return ct
        return self.num_edges_in_G2

    def get_edge_neighbour(self, e: tuple):
        # edge u -> v
        assert e in self.edges
        u, v = e[0], e[1]
        # choose between u or v with prob du/dv+du
        du_minus_one = self.tot_deg(u, d=2) - 1
        dv_minus_one = self.tot_deg(v, d=2) - 1
        du_dv_minus_two = du_minus_one + dv_minus_one

        vertex = random.choices([u, v], weights=[
                                du_minus_one/du_dv_minus_two, dv_minus_one/du_dv_minus_two], k=1)[0]
        # for the vertex, choose a neighbor
        # neigh = self.get_neighbour(vertex)
        # while (((vertex, neigh) == e) or ((neigh, vertex)==e)):
        #       neigh = self.get_neighbour(vertex)

        # if (vertex, neigh) in self.edges:
        #       return (vertex, neigh)
        # else:
        #       return (neigh, vertex)
        if vertex == u:
            # edge is u -> v
            u_neigh_lst = [i for i in set(
                self.in_neigh_valid[u] + self.out_neigh_valid[u]) if i != v]
            assert len(u_neigh_lst) == self.tot_deg(u, d=2)-1
            assert v not in u_neigh_lst
            v1 = random.choice(u_neigh_lst)
            if (u, v1) in self.edges:
                return (u, v1)
            else:
                return (v1, u)
        else:
            # edge is u -> v
            v_neigh_lst = [i for i in set(
                self.in_neigh_valid[v] + self.out_neigh_valid[v]) if i != u]
            assert len(v_neigh_lst) == self.tot_deg(v, d=2)-1
            assert u not in v_neigh_lst
            v2 = random.choice(v_neigh_lst)
            if (v, v2) in self.edges:
                return (v, v2)
            else:
                assert (v2, v) in self.edges
                return (v2, v)

    def get_adj_mat(self, states):
        adj_mat = []
        for vertex in states:
            temp = []
            for neighbor in states:
                temp.append(int(self.is_edge(vertex, neighbor)))
            adj_mat.append(temp)
        return adj_mat

    def state_inc_prob(self, state, d=1):
        if d != 1:
            # check if we can walk:
            curr_e = state[0]
            for e in state[1:]:
                next_e = e
                if len(set(curr_e) & set(next_e)) == 1:
                    curr_e = next_e
                    continue
                else:
                    return 0
            num_edges_G2 = self.num_edges_in_G2
            e1 = state[0]
            prob = self.edge_degree(e1)/(2*num_edges_G2)
            for e in state[:-1]:
                prob = prob/(self.edge_degree(e))
            return prob
        # first check if the walk in possible:

        num_edges = len(self.edges.keys())
        v1 = state[0]
        prob = self.tot_deg(v1)/(2*num_edges)
        for i in range(len(state)-1):
            curr = state[i]
            next = state[i+1]
            if self.is_edge(curr, next) and self.is_edge(next, curr):
                prob = prob*2/(self.tot_deg(curr))
            elif self.is_edge(curr, next) == False and self.is_edge(next, curr) == False:
                return 0
            else:
                prob = prob/(self.tot_deg(curr))
        return prob

    def motif_inc_prob_G1(self, states):
        prob = 0
        states = sorted(states)
        state_permutations = permutations(states)
        for permute in state_permutations:
            # print(permute)
            # for a given permutation:
            curr_prob = self.state_inc_prob(permute)
            prob += curr_prob
        #   print(permute, curr_prob)
        return prob

    def valid_walk_G2(self, walk):
        # walk is a collection of edges
        for i in range(len(walk)-1):
            if len(self.get_vertex_set([walk[i], walk[i+1]])) != 3:
                return False
        return True

    def motif_inc_prob(self, states, d):
        assert d in [1, 2]
        if d == 1:
            assert type(states[0]) == int
            return self.motif_inc_prob_G1(states)
        # note that every i in states is an edge
        prob = 0
        l = len(states)
        k = l+1
        assert type(states[0]) == tuple

        # get all edges of the graph in states
        vertex_list = list(self.get_vertex_set(states))
        edges_in_motif = []
        for u in vertex_list:
            for v in vertex_list:
                if self.is_edge(u, v) and (v, u) not in edges_in_motif:
                    edges_in_motif.append((u, v))
        # get a combination of l edges from edges_in_motif:
        # note that select is a list of edges
        for select in combinations(edges_in_motif, l):
            if len(self.get_vertex_set(select)) == k:
                # permute is a permutation of list of edges
                for permute in permutations(select):
                    # check if the walk is valid for permute
                    if self.valid_walk_G2(permute):
                        # calculate the state prob
                        prob += self.state_inc_prob(permute, d=2)
        return prob

    def random_walk_on_Gd(self, seed, walk_length: int, d: int):
        if self.is_clean == False:
            self.clean_graph()
        assert self.is_clean
        assert d in [1, 2]
        assert (d == 2 and type(seed) == tuple) or (
            d == 1 and type(seed) == int)
        method_list = [-1, self.get_neighbour, self.get_edge_neighbour]
        random_method = method_list[d]
        walk = [seed]
        for _ in range(walk_length-1):
            walk.append(random_method(walk[-1]))
        return walk
    
    def get_k_states_from_walk(self, walk: list, k:int, d:int):
        state_list = []
        start_ind = 0
        l = k-d+1
        last_ind = l
        seed = walk[0]
        assert d in [1, 2]
        assert (d == 2 and type(seed) == tuple) or (
            d == 1 and type(seed) == int)
        while (last_ind - start_ind == l):
            state_list.append(walk[start_ind: last_ind])
            start_ind = start_ind + 1
            last_ind = min(last_ind + 1, len(walk))
        return state_list

    def get_vertex_set(self, list_state):
        if type(list_state[0]) == int:
            return set(list_state)
        vertex_set = []
        for e in list_state:
            vertex_set.append(e[0])
            vertex_set.append(e[1])
        return set(vertex_set)

    def rw_count_motif_Gd(self, seed, walk_len: int, d: int, motif_adj_mat: list, k: int):
        graph = self
        assert len(motif_adj_mat) == len(motif_adj_mat[0])
        assert k == len(motif_adj_mat)
        walk = graph.random_walk_on_Gd(seed, walk_len, d)
        state_list = graph.get_k_states_from_walk(walk, k, d)
        # print(len(state_list))
        count = 0
        ls = []
        for state in state_list:
            vertex_set = graph.get_vertex_set(state)
            if len(vertex_set) == k:
                if are_isomorphic(graph.get_adj_mat(vertex_set), motif_adj_mat):
                    ls.append(state)
                    count += 1/graph.motif_inc_prob(state, d)
        # print(ls)
        return count/len(state_list)

    def plot_error_motif(self, list_walk_list: list, step_size: int, 
                         motif_adj_mat: list, k: int, actual_count: int, d=1):
        graph = self
        if graph.is_clean == False:
            graph.clean_graph()
        assert graph.is_clean
        assert len(motif_adj_mat) == len(motif_adj_mat[0])
        assert k == len(motif_adj_mat)
        iterations = len(list_walk_list)
        assert d in [1,2]
        seed = list_walk_list[0][0]
        assert (d == 2 and type(seed) == tuple) or (
            d == 1 and type(seed) == int)
        list_state_list = []

        def add_walk_to_list(walk):
            list_state_list.append(
                graph.get_k_states_from_walk(walk=walk, k=k, d=d))
                
        for i in range(iterations):
            #      print("Adding walk for itr: ", i+1)
            add_walk_to_list(list_walk_list[i])

        assert (len(list_state_list[i]) == len(list_state_list[i+1]) for i in range(iterations-1))
        r = []
        error_list = []
        start_ind = 0
        end_ind = step_size
        state_len = len(list_state_list[0])
        count = [0]*iterations

        while (end_ind <= state_len + step_size):
            for itr in range(iterations):
                state_list = list_state_list[itr]
                for state in state_list[start_ind: end_ind]:
                    vertex_set = graph.get_vertex_set(state)
                    if len(vertex_set) == k and are_isomorphic(graph.get_adj_mat(vertex_set), motif_adj_mat):
                        count[itr] += 1/graph.motif_inc_prob(state, d)
            # using the count for current end_ind, we calculate the error
            walk_len_curr = min(end_ind, state_len)
            r.append(walk_len_curr)
            curr_error = (sum([(i/walk_len_curr - actual_count)
                          ** 2 for i in count])/iterations)**0.5/actual_count
            error_list.append(curr_error)
            start_ind = end_ind
            end_ind = end_ind + step_size
        # print(count)
        return r, error_list


'''
****************************************************************************************************
Graph class RW-IS ends here
****************************************************************************************************
'''

'''
****************************************************************************************************
Graph class RW-RJ begins here
****************************************************************************************************
'''


class Graph_RWRJ():
    def __init__(self, V: int):
        self.V = V
        self.in_neigh = [[] for _ in range(V)]
        self.out_neigh = [[] for _ in range(V)]
        self.in_deg = [0 for _ in range(V)]
        self.out_deg = [0 for _ in range(V)]
        self.edges = {}
        self.Time = 0
        # vertices in the largest Connected Component of graph
        self.valid_vertices = []
        self.in_neigh_valid = {}
        self.out_neigh_valid = {}
        self.in_deg_valid = {}
        self.out_deg_valid = {}
        self.is_clean = False
        self.stationary_dist = {}

    def add_edge(self, u: int, v: int):  # add edge u -> v
        # u is in neighbor of v
        # v is out neighbor of u
        self.in_neigh[v].append(u)
        self.out_neigh[u].append(v)
        self.in_deg[v] += 1
        self.out_deg[u] += 1
        self.edges[(u, v)] = True

    def is_edge(self, u: int, v: int):
        # is there an edge u -> v
        return (u, v) in self.edges

    def tot_deg(self, v: int):
        return self.in_deg_valid[v] + self.out_deg_valid[v]

    def random_vertex(self):
        return random.choice(self.valid_vertices)

    def random_out_neigh(self, v: int):
        return random.sample(self.out_neigh_valid[v])

    def random_in_neigh(self, v: int):
        return random.sample(self.in_neigh_valid[v])

    def BFS(self, v: int):
        q = []
        q.append(v)
        visited = [False]*self.V
        visited[v] = True
        size = 1
        while(len(q) >= 1):
            # print(q)
            node = q.pop()
            neigh_lst = list(set(self.out_neigh[node] + self.in_neigh[node]))
            # print(node, neigh_lst)
            assert len(set(neigh_lst)) == len(neigh_lst)
            for neigh in neigh_lst:
                if visited[neigh] == False:
                    size += 1
                    q.append(neigh)
                    visited[neigh] = True
            # if size >= 0.5*self.V:
            #       print("broken because got a size match!")
            #       break
        return size, visited

    def clean_graph(self, largest_SCC):
        self.valid_vertices = largest_SCC.copy()
        # assert ((u,v) not in self.edges for (v,u) in self.edges)
        assert self.valid_vertices != []
        # print("start vertex is: ", self.start_vertex)
        # print("max size is: ", max_size)
        self.clean_neighbours()

    def clean_neighbours(self):
        for node in self.valid_vertices:
            self.in_neigh_valid[node] = list(
                set(self.in_neigh[node]) & set(self.valid_vertices))
            self.out_neigh_valid[node] = list(
                set(self.out_neigh[node]) & set(self.valid_vertices))
            self.in_deg_valid[node] = len(self.in_neigh_valid[node])
            self.out_deg_valid[node] = len(self.out_neigh_valid[node])
        self.is_clean = True

    def transition_kernel_sample(self, v: int, alpha: float):
        assert alpha <= 1
        if self.out_deg_valid[v] == 0:
            # go to a random vertex in graph
            return random.choice(self.valid_vertices)
        # else with alpha go to out-neigh. with 1-alpha go to some random vertex
        options = ['out_neigh', 'random']
        choice = random.choices(options, weights=[alpha, 1-alpha])[0]
        if choice == 'out_neigh':
            return random.choice(self.out_neigh_valid[v])
        elif choice == 'random':
            return random.choice(self.valid_vertices)

    '''gives the value for q(x_t | y_t) :: prob of transition from y_t -> x_t'''

    def transition_kernel(self, x_t: int, y_t: int, alpha: float):
        assert alpha <= 1 and alpha >= 0
        if self.out_deg_valid[y_t] == 0:
            return 1/len(self.valid_vertices)
        if self.is_edge(y_t, x_t):
            return (alpha/self.out_deg_valid[y_t]) + ((1-alpha)/len(self.valid_vertices))
        else:
            return (1-alpha)/len(self.valid_vertices)

    def get_neighbour(self, v: int, alpha: float):
        # x_t = v
        # y_t = self.transition_kernel_sample(x_t, alpha=alpha)
        # pi_yt = self.tot_deg(y_t)/(2*len(self.edges))
        # pi_xt = self.tot_deg(x_t)/(2*len(self.edges))
        # accept_prob = min((pi_yt*self.transition_kernel(x_t, y_t, alpha))/(pi_xt*self.transition_kernel(y_t, x_t, alpha)) , 1)
        # ans = random.choices([y_t, x_t], weights=[accept_prob, 1-accept_prob])[0]
        # return ans
        return self.transition_kernel_sample(v, alpha=alpha)

    def get_adj_mat(self, states):
        adj_mat = []
        for vertex in states:
            temp = []
            for neighbor in states:
                temp.append(int(self.is_edge(vertex, neighbor)))
            adj_mat.append(temp)
        return adj_mat

    def state_inc_prob(self, state: list, alpha: float):
        v1 = state[0]
        # prob = self.tot_deg(v1)/(2*num_edges)
        prob = self.stationary_dist[v1]
        for i in range(len(state)-1):
            v = state[i]
            next = state[i+1]
            prob = prob * self.transition_kernel(next, v, alpha)
        return prob

    def motif_inc_prob(self, states: list, alpha: float):
        assert len(self.stationary_dist.keys()) == len(self.valid_vertices)
        prob = 0
        states = sorted(states)
        state_permutations = permutations(states)
        for permute in state_permutations:
            # print(permute)
            # for a given permutation:
            prob += self.state_inc_prob(permute, alpha=alpha)
        assert prob > 0
        return prob

    def lmix_random_walk(self, seed: int, lmix: int, alpha: float):
        vertex = seed
        for _ in range(lmix):
            vertex = self.get_neighbour(vertex, alpha)
        return vertex

    def load_walk_distribution(self, iterations: int, alpha: float, file=None, accurate=True):
        if file is not None:
            f = open(file, "r")
            dict_ = eval(f.readline())
            assert len(dict_.keys()) == self.V
            self.stationary_dist = dict_
            f.close()
            assert min(self.stationary_dist.values()) > 0
            print("distribution loaded from file successfully for RWRJ")
            return
        # Using the matrix function
        if accurate:
            P = []
            print("All calculations are for alpha: ", alpha)
            print("Creating Transition Matrix")
            for u in self.valid_vertices:
                l = []
                for v in self.valid_vertices:
                    jumping_prob = self.transition_kernel(v, u, alpha)
                    l.append(jumping_prob)
                P.append(l)
            print("Transition Matrix Created")
            print("Getting Stationary Distribution")
            stationary = get_stationary_distribution(P)
            print("Stationary Distribution Received")
            assert list(range(self.V)) == self.valid_vertices
            for i in self.valid_vertices:
                self.stationary_dist[i] = stationary[i]
            print("Stationary Distribution Loaded")
            return
        # # In this code, we will plot the change in prob ditsribution
        # # Intially the change is INF. Then the change is computed as : normalised avg change
        # # \sum_i (|p(v)_{i-1} - p(v)_i|/p(v)_{i-1})/n
        '''
        else:
        '''
        # # num stores how many times a vertex has been encountered
        print("estimating stationary distribution")
        num = {}
        for v in self.valid_vertices:
            num[v] = 0
        min_ = self.V/(1-alpha) + 10000000
        vertex = self.valid_vertices[0]
        '''
        last_print = -1
        for itr in range(1, iterations+1):
            if int(itr *100/(iterations)) % 40 == 0:
                  if int(itr *100/(iterations)) != last_print:
                    print("{}% iterations done".format(int(itr *100/(iterations))))
                    last_print = int(itr *100/(iterations))
            vertex = self.lmix_random_walk(seed = vertex, lmix = lmix, alpha= alpha)
            num[vertex] = num[vertex] + 1
            min_ = min(num[vertex], min_)
        '''
        long_walk = self.random_walk_rwrj(vertex, iterations, alpha)
        for v in long_walk:
            num[v] = num[v]+1
            min_ = min(num[v], min_)
        for v in num.keys():
            if num[v] == 0:
                num[v] = min_
                iterations += min_
        assert sum(num.values()) == iterations
        for v in num.keys():
            self.stationary_dist[v] = num[v]/iterations
        assert sum(self.stationary_dist.values()) <= 1.05 and sum(
            self.stationary_dist.values()) >= 0.95
        # print(sum(self.stationary_dist.values()))
        print("Stationary distribbution loaded for alpha: {}".format(alpha))

    def random_walk_rwrj(self, seed, walk_length: int, alpha: float):
        # if self.is_clean == False:
        #     self.clean_graph()
        assert self.is_clean
        walk = [seed]
        for _ in range(walk_length-1):
            walk.append(self.get_neighbour(walk[-1], alpha=alpha))
        return walk
    
    def get_k_state_list(self, walk: list, k: int):
        l = k
        state_list = []
        start_ind = 0
        last_ind = l
        while (last_ind - start_ind == l):
            state_list.append(walk[start_ind: last_ind])
            start_ind = start_ind + 1
            last_ind = min(last_ind + 1, len(walk))
        return state_list

    def get_vertex_set(self, list_state):
        if type(list_state[0]) == int:
            return set(list_state)
        vertex_set = []
        for e in list_state:
            vertex_set.append(e[0])
            vertex_set.append(e[1])
        return set(vertex_set)

    def rw_count_motif_rwrj(self, seed, walk_len: int, motif_adj_mat: list, k: int, alpha: int):
        graph = self
        assert graph.is_clean
        if len(graph.stationary_dist.keys()) != graph.V:
            graph.load_walk_distribution(
                lmix=graph.V/10, iterations=200000, alpha=alpha)
        assert len(motif_adj_mat) == len(motif_adj_mat[0])
        assert k == len(motif_adj_mat)
        state_list = graph.random_walk_rwrj(
            seed, walk_length=walk_len, k=k, alpha=alpha)
        count = 0
        for state in state_list:
            vertex_set = graph.get_vertex_set(state)
            if len(vertex_set) == k:
                if are_isomorphic(graph.get_adj_mat(vertex_set), motif_adj_mat):
                    state_prob = graph.motif_inc_prob(state, alpha=alpha)
                    # print(state)
                    # print("found motif {} with probability {}".format(state, state_prob))
                    count += 1/state_prob
        return count/len(state_list)

    def plot_error_motif(self, list_walk_list : list, step_size: int, 
                                   motif_adj_mat: list, k: int, actual_count: int, alpha: float):
        graph = self
        if graph.is_clean == False:
            graph.clean_graph()
        assert graph.is_clean
        assert len(motif_adj_mat) == len(motif_adj_mat[0])
        assert k == len(motif_adj_mat)
        assert len(graph.stationary_dist.keys()) == len(graph.valid_vertices)
        iterations = len(list_walk_list)
        list_state_list = []

        def add_walk_to_list(walk):
            list_state_list.append(self.get_k_state_list(walk, k))

        for i in range(iterations):
            # print("Adding walk for itr: ", _+1)
            add_walk_to_list(list_walk_list[i])
        assert (len(list_state_list[i]) == len(
            list_state_list[i+1]) for i in range(iterations-1))
        r = []
        error_list = []
        start_ind = 0
        end_ind = step_size
        state_len = len(list_state_list[0])
        count = [0]*iterations
        while (end_ind <= state_len + step_size):
            # now we get counts across all lists interation
            for itr in range(iterations):
                state_list = list_state_list[itr]
                for state in state_list[start_ind: end_ind]:
                    vertex_set = graph.get_vertex_set(state)
                    if len(vertex_set) == k and are_isomorphic(graph.get_adj_mat(vertex_set), motif_adj_mat):
                        count[itr] += 1 / graph.motif_inc_prob(state, alpha=alpha)
            # using the count for current end_ind, we calculate the error
            walk_len_curr = min(end_ind, state_len)
            r.append(walk_len_curr)
            curr_error = (sum([(i/walk_len_curr - actual_count)
                               ** 2 for i in count])/iterations)**0.5/actual_count
            error_list.append(curr_error)
            start_ind = end_ind
            end_ind = end_ind + step_size
        return r, error_list


'''
****************************************************************************************************
Graph class RW-RJ ends here
****************************************************************************************************
'''

'''
****************************************************************************************************
Graph class RW-BB begins here
****************************************************************************************************
'''
class Graph_RWBB():
    def __init__(self, V: int):
        self.V = V
        self.in_neigh = [[] for _ in range(V)]
        self.out_neigh = [[] for _ in range(V)]
        self.in_deg = [0 for _ in range(V)]
        self.out_deg = [0 for _ in range(V)]
        self.edges = {}
        # vertices in the largest Connected Component of graph
        self.valid_vertices = []
        self.in_neigh_valid = {}
        self.out_neigh_valid = {}
        self.in_deg_valid = {}
        self.out_deg_valid = {}
        self.is_clean = False
        self.stationary_dist = {}
        self.history_stack = []

    def add_edge(self, u: int, v: int):  # add edge u -> v
        # u is in neighbor of v
        # v is out neighbor of u
        self.in_neigh[v].append(u)
        self.out_neigh[u].append(v)
        self.in_deg[v] += 1
        self.out_deg[u] += 1
        self.edges[(u, v)] = True

    def is_edge(self, u: int, v: int):
        # is there an edge u -> v
        return (u, v) in self.edges

    def tot_deg(self, v: int):
        return self.in_deg_valid[v] + self.out_deg_valid[v]

    def random_vertex(self):
        return random.choice(self.valid_vertices)

    def random_out_neigh(self, v: int):
        return random.sample(self.out_neigh_valid[v])

    def random_in_neigh(self, v: int):
        return random.sample(self.in_neigh_valid[v])

    def BFS(self, v: int):
        q = []
        q.append(v)
        visited = [False]*self.V
        visited[v] = True
        size = 1
        while(len(q) >= 1):
            # print(q)
            node = q.pop()
            neigh_lst = self.out_neigh[node]
            # print(node, neigh_lst)
            assert len(set(neigh_lst)) == len(neigh_lst)
            for neigh in neigh_lst:
                if visited[neigh] == False:
                    size += 1
                    q.append(neigh)
                    visited[neigh] = True
            # if size >= 0.5*self.V:
            #       print("broken because got a size match!")
            #       break
        return size, visited

    def clean_graph(self, largest_SCC):
        self.valid_vertices = largest_SCC.copy()
        # assert ((u,v) not in self.edges for (v,u) in self.edges)
        assert self.valid_vertices != []
        # print("start vertex is: ", self.start_vertex)
        # print("max size is: ", max_size)
        self.clean_neighbours()

    def clean_neighbours(self):
        for node in self.valid_vertices:
            self.in_neigh_valid[node] = list(
                set(self.in_neigh[node]) & set(self.valid_vertices))
            self.out_neigh_valid[node] = list(
                set(self.out_neigh[node]) & set(self.valid_vertices))
            try:
                assert set(self.out_neigh_valid[node]) == set(
                    self.out_neigh[node])
            except:
                print(node)
                print(self.out_neigh_valid[node])
                print(self.out_neigh[node])
            self.in_deg_valid[node] = len(self.in_neigh_valid[node])
            self.out_deg_valid[node] = len(self.out_neigh_valid[node])
        self.is_clean = True

    def get_neighbour(self, v: int, alpha: float):
        # with prob alpha go ahead or pop with prob (1-alpha)
        if self.out_deg_valid[v] == 0:
            return self.history_stack.pop()
        if len(self.history_stack) == 0:
            self.history_stack.append(v)
            return random.choice(self.out_neigh_valid[v])
        choice = random.choices(['ahead', 'pop'], weights=[
                                alpha, 1-alpha], k=1)[0]
        if choice == 'ahead':
            self.history_stack.append(v)
            return random.choice(self.out_neigh_valid[v])
        elif choice == 'pop':
            return self.history_stack.pop()
        print("RETURNING NONEE")
        return None

    def get_adj_mat(self, states):
        adj_mat = []
        for vertex in states:
            temp = []
            for neighbor in states:
                temp.append(int(self.is_edge(vertex, neighbor)))
            adj_mat.append(temp)
        return adj_mat
    
    '''
    prob of transition from u --> v
    '''
    def transition_kernel(self, u,v):
        prob = 0
        if self.is_edge(u,v):
            prob = 1/self.out_deg_valid[u]
        return prob
    
    ''' Based on the conjecture that you can switch walking direction atmost once
        and can only start with backward steps (if switching)
    '''
    def valid_walk(self, state: list):
        backward = True
        for i in range(1,len(state)):
            u,v = state[i-1], state[i]
            # if the new edge is backward and still in backward condition, go on
            if (u,v) not in self.edges and (v,u) not in self.edges:
                return False
            if (v,u) in self.edges and backward:
                continue
            # if the new edge is forward and last state is backward, change state
            if (u,v) in self.edges and backward:
                backward = False
                continue
            if (u,v) in self.edges and backward == False:
                continue
            # if the current edge is backward but the state is forward
            if (v,u) in self.edges and backward == False:
                # print("printing at code line 874 in class_graph.py",state)
                return False
        return True

    # def state_inc_prob(self, state: list, alpha: float):
    #     v1 = state[0]
    #     # prob = self.tot_deg(v1)/(2*num_edges)
    #     prob = self.stationary_dist[v1]
    #     for i in range(len(state)-1):
    #         v = state[i]
    #         next = state[i+1]
    #         # There are only 2 ways:
    #         # if there is a forward edge, with prob alpha it is taken
    #         # if there is a backward edge, with prob (1-alpha) it is taken
    #         if self.is_edge(v, next) and self.is_edge(next, v):
    #             print("RED ALERTTTT at like 889 - wasnt supposed to happen")
    #             prob = prob*((alpha / self.out_deg_valid[v]) + ((1-alpha)*(
    #                 alpha)*(self.stationary_dist[next])/self.out_deg_valid[next]))
    #         elif self.is_edge(v, next):
    #             prob = prob*(alpha / self.out_deg_valid[v])
    #         elif self.is_edge(next, v):
    #             # prob of landing at next, prob of going to v and back button to next
    #             assert self.stationary_dist[next] != 0
    #             # prob = prob * ((1-alpha)*(alpha)*(self.stationary_dist[next])/self.out_deg_valid[next])
    #             prob = prob * (1-alpha) * self.stationary_dist[v]
    #             # prob = prob * (1-alpha)
    #         else:
    #             prob = prob * 0
    #     return prob

    def state_inc_prob(self, state: list, alpha: float):
        # first see till what stage was the back button pushed
        ind = 0
        for i in range(len(state)-1):
            curr = state[i]
            next = state[i+1]
            if self.is_edge(next, curr):
                ind = i+1
            else:
                break
        # now that you know the index, go from that index to start and calculate the prob of that sequence
        # some back edges
        node = state[ind]
        prob = self.stationary_dist[node]
        for i in range(ind, 0, -1):
            # print("for chain: {} <--- {}".format(state[i-1], state[i]))
            curr = state[i]
            prob = prob * 1/self.out_deg_valid[curr] * (1-alpha) * (alpha)
        for i in range(ind, len(state)-1):
            # print("for chain: {} ----> {}".format(state[i], state[i+1]))
            curr = state[i]
            prob = prob * alpha * 1/self.out_deg_valid[curr]
        return prob

    def motif_inc_prob(self, states: list, alpha: float):
        prob = 0
        states = sorted(states)
        state_permutations = permutations(states)
        for permute in state_permutations:
            # print(permute)
            # for a given permutation:
            if self.valid_walk(permute):
                # print(permute)
                prob += self.state_inc_prob(permute, alpha=alpha)
            else:
                prob += 0
        if prob == 0:
            print(states)
            for state in states:
                print(self.out_deg_valid[state], self.stationary_dist[state])
        assert prob > 0
        return prob

    def lmix_random_walk(self, seed: int, lmix: int, alpha: float):
        while self.out_deg_valid[seed] == 0:
            seed = random.choice(self.valid_vertices)
        vertex = seed
        for _ in range(lmix):
            vertex = self.get_neighbour(vertex, alpha)
        return vertex

    def load_walk_distribution(self, iterations: int, alpha: float, file=None, accurate=True):
        if file is not None:
            f = open(file, "r")
            dict_ = eval(f.readline())
            assert len(dict_.keys()) == self.V
            self.stationary_dist = dict_
            f.close()
            assert min(self.stationary_dist.values()) > 0
            print("distribution loaded from file successfully for RWBB")
            return
        # Using the matrix function
        if accurate:
            P = []
            print("Creating Transition Matrix")
            for u in self.valid_vertices:
                l = []
                for v in self.valid_vertices:
                    jumping_prob = self.transition_kernel(u,v)
                    l.append(jumping_prob)
                P.append(l)
            print("Transition Matrix Created")
            print("Getting Stationary Distribution")
            stationary = get_stationary_distribution(P)
            print("Stationary Distribution Received")
            assert list(range(self.V)) == self.valid_vertices
            for i in self.valid_vertices:
                self.stationary_dist[i] = stationary[i]
            print("Stationary Distribution Loaded")
            return
        # # In this code, we will plot the change in prob ditsribution
        # # Intially the change is INF. Then the change is computed as : normalised avg change
        # # \sum_i (|p(v)_{i-1} - p(v)_i|/p(v)_{i-1})/n
        '''
        else:
        '''
        # # num stores how many times a vertex has been encountered
        print("estimating stationary distribution")
        num = {}
        for v in self.valid_vertices:
            num[v] = 0
        min_ = self.V/(alpha)**2 + 10000000
        vertex = self.valid_vertices[0]
 
        long_walk = self.random_walk_rwbb(vertex, iterations, alpha)
        for v in long_walk:
            num[v] = num[v]+1
            min_ = min(num[v], min_)
        for v in num.keys():
            if num[v] == 0:
                num[v] = min_
                iterations += min_
        assert sum(num.values()) == iterations
        for v in num.keys():
            self.stationary_dist[v] = num[v]/iterations
        assert sum(self.stationary_dist.values()) <= 1.05 and sum(
            self.stationary_dist.values()) >= 0.95
        # print(sum(self.stationary_dist.values()))
        print("Stationary distribbution loaded for alpha: {}".format(alpha))

    def random_walk_rwbb(self, seed, walk_length: int, alpha: float):
        if self.is_clean == False:
            self.clean_graph()
        assert self.is_clean
        walk = [seed]
        for _ in range(walk_length-1):
            walk.append(self.get_neighbour(walk[-1], alpha=alpha))
        return walk
    
    def get_k_state_list(self, walk: list, k: int):
        l = k
        state_list = []
        start_ind = 0
        last_ind = l
        while (last_ind - start_ind == l):
            state_list.append(walk[start_ind: last_ind])
            start_ind = start_ind + 1
            last_ind = min(last_ind + 1, len(walk))
        return state_list

    def get_vertex_set(self, list_state):
        if type(list_state[0]) == int:
            return set(list_state)
        vertex_set = []
        for e in list_state:
            vertex_set.append(e[0])
            vertex_set.append(e[1])
        return set(vertex_set)

    def rw_count_motif_back_button(self, seed, walk_len: int, motif_adj_mat: list, k: int, alpha: int):
        graph = self
        try:
            assert len(graph.stationary_dist.keys()) == len(graph.valid_vertices)
        except:
            print("Loading walk in method: rw_count_motif_back_button")
            graph.load_walk_distribution(
                lmix=graph.V//10, iterations=graph.V, alpha=alpha)
        assert len(motif_adj_mat) == len(motif_adj_mat[0])
        assert k == len(motif_adj_mat)
        walk = graph.random_walk_rwbb(seed, walk_len, alpha)
        # print(walk)
        state_list = graph.get_k_state_list(walk, k)
        # print(state_list)
        count = 0
        #   print(state_list)
        for state in state_list:
            vertex_set = graph.get_vertex_set(state)
            if len(vertex_set) == k:
                if are_isomorphic(graph.get_adj_mat(vertex_set), motif_adj_mat):
                    # print(walk)
                    # print(list(vertex_set), end = ", ")
                    state_prob = graph.motif_inc_prob(state, alpha=alpha)
                    # print("found motif {} with probability {}".format(state, state_prob))
                    count += 1/state_prob
        # print()
        return count/len(state_list)


    def plot_error_motif(self, list_walk_list : list, step_size: int, 
                                   motif_adj_mat: list, k: int, actual_count: int, alpha: float):
        graph = self
        if graph.is_clean == False:
            graph.clean_graph()
        assert graph.is_clean
        assert len(motif_adj_mat) == len(motif_adj_mat[0])
        assert k == len(motif_adj_mat)
        assert len(graph.stationary_dist.keys()) == len(graph.valid_vertices)
        iterations = len(list_walk_list)
        list_state_list = []

        def add_walk_to_list(walk):
            list_state_list.append(self.get_k_state_list(walk, k))

        for i in range(iterations):
            add_walk_to_list(list_walk_list[i])
        assert (len(list_state_list[i]) == len(list_state_list[i+1]) for i in range(iterations-1))
        r = []
        error_list = []
        start_ind = 0
        end_ind = step_size
        state_len = len(list_state_list[0])
        count = [0]*iterations
        while (end_ind <= state_len + step_size):
            # now we get counts across all lists interation
            for itr in range(iterations):
                state_list = list_state_list[itr]
                for state in state_list[start_ind: end_ind]:
                    vertex_set = graph.get_vertex_set(state)
                    if len(vertex_set) == k and are_isomorphic(graph.get_adj_mat(vertex_set), motif_adj_mat):
                        count[itr] += 1 / graph.motif_inc_prob(state, alpha=alpha)
            # using the count for current end_ind, we calculate the error
            walk_len_curr = min(end_ind, state_len)
            r.append(walk_len_curr)
            curr_error = (sum([(i/walk_len_curr - actual_count)** 2 for i in count])/iterations)**0.5/actual_count
            error_list.append(curr_error)
            start_ind = end_ind
            end_ind = end_ind + step_size
        return r, error_list


'''
****************************************************************************************************
Graph class RW-BB ends here
****************************************************************************************************
'''

######################################################################################################
# Classes end here
######################################################################################################

'''
Actual Counting Begins Here
'''
def count_cyclic_C3(graph: Union[Graph_RWIS, Graph_RWRJ, Graph_RWBB]):
    '''
    Counting Triangles (a,b,c) for form:
    a -> b -> c -> a
    Every traingle is associated with the vertex of min tag
    '''
    triangle_ct = 0
    for node in graph.valid_vertices:
        for in_n in graph.in_neigh_valid[node]:
            for out_n in graph.out_neigh_valid[node]:
                flag = are_isomorphic(graph.get_adj_mat(
                    [node, in_n, out_n]), cyclic_C3_adj_mat)
                triangle_ct += int(flag)
    return triangle_ct//3


def count_acyclic_C3(graph: Union[Graph_RWIS, Graph_RWRJ, Graph_RWBB]):
    '''
    Counting Triangles (a,b,c) for form:
    a -> b -> c and a -> c
    Every traingle is associated with the vertex of min tag
    '''
    triangle_ct = 0
    for node in graph.valid_vertices:
        for in_n in graph.in_neigh_valid[node]:
            for out_n in graph.out_neigh_valid[node]:
                flag = are_isomorphic(graph.get_adj_mat(
                    [node, in_n, out_n]), acyclic_C3_adj_mat)
                triangle_ct += int(flag)
    return triangle_ct


def count_cyclic_C4(graph: Union[Graph_RWIS, Graph_RWRJ, Graph_RWBB]):
    c4_ct = 0
    for v1 in graph.valid_vertices:
        # get a neighbour
        for v2 in graph.out_neigh_valid[v1]:
            if v2 < v1:
                continue
            for v3 in graph.out_neigh_valid[v2]:
                if v3 < v1 or (graph.is_edge(v1, v3) or graph.is_edge(v3, v1)):
                    continue
                for v4 in graph.out_neigh_valid[v3]:
                    if v4 < v1 or (graph.is_edge(v2, v4) or graph.is_edge(v4, v2)) or len(set([v1, v2, v3, v4])) != 4:
                        continue
                    c4_ct += int(are_isomorphic(graph.get_adj_mat(
                        [v1, v2, v3, v4]), cyclic_C4_adj_mat) and len(set([v1, v2, v3, v4])) == 4)
    return c4_ct


def count_source_S3(graph: Union[Graph_RWIS, Graph_RWRJ, Graph_RWBB]):
    ct = 0
    for node in graph.valid_vertices:
        # get 3 out neigh
        n_lst = sorted(graph.out_neigh_valid[node])
        tot_lst = [list(i) for i in list(combinations(n_lst, 3))]
        for vertices in tot_lst:
            vertices = vertices + [node]
            if are_isomorphic(graph.get_adj_mat(vertices), source_S3_adj_mat):
                # print(vertices)
                ct += 1
    return ct


# def count_leaf_C3(graph: Union[Graph_RWIS, Graph_RWRJ, Graph_RWBB]):
#     '''
#     Counting Triangles (a,b,c) --> d for form:
#     a -> b , a -> c -> b and c ---> d
#     '''
#     leaf_c3_ct = 0
#     for v1 in graph.valid_vertices:
#         for v2 in graph.in_neigh_valid[v1]:
#             for v3 in graph.out_neigh_valid[v1]:
#                 neigh_list = list(set(graph.out_neigh_valid[v1]) & set(
#                     graph.in_neigh_valid[v1]))
#                 for v4 in neigh_list:
#                     if len(set([v1, v2, v3, v4])) != 4 or (graph.is_edge(v4, v2)) or (graph.is_edge(v2, v4)) \
#                             or (graph.is_edge(v3, v4)) or (graph.is_edge(v4, v3)):
#                         continue
#                     flag = are_isomorphic(graph.get_adj_mat(
#                         [v1, v2, v3, v4]), leaf_C3_adj_mat)
#                     # if flag:
#                     #       print([v1, v2, v3, v4])
#                     leaf_c3_ct += int(flag)
#     return leaf_c3_ct

def count_leaf_C3(graph: Union[Graph_RWIS, Graph_RWRJ, Graph_RWBB]):
    '''
    Counting Triangles (a,b,c) --> d for form:
    a -> b , a -> c -> b and c ---> d
    '''
    leaf_c3_ct = 0
    for v1 in graph.valid_vertices:
        for v2 in graph.in_neigh_valid[v1]:
            for v3 in graph.out_neigh_valid[v1]:
                neigh_list = graph.out_neigh_valid[v1]
                for v4 in neigh_list:
                    if len(set([v1, v2, v3, v4])) != 4 or (graph.is_edge(v4, v2)) or (graph.is_edge(v2, v4)) \
                            or (graph.is_edge(v3, v4)) or (graph.is_edge(v4, v3)):
                        continue
                    flag = are_isomorphic(graph.get_adj_mat(
                        [v1, v2, v3, v4]), leaf_C3_adj_mat)
                    # if flag:
                    #       print([v1, v2, v3, v4])
                    leaf_c3_ct += int(flag)
    return leaf_c3_ct


# def count_K4_clique(graph: Union[Graph_RWIS, Graph_RWRJ, Graph_RWBB]):
#     '''
#     count strongly connected cliques of size 4
#     '''
#     clique_ct = 0
#     for v1 in graph.valid_vertices:
#         valid_neigh_v1 = list(
#             set(graph.out_neigh_valid[v1]) & set(graph.in_neigh_valid[v1]))
#         for v2 in valid_neigh_v1:
#             if v2 < v1:
#                 continue
#             valid_neigh_v2 = [v for v in list(
#                 set(graph.out_neigh_valid[v2]) & set(graph.in_neigh_valid[v2])) if v != v1]
#             for v3 in valid_neigh_v2:
#                 if v3 < v1:
#                     continue
#                 valid_neigh_v3 = [v for v in list(set(graph.out_neigh_valid[v3]) & set(graph.in_neigh_valid[v3])
#                                                   & set(graph.out_neigh_valid[v2]) & set(graph.in_neigh_valid[v2])
#                                                   & set(graph.out_neigh_valid[v1]) & set(graph.in_neigh_valid[v1])) if v != v1 and v != v2 and v != v3]
#                 for v4 in valid_neigh_v3:
#                     if v4 < v1 or sorted([v1, v2, v3, v4]) != [v1, v2, v3, v4]:
#                         continue
#                     flag = are_isomorphic(graph.get_adj_mat(
#                         [v1, v2, v3, v4]), clique_K4_mat) and len(set([v1, v2, v3, v4])) == 4
#                 #   if flag:
#                 #         print([v1, v2, v3, v4])
#                     clique_ct += int(flag)
#     return clique_ct

def count_K4_clique(graph: Union[Graph_RWIS, Graph_RWRJ, Graph_RWBB]):
    '''
    count strongly connected cliques of size 4
    '''
    clique_ct = 0
    for v1 in graph.valid_vertices:
        # get a neighbour
        for v2 in graph.out_neigh_valid[v1]:
            for v3 in graph.out_neigh_valid[v2]:
                if not graph.is_edge(v3, v1) or len(set([v1,v2,v3])) != 3:
                    continue
                for v4 in graph.out_neigh_valid[v3]:
                    if not graph.is_edge(v2, v4) or len(set([v1, v2, v3, v4])) != 4:
                        continue
                    # print([v1, v2, v3, v4])
                    clique_ct += int(are_isomorphic(graph.get_adj_mat(
                        [v1, v2, v3, v4]), clique_K4_mat) and len(set([v1, v2, v3, v4])) == 4)
    return clique_ct


def count_butterfly(graph: Union[Graph_RWIS, Graph_RWRJ, Graph_RWBB]):
    butterfly_ct = 0
    for v1 in graph.valid_vertices:
        # get an out neighbour
        for v2 in graph.out_neigh_valid[v1]:
            # get an in-neighbour for v2 which is bigger than v1
            for v3 in graph.in_neigh_valid[v2]:
                if v3 < v1 or (graph.is_edge(v1, v3) or graph.is_edge(v3, v1)):
                    continue
                for v4 in graph.out_neigh_valid[v3]:
                    if v4 < v2 or (graph.is_edge(v2, v4) or graph.is_edge(v4, v2)) or len(set([v1, v2, v3, v4])) != 4:
                        continue
                    flag = are_isomorphic(graph.get_adj_mat([v1, v2, v3, v4]), butterfly_adj_mat) \
                        and graph.is_edge(v1, v4)
                    butterfly_ct += int(flag)
                    # if (flag):
                    #       print([v1,v2,v3,v4])
    return butterfly_ct

'''
    b <--- a ---> c
'''
def count_wedge(graph: Union[Graph_RWIS, Graph_RWRJ, Graph_RWBB]):
    wedge_ct = 0
    for v1 in graph.valid_vertices:
        # get two out neighbours
        for tuple in list(combinations(graph.out_neigh_valid[v1],2)):
            if are_isomorphic(graph.get_adj_mat( list(tuple) + [v1] ), wedge_adj_mat):
                wedge_ct += 1
                # print(list(tuple) + [v1])
    return wedge_ct


def adj_mat_to_lst(adj_mat):
    adj_lst = {}
    n = len(adj_mat)
    for v in range(n):
        neigh_lst = []
        for neigh in range(n):
            if adj_mat[v][neigh] == 1:
                neigh_lst.append(neigh)
        adj_lst[v] = neigh_lst
    return adj_lst


'''Takes two adj_mat to check if isomorphic'''
def are_isomorphic(mat_A, mat_B):
    if len(mat_A) != len(mat_B):
        return False
    n = len(mat_A)
    gA = pn.Graph(number_of_vertices=n, directed=True)
    gA.set_adjacency_dict(adj_mat_to_lst(mat_A))
    gB = pn.Graph(number_of_vertices=n, directed=True)
    gB.set_adjacency_dict(adj_mat_to_lst(mat_B))

    return pn.isomorphic(gA, gB)
