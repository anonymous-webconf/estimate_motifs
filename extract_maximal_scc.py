from load_graphs import create_graph_from_file, write_to_file, create_graph_cleaned_file
from class_graphs import Graph_RWRJ
import sys

# a = input("Enter source file: ")
# file = "datasets/complete_datasets/"+ a + ".txt"

# limit = int(input("Enter node limit: "))
# graph = create_graph_from_file(file=file, mode="rwrj", do_clean = False, limit=limit, allow_multi_edges=False)
# assert graph.valid_vertices == []

list_of_comp = []

def SCCUtil(graph: Graph_RWRJ, u, low, disc, stackMember, st: list):
    disc[u] = graph.Time
    low[u] = graph.Time
    graph.Time += 1
    stackMember[u] = True
    st.append(u)
    # all vertices adjacent to u
    for v in graph.out_neigh[u]:
        if disc[v] == -1:
            SCCUtil(graph, v, low, disc, stackMember, st)
            low[u] = min(low[u], low[v])

        elif stackMember[v] == True:
            low[u] = min(low[u], disc[v])
    
    w = -1
    temp = []
    if low[u] == disc[u]:
        while w!= u:
            w = st.pop()
            temp.append(w)
            # print(temp)
            stackMember[w] = False

        list_of_comp.append(temp)
        temp = []

def SCC(graph : Graph_RWRJ):
    disc = [-1]*graph.V
    low = [-1]*graph.V
    stackMember = [False]*graph.V
    st = []
    for v in range(graph.V):
        if disc[v] == -1:
            SCCUtil(graph, v, low, disc, stackMember, st)


def find_maximal(graph):
    global list_of_comp
    list_of_comp = []
    SCC(graph)
    hash_ = []
    for lst in list_of_comp:
        for i in lst:
            hash_.append(i)
    assert graph.V == len(hash_)

    max_comp = []
    size = -1
    for comp in list_of_comp:
        if len(comp) > size:
            max_comp = comp
            size = len(comp)
    return max_comp

# sys.setrecursionlimit(graph.V)

# lst = find_maximal(graph=graph)
# print(len(lst))

# graph.clean_graph(lst)

# name = input("Enter file name to be written: ")
# desc = input("Enter file description if any: ")

# write_to_file(graph, name, desc, "All Modes")
# # -------------------------------------
# graph = create_graph_cleaned_file("datasets/cleaned_datasets_new/{}.txt".format(name), "rwrj")
# print(len(graph.edges))
# write_to_file(graph, name, desc, "All Modes")


# for (u,v) in graph.edges:
#     try:
#         assert (v,u) not in graph.edges
#     except:
#         print(u,v)

