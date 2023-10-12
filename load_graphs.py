import re
import random
from class_graphs import Graph_RWBB, Graph_RWRJ, Graph_RWIS


def load_graph_from_file(file_path, line_skips: int, limit = 1e100):
  edges=[]
  with open(file_path, 'r') as f:
      ct=0
      node_dict = {}
      curr_ind = 0
      start_vertex = None
      while True:
          ct +=1
          line = f.readline()
          if ct <= line_skips:
              continue
          if not line:
              break
          line = re.split('\t|\n|,| ', line)
        #   print(line)
          u = int(line[0])
          try:
            v = int(line[1])
          except:
            print(line, ct)
          if u==v:
                continue
          if u not in node_dict and curr_ind < limit:
                node_dict[u] = curr_ind
                curr_ind +=1
          if v not in node_dict and curr_ind < limit:
                node_dict[v] = curr_ind
                curr_ind +=1
          if u in node_dict and v in node_dict:
            edges.append((node_dict[u], node_dict[v]))
      return curr_ind, edges

def create_graph_from_file (file : str, mode : str, do_clean = True, line_skips = 4, limit = 1e100, allow_multi_edges = True):
      assert mode in ["rwrj", "rwbb", "rwis"]
      num_nodes, edges = load_graph_from_file(file, line_skips, limit)
      graph_type = None
      if mode == "rwrj":
          graph_type = Graph_RWRJ
      elif mode == "rwis":
          graph_type = Graph_RWIS
      elif mode == "rwbb":
          graph_type = Graph_RWBB
      else:
          graph_type = None
      graph_obj = graph_type(num_nodes)
      if allow_multi_edges:
        for e in edges:
                if e not in graph_obj.edges:
                    graph_obj.add_edge(e[0], e[1])
      else:
            for (u,v) in edges:
                 if (u,v) not in graph_obj.edges and (v,u) not in graph_obj.edges:
                        graph_obj.add_edge(u,v)
      print("Loaded Graph")
      if do_clean:
            print("now cleaning")
            graph_obj.clean_graph()
            print("Cleaned Graph")
      return graph_obj

def write_to_file(graph, file_name : str, graph_name : str, usage: str):
    assert graph.is_clean
    f = open("datasets/cleaned_datasets_new/"+file_name + ".txt", "w")
    f.write(graph_name + "\n")
    f.write("Number of nodes: "+str(len(graph.valid_vertices)) + "\n")
    valid_edges = []
    for u in graph.valid_vertices:
      for v in graph.valid_vertices:
            if (u,v) in graph.edges:
                valid_edges.append((u,v))

    f.write("Number of edges: " + str(len(valid_edges)) + "\n")
    f.write(usage+"\n")
    for e in valid_edges:
          f.write(str(e[0]) + "\t" + str(e[1]) + "\n")

# MAIN READING DUNCTION

def create_graph_cleaned_file (file_path: str, mode : str, allow_multi_edges = True):
      assert mode in ["rwrj", "rwbb", "rwis"]
      num_nodes, edges = load_graph_from_file(file_path, 4)
      graph_type = None
      if mode == "rwrj":
          graph_type = Graph_RWRJ
      elif mode == "rwis":
          graph_type = Graph_RWIS
      elif mode == "rwbb":
          graph_type = Graph_RWBB
      else:
          graph_type = None
      graph_obj = graph_type(num_nodes)
      if allow_multi_edges:
            for e in edges:
                if e not in graph_obj.edges:
                    graph_obj.add_edge(e[0], e[1])
      else:
            for (u,v) in edges:
                 if (u,v) not in graph_obj.edges and (v,u) not in graph_obj.edges:
                        graph_obj.add_edge(u,v)
      print("Loaded Graph from cleaned file... Now initializing")
      graph_obj.valid_vertices = list(range(graph_obj.V))
      # check for the start vertex
      size, _ = graph_obj.BFS(random.choice(graph_obj.valid_vertices))
      assert size == graph_obj.V
      # assert ((u,v) not in graph_obj.edges for (v,u) in graph_obj.edges)
      if mode == "rwbb":
            graph_obj.history_stack = []
      assert graph_obj.valid_vertices != -1
      graph_obj.clean_neighbours()
      for node in range(graph_obj.V):
      #       graph_obj.in_neigh_valid[node] = graph_obj.in_neigh[node]
      #       graph_obj.out_neigh_valid[node] = graph_obj.out_neigh[node]
      #       graph_obj.in_deg_valid[node] = len(graph_obj.in_neigh_valid[node])
      #       graph_obj.out_deg_valid[node] = len(graph_obj.out_neigh_valid[node])
            assert graph_obj.in_deg_valid[node] == graph_obj.in_deg[node]
            assert graph_obj.out_deg_valid[node] == graph_obj.out_deg[node]
      print("Graph is Ready!")
      return graph_obj
          
    