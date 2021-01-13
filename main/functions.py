import pandas as pd
import os
import random
from tqdm import tqdm
import pickle
import numpy as np
from collections import defaultdict, Counter


"""
-------------------------------------------------------------------
Functions of the preliminary section
-------------------------------------------------------------------
"""


def write_pickle(file_name, content):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'wb') as handle:
        pickle.dump(content, handle)


def read_pickle(file_name):
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)

"""
-------------------------------------------------------------------
REQUEST 1
-------------------------------------------------------------------
"""


def get_graph_dictionary(data):

    concat = [data['Source'], data['Target']]
    df_concat = pd.concat(concat)
    data_2 = data.set_index('Source')


    graph = defaultdict(list)
    for row in tqdm(df_concat.unique()):
        try:
            graph[row] = data_2.loc[row, 'Target'].tolist()
        except AttributeError:
            graph[row] = data_2.loc[row, 'Target'].flatten().tolist()
        except KeyError:
            graph[row]
    return(graph)


class Graph(object):
    def __init__(self, graph_d=None):
        if graph_d == None:
            graph_d = {}
        self.graph_d = graph_d
    # get the vertices of the graph
    def vertices(self):
        return list(self.graph_d.keys())
    # get the edges of the graph
    def edges(self):
        edges_lst = []
        for node in self.graph_d:
            try:
                for neigh in self.graph_d[node]:
                    edges_lst.append((node, neigh))
            except TypeError:
                edges_lst.append((node, self.graph_d[node]))
        return edges_lst


def average_number_pages1(g):
    idx = (random.choice(g.vertices()))
    if isinstance(g.graph_d[idx], list):
        return (len(g.graph_d[idx]))
    else:
        return 1


def average_number_pages(g):
    """
    Method to calculate the average number of links in a random page
    """
    arbitrary_page = random.choice(g.vertices())
    av_number_links_lst = []
    for link in g.graph_d[arbitrary_page]:
        av_number_links_lst.append(link)
    av_number_links = len(av_number_links_lst)
    return av_number_links


def density_graph(g):
    V = len(g.vertices())
    E = len(g.edges())
    return E / (V *(V - 1))


"""
-------------------------------------------------------------------
REQUEST 2
-------------------------------------------------------------------
"""


"""
-------------------------------------------------------------------
REQUEST 3
-------------------------------------------------------------------
"""
def in_degree_centrality(data):
    """
    function that look for the in-degree values of each node
    """
    concat = [data['Source'], data['Target']]    # concat all the nodes
    all_nodes = list(pd.concat(concat).unique()) # get the list of the unique values
    in_degree = dict.fromkeys(all_nodes, 0)      # dict with keys all the nodes and values the 0s
    only_target_node = list(data.Target)         # list of the target nodes which have at least a in-degree value
    for node in only_target_node:
        in_degree[node] +=1                      # for each node in the target, update the dict adding 1
    return in_degree

def most_central_article(category, in_degree_dict):
    """
    function that return the vertex with the highest in-degree centrality
    """
    max_in_degree_value = 0
    max_in_degree_vertex = ''
    not_degree_article = []

    for vertex in category:
        if vertex in in_degree_dict:
            vertex_degree = in_degree_dict[vertex]
            if vertex_degree > max_in_degree_value:
                max_in_degree_value = vertex_degree
                max_in_degree_vertex = vertex
        else:
            not_degree_article.append(vertex)
            continue
    return max_in_degree_vertex, max_in_degree_value, not_degree_article


"""
-------------------------------------------------------------------
REQUEST 4
-------------------------------------------------------------------
"""
#first function
def get_graph_dictionary(data):

    concat = [data['Source'], data['Target']]
    df_concat = pd.concat(concat)
    data_2 = data.set_index('Source')


    graph = defaultdict(list)
    for row in df_concat.unique():
        try:
            graph[row] = data_2.loc[row, 'Target'].tolist()
        except AttributeError:
            graph[row] = data_2.loc[row, 'Target'].flatten().tolist()
        except KeyError:
            graph[row]
    return(graph)

def cat_subgraph(c1, c2, df):
    
    #first get the two lists of pages of the categories
    a = cat[c1]
    b = cat[c2]
    
    #given those lists, find the sources for both
    source_a = df[df['Source'].isin(a)]
    source_b = df[df['Source'].isin(b)]
    
    #now find the edges, that have as targets the other list
    edges_ab = source_a[source_a['Target'].isin(b)]
    edges_ba = source_b[source_b['Target'].isin(a)]
    
    #edges within the categories
    edges_aa = source_a[source_a['Target'].isin(a)]
    edges_bb = source_b[source_b['Target'].isin(b)]
    
    #put them together
    sub_df = pd.concat([edges_ab, edges_ba, edges_aa, edges_bb])
    
    #convert input graph in a dict and give that as output
    sub_graph = get_graph_dictionary(sub_df)

    return sub_graph

def show_first_order_neigbors(G, start_node):
    try: 
        sub_nodes = [n for n in G[start_node]]
        edges_sub = [item for item in g.edges() if item[0] == start_node]
        G_sub = nx.DiGraph()
        G_sub.add_nodes_from(sub_nodes)
        G_sub.add_edges_from(edges_sub)
        pos = nx.spring_layout(G_sub)
        nx.draw_networkx_nodes(G_sub, pos, cmap=plt.get_cmap('jet'), 
                               node_color = "orange", node_size = 800)
        nx.draw_networkx_labels(G_sub, pos)
        nx.draw_networkx_edges(G_sub, pos, arrows=True)
        plt.show()
    except:
        print("No link avaible")


# second function
def find_hyperlinks(graph, u, v, link=[]):
        link = link + [u]
        final_links = []
        if u == v:
            return [link]
        for i in graph[u]:
            if i not in link:
                other_links = find_hyperlinks(graph, i, v, link)
                for l in other_links: 
                    final_links.append(l)
        return final_links

def min_hyperlinks(graph, u, v):
    return len(find_hyperlinks(graph, u, v))

"""
-------------------------------------------------------------------
REQUEST 6
-------------------------------------------------------------------
"""
def out_degree_centrality(g):
    out_deg_list = []
    for key in g.graph_d:
        if isinstance(g.graph_d[key], list):
            out_deg_list.append(len(g.graph_d[key]))
        elif (isinstance(g.graph_d[key], int)): 
            out_deg_list.append(1) 
    out_deg = Counter(sorted(out_deg_list))
    return out_deg