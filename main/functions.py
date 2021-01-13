import pandas as pd
import os
import random
from tqdm import tqdm
import pickle
import numpy as np
from collections import defaultdict


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