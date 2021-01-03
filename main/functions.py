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
        for node in g.graph_d:
            try:
                for neigh in g.graph_d[node]:
                    edges_lst.append((node, neigh))
            except TypeError:
                edges_lst.append((node, g.graph_d[node]))
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
