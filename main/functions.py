import pandas as pd
import os
import random
from tqdm import tqdm
import pickle
import numpy as np
from collections import defaultdict, Counter
from prettytable import PrettyTable
import itertools
import scipy.sparse as sp


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

def IsSymmetric(mat, g):
    """
    Build a lil matrix to create a sparse matrix of the vertices and edges,
    get the sum of the point in the matrix,
    check if the matrix is symmetric or not
    """
    # looping on each vertex to assign the edges == 1
    for vertex in g.graph_d:   
        if isinstance(g.graph_d[vertex], int):
            mat[vertex, g.graph_d[vertex]] = 1
        else:
            for target in g.graph_d[vertex]:
                mat[vertex, target] = 1
    
    rows, cols = mat.nonzero() # get only the non zero elements from the sparse matrix 
    return rows, cols

"""
-------------------------------------------------------------------
REQUEST 2
-------------------------------------------------------------------
"""
def pages_reached(page, click, dic):
    total_pages = [] # This list will store number of pages
    page_list = [] #This list will store input value initially and then will add correspondence value as per number of click
    page_list.append(str(page))
    for no_of_click in range(click): #This will run as per number of clicks
        new_lst = []                 
        for i in page_list:
            for j in dic[i]:
                new_lst.append(str(j))
                total_pages.append(str(j))
        page_list = new_lst
    return total_pages

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

def minimum_number_clicks(graph, categories_red, data):
    print('Write the category')
    while True:
        category_input = str(input())
        if category_input not in categories_red:
            print(category_input, ' not exist as category, change category')
        else:
            break
    print()
    print("Write the set of pages in the category chosen separated by a ',':")
    pages_input = input()
    pages_input = pages_input.split(',')
    pages_input = [int(i) for i in pages_input]

    print()
    pages_not = []
    for pages in pages_input:
        if pages not in categories_red[category_input]:
            print(pages, ' not in ', category_input)
            pages_not.append(pages)
    pages_input = [i for i in pages_input if i not in pages_not]  
    
    graph = graph                       # the graph
    central_vertex = most_central_article(categories_red[category_input], in_degree_centrality(data))[0]   # set the max vertex
    v = central_vertex

    visited = [False] * (max(graph) + 1) # set as False the visited vertex
    queue = []                           # set the queue list
    queue.append(v)                      # append the starting vertex to the list
    
    visited[v] = True                    # set the starting vertex as visited
    reached = 0                          # initialize the number of reached vertex
    reached_vertex = []                  # initialize the list of reached vertex
    number_of_click = 0

    while queue:
        if reached < (len(pages_input)):
            v = queue.pop(0)
            
            try:
                number_of_click += 1
                for i in graph[v]:
                    if visited[i] == False:
                        visited[i] = True
                        queue.append(i)
                    
                        if i in pages_input:
                            reached += 1
                            reached_vertex.append(i)
                            
            except TypeError:
                number_of_click += 1
                j = graph[v]
                if visited[j] == False:
                    visited[j] = True
                    queue.append(j)
                    
                    if j in pages_input:
                        reached += 1
                        reached_vertex.append(j)
                        

        else:
            break
    print('Reached vertex are: {}'.format(reached_vertex))
    print('Minimum number of clicks, from most central article {} to reach the set of pages, is {}.'.format(central_vertex, number_of_click))
    not_reached_vertex = [i for i in pages_input if i not in reached_vertex]
    print('Not possible to reach {}'.format(not_reached_vertex))

"""
-------------------------------------------------------------------
REQUEST 4
-------------------------------------------------------------------
"""

def cat_subgraph(category_1, category_2, categories_dict, dataframe):

    #first get the two lists of pages of the categories
    a = categories_dict[category_1]
    b = categories_dict[category_2]

    #given those lists, find the sources for both
    source_a = dataframe[dataframe['Source'].isin(a)]
    source_b = dataframe[dataframe['Source'].isin(b)]

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
REQUEST 5
-------------------------------------------------------------------
"""

def relevant_pages(category,categories_dict,  dataframe):
    a = categories_dict[category]
    d1 = dataframe[dataframe['Source'].isin(a)]
    d2 = dataframe[dataframe['Target'].isin(a)]
    concat = [d1['Source'], d2['Target']]
    df_concat = pd.concat(concat)
    d = list(df_concat.unique())

    return d


def page_distance(start, visited, distance,sub_graph, category, categories_dict, dataframe):

    queue = [] + [start]
    visited[start] = True
    distance[start] = 0
    l = relevant_pages(category, categories_dict, dataframe)
    distances = []

    while len(queue) > 0 :
        z = queue[0]
        queue.pop(0)
        for i in sub_graph[z]:
            if i in l:
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True
                if distance[i] > distance[z]:
                    distance [i] = distance[z] + 1
                    distances.append(distance[i])

    return (distances)

def distances_from_category(input_category,categories_dict, dataframe, all_nodes ):
    results = {}

    for cat1 in tqdm(categories_dict.keys()):
        if cat1 != input_category:
            visited = dict.fromkeys(all_nodes, False)
            distance = dict.fromkeys(all_nodes, float('inf'))
            sub_graph = cat_subgraph(input_category, cat1, categories_dict, dataframe)

            for key,val in sub_graph.items():
                if isinstance(sub_graph[key], int):
                    sub_graph[key] = [sub_graph[key]]
            pages = relevant_pages(cat1,categories_dict, dataframe)
            aux = []
            for i in pages:
                x = page_distance(i,visited, distance,sub_graph,
                                  input_category, categories_dict, dataframe)
                if x:
                    aux.append(x)
            merged = np.array(list(itertools.chain(*aux)))
            m = np.median(merged)
            #print(m)
            results[cat1] = m
    write_pickle('data/' + input_category, results)
    return (results)

def pretty_table(table):    
    d_view = [ (v,k) for k,v in table.items() ]
    d_view.sort(reverse=False) # natively sort tuples by first element
    t = PrettyTable(['Categories', 'Distance'])
    for v,k in d_view:
        if not np.isnan(v):
            t.add_row([k,v])
    print(t)

def score_categories(category_name):
    if isinstance(category_name, str):
        cat_name = "data/"+category_name+".pkl"
        ll = read_pickle(cat_name)
        d_view = [ (v,k) for k,v in ll.items() ]
        d_view.sort(reverse=False) # natively sort tuples by first element
        t = PrettyTable(['Categories', 'Distance'])
        for v,k in d_view:
            if not np.isnan(v):
                t.add_row([k,v])
        print(t)
    else:
        print("Category name must be string")



"""
-------------------------------------------------------------------
REQUEST 6
-------------------------------------------------------------------
"""
def inbound(data):
    """
    create a dictionary to get the inbound vertices of each vertex
    """
    concat = [data['Source'], data['Target']]    # concat all the nodes
    all_nodes = list(pd.concat(concat).unique()) 
    inbound = {}
    for node in tqdm(all_nodes):
        inbound[node] = data[data['Target'] == node].Source.values.tolist()
    return inbound

def out_bound(g):
    """
    create a dictionary to get the number of outbound vertices of each vertex
    """
    out_deg = {}
    for key in g.graph_d:
        if isinstance(g.graph_d[key], list):
            out_deg[key] = len(g.graph_d[key])
        elif (isinstance(g.graph_d[key], int)): 
            out_deg[key] = 1
    return out_deg

def page_rank(max_iterations, categories_red, inbound, outbound):    
    """
    function of the implementation of the Page Rank algorithm
    
    Parameters
    ----------
    max_iterations : int, number of iterations
    categories_red : dict, dictionary of categories
    inbound : dict, dictionary of inbound vertex
    outbound : dict, dictionary of outbound vertex

    Returns
    -------
    dict with keys as categories and values as score of Page Rank
    """
    pr_categories = {}
    
    for cat in tqdm(categories_red):
        normalized_vertex = dict.fromkeys(categories_red[cat], 
                                          1/len(categories_red[cat])) 
        for i in range(max_iterations):
            normalized_vertex_temp = dict.fromkeys(categories_red[cat], 0) 
            for vertex in normalized_vertex:
                pr = 0
                try:
                    for in_bound in inbound[vertex]:
                        try:
                            pr += normalized_vertex[in_bound]/outbound[in_bound]
                        except KeyError:
                            pass
                except KeyError:
                    pass
                normalized_vertex_temp[vertex] = pr
            normalized_vertex = normalized_vertex_temp
        pr_categories[cat] = np.max(np.array(list(normalized_vertex.values())))
    return pr_categories

def pretty_table_2(table):    
    d_view = [ (v,k) for k,v in table.items() ]
    d_view.sort(reverse=False) # natively sort tuples by first element
    t = PrettyTable(['Categories', 'Score', 'Page Rank'])
    count=0
    for v,k in d_view:
        count+=1
        if not np.isnan(v):
            t.add_row([k,v,count])
    print(t)