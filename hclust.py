"""
Executing code: 
Python hclust.py iris.dat 3

"""

"""
Change log: 

- Nov 8, 2015
1. Change the logic to calculation centroid
2. Add judgement for some invalid input cases

"""

import sys
import math
import os
import heapq
import itertools

class Hierarchical_Clustering:
    def __init__(self, ipt_data, ipt_k):
        self.input_file_name = ipt_data
        self.k = ipt_k
        self.dataset = None
        self.dataset_size = 0
        self.dimension = 0
        self.heap = []
        self.clusters = []
        self.gold_standard = {}

    def initialize(self):
        """
        Initialize and check parameters

        """
        # check file exist and if it's a file or dir
        if not os.path.isfile(self.input_file_name):
            self.quit("Input file doesn't exist or it's not a file")

        self.dataset, self.clusters, self.gold_standard = self.load_data(self.input_file_name)
        self.dataset_size = len(self.dataset)

        if self.dataset_size == 0:
            self.quit("Input file doesn't include any data")

        if self.k == 0:
            self.quit("k = 0, no cluster will be generated")

        if self.k > self.dataset_size:
            self.quit("k is larger than the number of existing clusters")

        self.dimension = len(self.dataset[0]["data"])

        if self.dimension == 0:
            self.quit("dimension for dataset cannot be zero")

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                      Hierarchical Clustering Functions                       """
    """                                                                              """    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def euclidean_distance(self, data_point_one, data_point_two):
        """
        euclidean distance: https://en.wikipedia.org/wiki/Euclidean_distance
        assume that two data points have same dimension

        """
        size = len(data_point_one)
        result = 0.0
        for i in range(size):
            f1 = float(data_point_one[i])   # feature for data one
            f2 = float(data_point_two[i])   # feature for data two
            tmp = f1 - f2
            result += pow(tmp, 2)
        result = math.sqrt(result)
        return result

    def compute_pairwise_distance(self, dataset):
        result = []
        dataset_size = len(dataset)
        for i in range(dataset_size-1):    # ignore last i
            for j in range(i+1, dataset_size):     # ignore duplication
                dist = self.euclidean_distance(dataset[i]["data"], dataset[j]["data"])

                # duplicate dist, need to be remove, and there is no difference to use tuple only
                # leave second dist here is to take up a position for tie selection
                result.append( (dist, [dist, [[i], [j]]]) )

        return result
                
    def build_priority_queue(self, distance_list):
        heapq.heapify(distance_list)
        self.heap = distance_list
        return self.heap

    def compute_centroid_two_clusters(self, current_clusters, data_points_index):
        size = len(data_points_index)
        dim = self.dimension
        centroid = [0.0]*dim
        for index in data_points_index:
            dim_data = current_clusters[str(index)]["centroid"]
            for i in range(dim):
                centroid[i] += float(dim_data[i])
        for i in range(dim):
            centroid[i] /= size
        return centroid

    def compute_centroid(self, dataset, data_points_index):
        size = len(data_points_index)
        dim = self.dimension
        centroid = [0.0]*dim
        for idx in data_points_index:
            dim_data = dataset[idx]["data"]
            for i in range(dim):
                centroid[i] += float(dim_data[i])
        for i in range(dim):
            centroid[i] /= size
        return centroid

    def hierarchical_clustering(self):
        """
        Main Process for hierarchical clustering

        """
        dataset = self.dataset
        current_clusters = self.clusters
        old_clusters = []
        heap = hc.compute_pairwise_distance(dataset)
        heap = hc.build_priority_queue(heap)

        while len(current_clusters) > self.k:
            dist, min_item = heapq.heappop(heap)
            # pair_dist = min_item[0]
            pair_data = min_item[1]

            # judge if include old cluster
            if not self.valid_heap_node(min_item, old_clusters):
                continue

            new_cluster = {}
            new_cluster_elements = sum(pair_data, [])
            new_cluster_cendroid = self.compute_centroid(dataset, new_cluster_elements)
            new_cluster_elements.sort()
            new_cluster.setdefault("centroid", new_cluster_cendroid)
            new_cluster.setdefault("elements", new_cluster_elements)
            for pair_item in pair_data:
                old_clusters.append(pair_item)
                del current_clusters[str(pair_item)]
            self.add_heap_entry(heap, new_cluster, current_clusters)
            current_clusters[str(new_cluster_elements)] = new_cluster
        current_clusters.sort()
        return current_clusters
            
    def valid_heap_node(self, heap_node, old_clusters):
        pair_dist = heap_node[0]
        pair_data = heap_node[1]
        for old_cluster in old_clusters:
            if old_cluster in pair_data:
                return False
        return True
            
    def add_heap_entry(self, heap, new_cluster, current_clusters):
        for ex_cluster in current_clusters.values():
            new_heap_entry = []
            dist = self.euclidean_distance(ex_cluster["centroid"], new_cluster["centroid"])
            new_heap_entry.append(dist)
            new_heap_entry.append([new_cluster["elements"], ex_cluster["elements"]])
            heapq.heappush(heap, (dist, new_heap_entry))

    def evaluate(self, current_clusters):
        gold_standard = self.gold_standard
        current_clustes_pairs = []

        for (current_cluster_key, current_cluster_value) in current_clusters.items():
            tmp = list(itertools.combinations(current_cluster_value["elements"], 2))
            current_clustes_pairs.extend(tmp)
        tp_fp = len(current_clustes_pairs)

        gold_standard_pairs = []
        for (gold_standard_key, gold_standard_value) in gold_standard.items():
            tmp = list(itertools.combinations(gold_standard_value, 2))
            gold_standard_pairs.extend(tmp)
        tp_fn = len(gold_standard_pairs)

        tp = 0.0
        for ccp in current_clustes_pairs:
            if ccp in gold_standard_pairs:
                tp += 1

        if tp_fp == 0:
            precision = 0.0
        else:
            precision = tp/tp_fp
        if tp_fn == 0:
            precision = 0.0
        else:
            recall = tp/tp_fn

        return precision, recall

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                             Helper Functions                                 """
    """                                                                              """    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def load_data(self, input_file_name):
        """
        load data and do some preparations

        """
        input_file = open(input_file_name, 'rU')
        dataset = []
        clusters = {}
        gold_standard = {}
        id = 0
        for line in input_file:
            line = line.strip('\n')
            row = str(line)
            row = row.split(",")
            iris_class = row[-1]

            data = {}
            data.setdefault("id", id)   # duplicate
            data.setdefault("data", row[:-1])
            data.setdefault("class", row[-1])
            dataset.append(data)

            clusters_key = str([id])
            clusters.setdefault(clusters_key, {})
            clusters[clusters_key].setdefault("centroid", row[:-1])
            clusters[clusters_key].setdefault("elements", [id])

            gold_standard.setdefault(iris_class, [])
            gold_standard[iris_class].append(id)

            id += 1
        return dataset, clusters, gold_standard

    def quit(self, err_desc):
        raise SystemExit('\n'+ "PROGRAM EXIT: " + err_desc + ', please check your input' + '\n')

    def loaded_dataset(self):
        """
        use for test only

        """
        return self.dataset

    def display(self, current_clusters, precision, recall):
        print precision
        print recall
        clusters = current_clusters.values()
        for cluster in clusters:
            cluster["elements"].sort()
            print cluster["elements"]



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                               Main Method                                    """
"""                                                                              """    
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
    """
    inputs:
    - ipt_data: a text file name for the input data
    - ipt_k: a value k for the number of desired clusters.

    outputs:
    - opt_clusters: output k clusters, with each cluster contains a set of data points (index for input data)
    - opt_precision
    - opt_recall

    """

    ## input test
    # ipt_data = "iris.dat"
    # ipt_data = "iris_dataset1.txt"
    # ipt_k = 3

    ipt_data = sys.argv[1]      # input data, e.g. iris.dat
    ipt_k = int(sys.argv[2])    # number of clusters, e.g. 3

    hc = Hierarchical_Clustering(ipt_data, ipt_k)
    hc.initialize()
    current_clusters = hc.hierarchical_clustering()
    precision, recall = hc.evaluate(current_clusters)
    hc.display(current_clusters, precision, recall)

    ## euclidean_distance() test
    # loaded_data = hc.loaded_dataset()
    # print loaded_data
    # print hc.euclidean_distance(loaded_data[0]["data"],loaded_data[1]["data"])

    ## compute_centroid() test
    # loaded_data = hc.loaded_dataset()
    # hc.compute_centroid(loaded_data, [10, 11, 12, 13])

    ## distance_list test
    # distance_list = hc.compute_pairwise_distance()
    # distance_list.sort()
    # print distance_list
    
    ## heapq test
    # heap = []
    # data = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
    # data = [[1,4,5], [3,6,1], [5,6,10], [7,2,11], [9,6,1], [2,1,5], [4,2,1], [6,6,5], [8,7,1], [0,1,0]]
    # heapq.heapify(data)
    # print data




