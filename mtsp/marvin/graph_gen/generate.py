'''
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
'''

'''
This file processes the map graphs and creates objects that can be called upon
to receive training and test data.
'''

import json
import os
from marvin.utils.utils import optimized
import numpy as np
import pickle
from scipy.spatial import distance_matrix
from random import seed, shuffle, randint

import torch  # noqa F401

seed(0)


class LoadGraph():
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)


class NoViableGraph(Exception):
    """Raised when no viable graph is found when loading graphs"""
    pass


class DataGenerator:
    train_path = 'marvin/data/train_graphs'
    train_files = os.listdir(
        train_path
    )

    val_path = 'marvin/data/val_graphs'
    val_files = os.listdir(
        val_path
    )

    test_path = 'marvin/data/test_graphs'
    test_files = os.listdir(
        test_path
    )

    index = 0

    @classmethod
    def shuffle_dataset(cls):
        """Shuffles the training dataset"""

        shuffle(cls.train_files)

    def _load_dataset(self, dataset, dataset_path, num_graphs,
                      args=None, min_size=10, max_size=30):
        """Loads a dataset as an array of preprocessed graphs

        Arguments:
            dataset {List[str]} -- list of files making up the dataset
            dataset_path {str} -- overall path to the top directory of the
                dataset
            num_graphs {int} -- max number of graphs to load (could be less)

        Keyword Arguments:
            args {argparse.ArgumentParser} --
                [arguments object] (default: {None})
            min_size {int} -- [min graph size to be loaded] (default: {10})
            max_size {int} -- [max graph size to be loaded] (default: {30})

        Returns:
            List[ProcessedGraph] -- List of all processed graphs
        """

        shuffle(dataset)
        graphs = []

        if args is None:
            max_nodes = max_size
            min_nodes = min_size
        else:
            max_nodes = args.max_size
            min_nodes = args.min_size

        for i in range(len(dataset)):
            if len(graphs) >= num_graphs:
                break

            p = dataset[i]

            num_nodes = int(p.split("_")[0])

            if num_nodes < max_nodes and num_nodes > min_nodes:
                obj = json.load(open(os.path.join(
                        dataset_path,
                        p
                ), 'r'))

                items = obj.items()
                for key, val in items:
                    if key != 'points':
                        obj[key] = torch.tensor(val)

                graphs.append(LoadGraph(obj))

        return graphs

    @classmethod
    def dataset_testset(cls, num_graphs, **kwargs):
        return cls._load_dataset(cls, cls.test_files, cls.test_path,
                                 num_graphs, **kwargs)

    @classmethod
    def dataset_valset(cls, num_graphs, **kwargs):
        return cls._load_dataset(cls, cls.val_files, cls.val_path,
                                 num_graphs, **kwargs)

    @classmethod
    def random_dataset(cls, num_graphs, args):
        """
        Generates a random set of tsp graphs for generic training purposes

        Arguments:
            num_graphs {int} -- number of graph to generate in this dataset

        Keyword Arguments:
            args {argparse.ArgumentParser} --
                [arguments object] (default: {None})
            min_size {int} -- [min graph size to be loaded] (default: {10})
            max_size {int} -- [max graph size to be loaded] (default: {30})

        Returns:
            List[ProcessedGraph] -- list of all the process graphs
        """

        max_size = args.max_size
        min_size = args.min_size

        graphs = []
        for i in range(num_graphs):
            size = randint(min_size, max_size)

            # randomly generate points of the desired size for standard tsp graph            
            points = torch.torch.distributions.Uniform(-1, 1).sample((size, 1, 2))
            # points = torch.rand((size, 1, 2))

            adj = points.expand(size, size, 2) - points.transpose(0, 1).expand(size, size, 2)

            # distance matrix
            # adj = distance_matrix([points, points])
            adj = adj.norm(dim=-1)

            transition_matrix = torch.ones((size, size))
            speeds = torch.ones((size, size))

            # all nodes can connect to all other nodes directly
            norm_trans = transition_matrix / size

            # finds shortest path between all nodes
            pred = torch.tensor(optimized.fw_pred(adj.numpy()))
            dense = torch.clone(adj)
            actual_distance = torch.clone(dense)

            # normalized distance matrix
            dense_norm = (dense - dense.mean()) / (1e-5 + dense.std())

            graphs.append(LoadGraph({
                'points': points,
                'speeds': speeds,
                'adj': adj,
                'transition_matrix': transition_matrix,
                'norm_trans': norm_trans,
                'pred': pred,
                'dense': dense,
                'actual_distance': actual_distance,
                'dense_norm': dense_norm
            }))

        return graphs


    @classmethod
    def get_graph(cls, args=None, min_size=10, max_size=30):
        """Loads a graph object within the sizing constraints

        Keyword Arguments:
            args {argparse.ArgumentParser} --
                [arguments object] (default: {None})
            min_size {int} -- [min graph size to be loaded] (default: {10})
            max_size {int} -- [max graph size to be loaded] (default: {30})

        Returns:
            ProcessedGraph -- A Processed graph object with all
                transformed versions
            of the graph
        """
        if args is None:
            max_nodes = max_size
            min_nodes = min_size
        else:
            max_nodes = args.max_size
            min_nodes = args.min_size

        # tries to find a graph with the correct sizing constraints
        # over 10000 graph
        count = 0
        while count < 10000:
            if cls.index == len(cls.train_files):
                shuffle(cls.train_files)
                cls.index = 0

            path = cls.train_files[cls.index]

            num_nodes = int(path.split("_")[0])

            cls.index += 1
            if num_nodes < max_nodes and num_nodes > min_nodes:
                obj = json.load(open(os.path.join(
                        cls.train_path,
                        path
                ), 'r'))

                items = obj.items()
                for key, val in items:
                    if key != 'points':
                        obj[key] = torch.tensor(val)

                import ipdb; ipdb.set_trace()

                return LoadGraph(obj)

        raise NoViableGraph("No graph within the sizing constraints was found")
