import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError



class MoviesLoader(BasicDataset):
    """
    Dataset type for PyTorch \n
    Includes graph information for the MovieLens dataset
    """

    def __init__(self, path="../data/ml-latest-small", test_ratio=0.1, seed=2020):
        # Train or test
        cprint(f'loading [{path}]')
        self.path = path
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']

        # Load data
        movies_file = os.path.join(path, 'movies.csv')
        ratings_file = os.path.join(path, 'ratings.csv')

        movies = pd.read_csv(movies_file)
        ratings = pd.read_csv(ratings_file)

        # Map userId and movieId to indices starting from 0
        self.user_map = {uid: i for i, uid in enumerate(ratings['userId'].unique())}
        self.movie_map = {mid: i for i, mid in enumerate(movies['movieId'].unique())}

        ratings['userIndex'] = ratings['userId'].map(self.user_map)
        ratings['movieIndex'] = ratings['movieId'].map(self.movie_map)

        # Filter interactions with rating > 4
        filtered_ratings = ratings[ratings['rating'] > 4]

        self.n_user = len(self.user_map)
        self.m_item = len(self.movie_map)

        # Split into train and test
        np.random.seed(seed)
        test_indices = np.random.choice(
            filtered_ratings.index, 
            size=int(test_ratio * len(filtered_ratings)), 
            replace=False
        )
        train_indices = list(set(filtered_ratings.index) - set(test_indices))

        train_data = filtered_ratings.loc[train_indices]
        test_data = filtered_ratings.loc[test_indices]

        self.trainUser = train_data['userIndex'].to_numpy()
        self.trainItem = train_data['movieIndex'].to_numpy()
        self.testUser = test_data['userIndex'].to_numpy()
        self.testItem = test_data['movieIndex'].to_numpy()

        self.traindataSize = len(self.trainUser)
        self.testDataSize = len(self.testUser)

        print(f"Number of users: {self.n_user}")
        print(f"Number of movies: {self.m_item}")
        print(f"Number of training interactions: {self.traindataSize}")
        print(f"Number of testing interactions: {self.testDataSize}")

        # Create User-Item Interaction Matrix
        self.UserItemNet = csr_matrix(
            (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item)
        )
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        # Pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"Dataset is ready to go!")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def allPos(self):
        return self._allPos

    @property
    def testDict(self):
        return self.__testDict

    def getSparseGraph(self):
        """
        Build a sparse adjacency matrix for LightGCN
        """
        print("Loading adjacency matrix")
        if hasattr(self, "Graph") and self.Graph is not None:
            return self.Graph

        user_dim = torch.LongTensor(self.trainUser)
        item_dim = torch.LongTensor(self.trainItem)
        
        first_sub = torch.stack([user_dim, item_dim + self.n_user])
        second_sub = torch.stack([item_dim + self.n_user, user_dim])
        index = torch.cat([first_sub, second_sub], dim=1)
        data = torch.ones(index.size(-1)).int()
        
        self.Graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_user + self.m_item, self.n_user + self.m_item]))
        dense = self.Graph.to_dense()
        
        D = torch.sum(dense, dim=1).float()
        D[D == 0.] = 1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense / D_sqrt
        dense = dense / D_sqrt.t()
        
        index = dense.nonzero()
        data = dense[dense >= 1e-9]
        assert len(index) == len(data)
        
        self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_user + self.m_item, self.n_user + self.m_item]))
        self.Graph = self.Graph.coalesce().to(world.device)
        
        print("Graph is ready!")
        return self.Graph

    def __build_test(self):
        """
        Construct a dictionary {user: [items]}
        """
        test_data = {}
        for user, item in zip(self.testUser, self.testItem):
            if user not in test_data:
                test_data[user] = []
            test_data[user].append(item)
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        Get user-item feedback
        users: [-1] shape
        items: [-1] shape
        return: feedback [-1]
        """
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems