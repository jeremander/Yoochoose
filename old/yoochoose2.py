import pandas as pd
import numpy as np
import igraph as ig
import os
from copy import deepcopy
from pybloom import BloomFilter
from importlib import reload
from collections import defaultdict

clicks_filename = 'data/yoochoose-clicks.dat'
buys_filename = 'data/yoochoose-buys.dat'

# total click sessions:  33003944
# unique click sessions:  9249729
# total buy sessions:     1150753
# unique buy sessions:     509696
# unique clicked item IDs:  52739
# unique bought item IDs:   19949

class PosNegBloomFilter(BloomFilter):
    """Bloom Filter for either set inclusion or exclusion."""
    def __init__(self, capacity, error_rate = .001, negative = False):
        """If negative = True, user specifies the items not belonging to the desired set."""
        super(PosNegBloomFilter, self).__init__(capacity, error_rate)
        self.negative = negative
    def __contains__(self, key):
        contains = super(PosNegBloomFilter, self).__contains__(key)
        return (not contains if self.negative else contains)

class YoochooseClicksOrBuys(pd.DataFrame):
    """Base subclass for both click and buy Yoochoose data."""
    def __init__(self, df):
        super(YoochooseClicksOrBuys, self).__init__(df)
        self.sessionIDs = sorted(list(set(df['Session ID'])))
        self.itemIDs = sorted(list(set(df['Item ID'])))
        self.num_sessions = len(self.sessionIDs)
        self.num_items = len(self.itemIDs)

class YoochooseClicks(YoochooseClicksOrBuys):
    """Data frame subclass for click events."""
    @classmethod
    def from_csv(cls, filename = clicks_filename):
        return cls(pd.read_csv(filename, header = None, names = ['Session ID', 'Timestamp', 'Item ID', 'Category'], dtype = {'Category' : str}, parse_dates = [1]))

class YoochooseBuys(YoochooseClicksOrBuys):
    """Data frame subclass for buy events."""
    @classmethod
    def from_csv(cls, filename = buys_filename):
        return cls(pd.read_csv(filename, header = None, names = ['Session ID', 'Timestamp', 'Item ID', 'Price', 'Quantity'], parse_dates = [1]))

class Yoochoose(object):
    """Class encapsulating the click events and buy events from the Yoochoose data set."""
    def __init__(self, clicks, buys):
        self.clicks = clicks
        self.buys = buys
    def sample_subsets(self, session_samp_sizes = [100000], item_samp_sizes = [None], disjoint_sessions = True, disjoint_items = False, error_rate = 1e-9):
        """Takes random samples of the sessions and items and filters by only those lines of the table. session_samp_sizes and item_samp_sizes are lists of sample sizes to take for sessions and items, respectively. If None is passed as a sample size, it is taken to be the entire set. disjoint_sessions and disjoint_items are flags to ensure that session samples or item samples are disjoint. error_rate is the false positive rate of the Bloom filters."""
        # Enforce that sample sizes are permitted
        session_samp_sizes = [self.clicks.num_sessions if (size is None) else size for size in session_samp_sizes]
        item_samp_sizes = [self.clicks.num_items if (size is None) else size for size in item_samp_sizes]
        assert (len(session_samp_sizes) == len(item_samp_sizes)), "Same number of session and item samples required."
        assert (((not disjoint_sessions) or (sum(session_samp_sizes) <= len(self.clicks.sessionIDs))) and (max(session_samp_sizes) <= self.clicks.num_sessions)), "Too many distinct session IDs requested."
        assert (((not disjoint_items) or (sum(item_samp_sizes) <= self.clicks.num_items)) and (max(item_samp_sizes) <= self.clicks.num_items)), "Too many distinct item IDs requested."
        # Get the samples to keep
        if disjoint_sessions:
            session_samp_sizes_cumsum = np.cumsum([0] + session_samp_sizes)
            perm = np.random.permutation(self.clicks.sessionIDs)
            sessionID_samples = [[perm[j] for j in range(session_samp_sizes_cumsum[i], session_samp_sizes_cumsum[i + 1])] for i in range(len(session_samp_sizes))]
        else:
            sessionID_samples = [np.random.permutation(self.clicks.sessionIDs)[:size] for size in session_samp_sizes]
        if disjoint_items:
            item_samp_sizes_cumsum = np.cumsum([0] + item_samp_sizes)
            perm = np.random.permutation(self.clicks.itemIDs)
            itemID_samples = [[perm[j] for j in range(item_samp_sizes_cumsum[i], item_samp_sizes_cumsum[i + 1])] for i in range(len(item_samp_sizes))]
        else:
            itemID_samples = [np.random.permutation(self.clicks.itemIDs)[:size] for size in item_samp_sizes]
        samples = []
        for i in range(len(session_samp_sizes)):
            # Build Bloom filters
            session_filter = PosNegBloomFilter(capacity = session_samp_sizes[i], error_rate = error_rate, negative = (float(session_samp_sizes[i]) / self.clicks.num_sessions > 0.5))
            filter_set = set(self.clicks.sessionIDs).difference(sessionID_samples[i]) if session_filter.negative else set(sessionID_samples[i])
            for ID in filter_set:
                session_filter.add(ID)
            item_filter = PosNegBloomFilter(capacity = item_samp_sizes[i], error_rate = error_rate, negative = (float(item_samp_sizes[i]) / self.clicks.num_sessions > 0.5))
            filter_set = set(self.clicks.itemIDs).difference(itemID_samples[i]) if item_filter.negative else set(itemID_samples[i])
            for ID in filter_set:
                item_filter.add(ID)
            # Filter session IDs and item IDs in the sample using the Bloom filters
            clicks = YoochooseClicks(self.clicks[self.clicks['Session ID'].map(lambda x : x in session_filter) & self.clicks['Item ID'].map(lambda x : x in item_filter)])
            buys = YoochooseBuys(self.buys[self.buys['Session ID'].map(lambda x : x in session_filter) & self.buys['Item ID'].map(lambda x : x in item_filter)])
            samples.append(Yoochoose(clicks, buys))
        return samples
    def to_csv(self, prefix = None):
        """Writes clicks and buys to separate CSV files."""
        if (prefix is None):
            ctr = 1
            while (os.path.exists('data/yoochoose%d-clicks.dat' % ctr) or os.path.exists('data/yoochoose%d-buys.dat' % ctr)):
                ctr += 1
            prefix = 'data/yoochoose%d' % ctr
        self.clicks.to_csv(prefix + '-clicks.dat', header = False, index = False)
        self.buys.to_csv(prefix + '-buys.dat', header = False, index = False)
    def build_graphs(self):
        """Builds bipartite graphs of the data using networkx, one for clicks and one for buys. An edge exists between a session and an item if a click or buy occurred for that session/item pair. 
        Vertex attributes for the click graph are: (For sessions) 1) Number of clicks (same as number of incident edges), 2) Duration of session (0 if only one click occurred, else the time difference in seconds between the first click and the last click). (For items) 1) Number of clicks.
        Edge attributes for the click graph are: 1) Timestamp, 2) Category.
        Vertices for the buy graph are the same as for the click graph, even if they have degree zero.
        Edge attributes for the buy graph are: 1) Timestamp, 2) Price, 3) Quantity."""
        self.click_graph = ig.Graph()
        num_clicks_dict = defaultdict(int)
        timestamp_dict = defaultdict(list)
        self.click_graph.add_vertices([str(ID) for ID in self.clicks.sessionIDs])  # make the IDs strings
        self.click_graph.vs['type'] = 'Session'
        self.click_graph.add_vertices([str(ID) for ID in self.clicks.itemIDs])
        self.click_graph.vs[self.clicks.num_sessions:]['type'] = 'Item'
        for i in range(len(self.clicks)):
            click = self.clicks.iloc[i]
            sessionID = click['Session ID']
            itemID = click['Item ID']
            num_clicks_dict[sessionID] += 1
            num_clicks_dict[itemID] += 1
            timestamp_dict[sessionID].append(click['Timestamp'])
            self.click_graph.add_edge(str(sessionID), str(itemID), timestamp = click['Timestamp'], category = click['Category'])
        for v in self.click_graph.vs.select(type = 'Session'):
            sessionID = int(v['name'])
            timestamps = timestamp_dict[sessionID]
            timestamps.sort()
            duration = 0.0 if (len(timestamps) <= 1) else (timestamps[-1] - timestamps[0]).total_seconds()
            v['duration'] = duration
            v['numClicks'] = num_clicks_dict[sessionID]
        for v in self.click_graph.vs.select(type = 'Item'):
            v['numClicks'] = num_clicks_dict[int(v['name'])]
        self.buy_graph = deepcopy(self.click_graph)
        self.buy_graph.delete_edges(None)  # delete all edges
        for i in range(len(self.buys)):
            buy = self.buys.iloc[i]
            self.buy_graph.add_edge(str(buy['Session ID']), str(buy['Item ID']), timestamp = buy['Timestamp'], price = buy['Price'], quantity = buy['Quantity'])
    # Histograms: Number of clicks (by session, item), number of buys (by session, item), durations (session), price (buy event), quantity (buy event)
    @classmethod
    def from_training(cls, prefix = 'data/yoochoose'):
        """Loads clicks and buys from training data files."""
        return cls(YoochooseClicks.from_csv(prefix + '-clicks.dat'), YoochooseBuys.from_csv(prefix + '-buys.dat'))
