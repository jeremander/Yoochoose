import pandas as pd
import numpy as np
import networkx as nx
import os
import pickle
from pybloom import BloomFilter
from importlib import reload
from collections import defaultdict

clicks_filename = 'data/yoochoose-clicks.dat'
buys_filename = 'data/yoochoose-buys.dat'
heldout_filename = 'data/yoochoose-heldout.dat'
solution_filename = 'data/yoochoose-solution.dat'
ZERO_THRESH = 1e-12

# total clicks:          33003944
# unique click sessions:  9249729
# total buys:             1150753
# unique buy sessions:     509696
# unique clicked itemIDs:  52739
# unique bought itemIDs:   19949
# purchases with price & quantity == 0:  610030

def dict_to_csv(filename, d):
    """Writes dictionary to a CSV file in the following format: key;item1,item2,... for each line."""
    with open(filename, 'w') as f:
        for key in sorted(d.keys()):
            line = str(key) + ';'
            for val in sorted(d[key]):
                line += str(val) + ','
            f.write(line.strip(',') + '\n')

def csv_to_dict(filename, key_type = str, val_type = str):
    """Reads dictionary from a CSV file in the following format: key;item1,item2,... for each line. key_type and val_type refer to the type of keys and values (e.g. str, int, float)."""
    d = defaultdict(list)
    with open(filename, 'r') as f:
        for line in f:
            key_part, values_part = line.split(';')
            key = key_type(key_part)
            for val in values_part.split(','):
                d[key].append(val_type(val))
    return d


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
        self.sessionIDs = sorted([int(ID) for ID in set(df['sessionID'])])
        self.itemIDs = sorted([int(ID) for ID in set(df['itemID'])])
        self.num_sessions = len(self.sessionIDs)
        self.num_items = len(self.itemIDs)

class YoochooseClicks(YoochooseClicksOrBuys):
    """Data frame subclass for click events."""
    @classmethod
    def from_csv(cls, filename = clicks_filename):
        return cls(pd.read_csv(filename, header = None, names = ['sessionID', 'timestamp', 'itemID', 'category'], dtype = {'category' : str}, parse_dates = [1]))

class YoochooseBuys(YoochooseClicksOrBuys):
    """Data frame subclass for buy events."""
    @classmethod
    def from_csv(cls, filename = buys_filename):
        return cls(pd.read_csv(filename, header = None, names = ['sessionID', 'timestamp', 'itemID', 'price', 'quantity'], parse_dates = [1]))
    @classmethod
    def from_solution(cls, filename = solution_filename):
        buys_by_session = csv_to_dict(filename, key_type = int, val_type = int)
        rows_list = []
        for sessionID in sorted(buys_by_session.keys()):
            for itemID in sorted(buys_by_session[sessionID]):
                rows_list.append([sessionID, itemID])
        return cls(pd.DataFrame(rows_list, columns = ['sessionID', 'itemID']))

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
        assert (((not disjoint_sessions) or (sum(session_samp_sizes) <= len(self.clicks.sessionIDs))) and (max(session_samp_sizes) <= self.clicks.num_sessions)), "Too many distinct sessionIDs requested."
        assert (((not disjoint_items) or (sum(item_samp_sizes) <= self.clicks.num_items)) and (max(item_samp_sizes) <= self.clicks.num_items)), "Too many distinct itemIDs requested."
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
            # Filter sessionIDs and itemIDs in the sample using the Bloom filters
            clicks = YoochooseClicks(self.clicks[self.clicks['sessionID'].map(lambda x : x in session_filter) & self.clicks['itemID'].map(lambda x : x in item_filter)])
            buys = YoochooseBuys(self.buys[self.buys['sessionID'].map(lambda x : x in session_filter) & self.buys['itemID'].map(lambda x : x in item_filter)])
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
    def make_features(self, prefix = None):
        """Creates a DataFrame of features for each session, and a DataFrame of features for each item. Saves resulting DataFrame to two CSV files (one for session features, one for item features), if filename prefix is given. The features for sessions are:
            1) sessionID, 
            2) numClicks: Number of clicks during session, 
            3) numBuys: Number of buys during session, 
            4) bought: Whether something was bought during session, 
            5) buyRatio: Ratio of items bought to items clicked, 
            6) duration: Duration of session in minutes (0 if only one click), 
            7) minInterval: Minimum mins. between clicks, 
            8) maxInterval: Maximum mins. between clicks, 
            9) meanInterval: Mean mins. between clicks,
            10) cvInterval: stdev / mean of mins. between clicks,
            11) Mid time (average of first and last click timestamps), 
            12) Month of the year (1 to 12), 
            13) Day of the week (0=Monday, 6=Sunday), 
            14) Day of the year (1 to 365), 
            15) Hour of the day (0 to 23), 
            16) Max number of times an item was clicked in session, 
            17) Number of items clicked more than once, 
            18) Proportion of items clicked more than once, 
            19) Whether a special offer was clicked during session. The last three entries are based on the mid time.
        The features for items are:
            1) Total number of clicks on item, 
            2) Total number of times item was bought, 
            3) Ratio of items bought to items clicked."""
        timestamp_dict = defaultdict(list)      # sorted list of click timestamps for each session
        items_clicked_dict = defaultdict(list)  # list of items clicked on for each session
        num_clicks_dict = defaultdict(int)  # number of clicks for each session or item
        num_buys_dict = defaultdict(int)    # number of buys for each session or item
        clicked_special_dict = defaultdict(bool)    # flags for whether a special offer was clicked during each session
        self.session_features = pd.DataFrame(columns = ['sessionID', 'numClicks', 'numBuys', 'bought', 'buyRatio', 'duration', 'minInterval', 'maxInterval', 'meanInterval', 'cvInterval', 'midTime', 'monthOfYear', 'dayOfWeek', 'dayOfYear', 'hourOfDay', 'maxRepeatClicks', 'numRepeatItems', 'ratioRepeatItems', 'clickedSpecial'])
        self.session_features['sessionID'] = self.clicks.sessionIDs
        self.item_features = pd.DataFrame(columns = ['itemID', 'numClicks', 'numBuys'])
        self.item_features['itemID'] = self.clicks.itemIDs
        for row in self.clicks.itertuples():
            sessionID, timestamp, itemID, category = row[1], row[2], row[3], row[4]
            timestamp_dict[sessionID].append(timestamp)
            items_clicked_dict[sessionID].append(itemID)
            clicked_special_dict[sessionID] |= (category == 'S')
            num_clicks_dict[sessionID] += 1
            num_clicks_dict[itemID] += 1
        session_index, item_index = list(self.buys.columns).index('sessionID') + 1, list(self.buys.columns).index('itemID') + 1
        for row in self.buys.itertuples():
            sessionID, itemID = row[session_index], row[item_index]
            num_buys_dict[sessionID] += 1
            num_buys_dict[itemID] += 1
        for (i, sessionID) in enumerate(self.clicks.sessionIDs):
            num_clicks = num_clicks_dict[sessionID]
            num_buys = num_buys_dict[sessionID]
            timestamps = timestamp_dict[sessionID]
            timestamps.sort()
            duration = 0.0 if (len(timestamps) <= 1) else (timestamps[-1] - timestamps[0]).total_seconds() / 60.0
            intervals = [(timestamps[i + 1] - timestamps[i]).total_seconds() / 60.0 for i in range(len(timestamps) - 1)]
            min_interval = 0.0 if (len(intervals) == 0) else min(intervals)
            max_interval = 0.0 if (len(intervals) == 0) else max(intervals)
            mean_interval = 0.0 if (len(intervals) == 0) else np.mean(intervals)
            cv_interval = 0.0 if (duration < ZERO_THRESH) else np.std(intervals) / mean_interval  # coefficient of variation
            mid_time = timestamps[0] + (timestamps[-1] - timestamps[0]) / 2
            items_clicked = items_clicked_dict[sessionID]
            counts = [items_clicked.count(item) for item in set(items_clicked)]
            max_repeat_clicks = max(counts)  # max number of times the same item was clicked
            num_repeat_items = len([count for count in counts if (count > 1)])  # number of items clicked more than once
            ratio_repeat_items = float(num_repeat_items) / len(counts)  # ratio of items clicked more than once
            self.session_features.set_value(i, 'numClicks', num_clicks)
            self.session_features.set_value(i, 'numBuys', num_buys)
            self.session_features.set_value(i, 'bought', num_buys > 0)
            self.session_features.set_value(i, 'buyRatio', float(num_buys) / num_clicks)
            self.session_features.set_value(i, 'duration', duration)
            self.session_features.set_value(i, 'minInterval', min_interval)
            self.session_features.set_value(i, 'maxInterval', max_interval)
            self.session_features.set_value(i, 'meanInterval', mean_interval)
            self.session_features.set_value(i, 'cvInterval', cv_interval)
            self.session_features.set_value(i, 'midTime', mid_time)
            self.session_features.set_value(i, 'monthOfYear', mid_time.month)
            self.session_features.set_value(i, 'dayOfWeek', mid_time.dayofweek)
            self.session_features.set_value(i, 'dayOfYear', mid_time.dayofyear)
            self.session_features.set_value(i, 'hourOfDay', mid_time.hour)
            self.session_features.set_value(i, 'maxRepeatClicks', max_repeat_clicks)
            self.session_features.set_value(i, 'numRepeatItems', num_repeat_items)
            self.session_features.set_value(i, 'ratioRepeatItems', ratio_repeat_items)
            self.session_features.set_value(i, 'clickedSpecial', clicked_special_dict[sessionID])
        for (i, itemID) in enumerate(self.clicks.itemIDs):
            num_clicks = num_clicks_dict[itemID]
            num_buys = num_buys_dict[itemID]
            self.item_features.set_value(i, 'numClicks', num_clicks)
            self.item_features.set_value(i, 'numBuys', num_buys)
            self.item_features.set_value(i, 'bought', num_buys > 0)
            self.item_features.set_value(i, 'buyRatio', float(num_buys) / num_clicks)
        if prefix:
            self.session_features.to_csv(prefix + '-session_features.csv', index = False)
            self.item_features.to_csv(prefix + '-item_features.csv', index = False)
    def build_click_cliques(self, prefix = None, load = True):
        """Makes dictionary of cliques (sessions that all clicked on the same item) by mapping item IDs to lists of session IDs. Saves the result as a CSV file (itemID;session1,session2... for each line) if filename prefix is specified."""
        success = False
        if load:
            try:
                print("Loading cliques...")
                self.click_cliques = csv_to_dict(prefix + 'click_cliques.csv', key_type = int, val_type = int)
                success = True
            except:
                print("Failed to load cliques from file.")
        if (not success):
            self.click_cliques = defaultdict(list)
            for row in self.clicks.itertuples():
                sessionID, itemID = row[1], row[3]
                self.click_cliques[itemID].append(sessionID)
            if prefix:
                dict_to_csv(prefix + '-click_cliques.csv', self.click_cliques)
    def build_graphs(self):
        """Builds bipartite graphs of the data using networkx, one for clicks and one for buys. An edge exists between a session and an item if a click or buy occurred for that session/item pair. 
        Vertex attributes for the click graph are the same as in 'make_session_features'.
        Edge attributes for the click graph are: 1) Timestamp, 2) Category.
        Vertices for the buy graph are the same as for the click graph, even if they have degree zero.
        Edge attributes for the buy graph are: 1) Timestamp, 2) Price, 3) Quantity."""
        self.click_graph = nx.Graph()
        self.click_graph.add_nodes_from(self.clicks.sessionIDs, type = 'Session')
        self.click_graph.add_nodes_from(self.clicks.itemIDs, type = 'Item')
        self.buy_graph = nx.Graph()
        self.buy_graph.add_nodes_from(self.click_graph.nodes())
        for (i, sessionID, timestamp, itemID, category) in self.clicks.itertuples():
            self.click_graph.add_edge(sessionID, itemID, timestamp = str(timestamp), category = category)
        for (i, sessionID, timestamp, itemID, price, quantity) in self.buys.itertuples():
            self.buy_graph.add_edge(sessionID, itemID, timestamp = str(timestamp), price = int(price), quantity = int(quantity))
        # if hasattr(self, 'session_features'):  # merge in the vertex attributes from Data Frame
        #     for row in self.session_features.itertuples():
        #         sessionID = row[1]
        #         session_attr_dict = dict(zip(self.session_features.columns[1:], row[2:]))
        #         self.buy_graph.node[sessionID].update(session_attr_dict)
        #         del(session_attr_dict['numBuys'])  # hide this from the click graph
        #         self.click_graph.node[sessionID].update(session_attr_dict)
        #     for row in self.item_features.itertuples():
        #         itemID = row[1]
        #         item_attr_dict = dict(zip(self.item_features.columns[1:], row[2:]))
        #         self.buy_graph.node[itemID].update(item_attr_dict)
        #         del(item_attr_dict['numBuys'])
        #         self.click_graph.node[itemID].update(item_attr_dict)
    def save_graphs(prefix = 'data/yoochoose', format = 'gml'):
        """Save the click and buy graphs to separate files. gml (GraphML) is the recommended format."""
        if (format == 'gml'):
            nx.write_gml(self.click_graph, prefix + '-clicks.gml')
            nx.write_gml(self.buy_graph, prefix + '-buys.gml')
        else:
            raise IOError("Invalid file format '%s'" % format)
    @classmethod
    def from_training(cls, prefix = 'data/yoochoose'):
        """Loads clicks and buys from training data files."""
        return cls(YoochooseClicks.from_csv(prefix + '-clicks.dat'), YoochooseBuys.from_csv(prefix + '-buys.dat'))
    @classmethod
    def from_solution(cls, test_filename = heldout_filename, solution_filename = solution_filename):
        """Loads clicks and buys from heldout test set."""
        return cls(YoochooseClicks.from_csv(heldout_filename), YoochooseBuys.from_solution(solution_filename))
