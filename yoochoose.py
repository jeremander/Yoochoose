import pandas as pd
import numpy as np
import networkx as nx
import igraph as ig
import os
import pickle
import tempfile
import louvain
from pybloom import BloomFilter
from importlib import reload
from collections import defaultdict
from scipy.stats import chisqprob
from autoreadwrite import *

ZERO_THRESH = 1e-12

# total clicks:          33003944
# unique click sessions:  9249729
# total buys:             1150753
# unique buy sessions:     509696
# unique clicked itemIDs:  52739
# unique bought itemIDs:   19949
# purchases with price & quantity == 0:  610030

def safe_divide(num, den):
    """Floating point division, with the convention that 0 / 0 = 0."""
    return 0.0 if (num == 0.0) else (num / den)

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
    def from_csv(cls, filename):
        return cls(pd.read_csv(filename, header = None, names = ['sessionID', 'timestamp', 'itemID', 'category'], dtype = {'category' : str}, parse_dates = [1]))

class YoochooseBuys(YoochooseClicksOrBuys):
    """Data frame subclass for buy events."""
    @classmethod
    def from_csv(cls, filename):
        return cls(pd.read_csv(filename, header = None, names = ['sessionID', 'timestamp', 'itemID', 'price', 'quantity'], parse_dates = [1]))
    @classmethod
    def from_solution(cls, filename):
        buys_by_session = csv_to_dict(filename, key_type = int, val_type = int)
        rows_list = []
        for sessionID in sorted(buys_by_session.keys()):
            for itemID in sorted(buys_by_session[sessionID]):
                rows_list.append([sessionID, itemID])
        return cls(pd.DataFrame(rows_list, columns = ['sessionID', 'itemID']))

class Yoochoose(ObjectWithReadwriteProperties):
    """Class encapsulating the click events and buy events from the Yoochoose data set."""
    readwrite_properties = {'session_features' : 'csv', 'item_features' : 'csv', 'timestamps_by_session' : 'pickle', 'items_clicked_by_session' : 'pickle', 'sessions_by_item_clicked' : 'pickle', 'nx_click_graph' : 'nx.edges', 'ig_click_graph' : 'ig.edges', 'ig_to_nx_labels' : 'csv', 'louvain_memberships' : 'csv', 'community_features' : 'csv'}
    def __init__(self, clicks, buys, folder = 'yoochoose/data'):
        """Constructs Yoochoose object from YoochooseClicks and YoochooseBuys, with an optional data folder name for saving and loading data."""
        super().__init__(folder)
        self.clicks = clicks
        self.buys = buys
    @autoreadwrite(['session_features', 'item_features', 'timestamps_by_session', 'items_clicked_by_session', 'sessions_by_item_clicked'], ['csv', 'csv', 'pickle', 'pickle', 'pickle'])
    def make_features(self, save = False):
        """Creates a DataFrame of features for each session, and a DataFrame of features for each item. Saves resulting DataFrame to two CSV files (one for session features, one for item features), if filename prefix is given. The features for sessions are:
            sessionID, 
            numClicks: Number of clicks during session, 
            numItemsClicked: Number of items clicked during session,
            numBuys: Number of buys during session, 
            numItemsBought: Number of items bought during session,
            bought: Whether something was bought during session, 
            buyRatio: Ratio of buy events to click events,
            boughtItemRatio: Ratio of bought items to clicked items,
            duration: Duration of session in minutes (0 if only one click), 
            minInterval: Minimum mins. between clicks, 
            maxInterval: Maximum mins. between clicks, 
            meanInterval: Mean mins. between clicks,
            cvInterval: stdev / mean of mins. between clicks,
            midTime: Mid time (average of first and last click timestamps), 
            monthOfYear: Month of the year (1 to 12), 
            dayOfWeek: Day of the week (0=Monday, 6=Sunday), 
            dayOfYear: Day of the year (1 to 365), 
            hourOfDay: Hour of the day (0 to 23), 
            maxRepeatClicks: Max number of times an item was clicked in session, 
            numRepeatItems: Number of items clicked more than once, 
            ratioRepeatItems: Proportion of items clicked more than once, 
            clickedSpecial: Whether a special offer was clicked during session. The last three entries are based on the mid time.
        The features for items are:
            numClicks: Total number of clicks on item, 
            numClickers: Total number of sessions that clicked on item,
            numBuys: Total number of times item was bought, 
            numBuyers: Total number of sessions that bought item,
            buyRatio: Ratio of buy events to click events,
            buyerRatio: Ratio of sessions that bought to sessions that clicked."""
        self._timestamps_by_session = defaultdict(list)      # sorted list of click timestamps for each session
        self._items_clicked_by_session = defaultdict(list)  # set of items clicked on for each session
        self._sessions_by_item_clicked = defaultdict(list)  # set of sessions that clicked on each item
        items_bought_by_session = defaultdict(list)  # set of items bought for each session
        sessions_by_item_bought = defaultdict(list)  # set of sessions that bought each item
        clicked_special_dict = defaultdict(bool)    # flags for whether a special offer was clicked during each session
        self._session_features = pd.DataFrame(columns = ['sessionID', 'numClicks', 'numBuys', 'numItemsClicked', 'numItemsBought', 'didBuy', 'buyRatio', 'boughtItemRatio', 'duration', 'minInterval', 'maxInterval', 'meanInterval', 'cvInterval', 'midTime', 'monthOfYear', 'dayOfWeek', 'dayOfYear', 'hourOfDay', 'maxRepeatClicks', 'numRepeatItems', 'ratioRepeatItems', 'clickedSpecial'])
        self._session_features['sessionID'] = self.clicks.sessionIDs
        self._item_features = pd.DataFrame(columns = ['itemID', 'numClicks', 'numClickers', 'numBuys', 'numBuyers', 'wasBought', 'buyRatio', 'buyerRatio'])
        self._item_features['itemID'] = self.clicks.itemIDs
        for row in self.clicks.itertuples():
            (i, sessionID, timestamp, itemID, category) = row
            self._timestamps_by_session[sessionID].append(timestamp)
            self._items_clicked_by_session[sessionID].append(itemID)
            self._sessions_by_item_clicked[itemID].append(sessionID)
            clicked_special_dict[sessionID] |= (category == 'S')
        session_index, item_index = list(self.buys.columns).index('sessionID') + 1, list(self.buys.columns).index('itemID') + 1
        for row in self.buys.itertuples():
            sessionID, itemID = row[session_index], row[item_index]
            items_bought_by_session[sessionID].append(itemID)
            sessions_by_item_bought[itemID].append(sessionID)
        for (i, sessionID) in enumerate(self.clicks.sessionIDs):
            timestamps = self._timestamps_by_session[sessionID]
            timestamps.sort()
            duration = 0.0 if (len(timestamps) <= 1) else (timestamps[-1] - timestamps[0]).total_seconds() / 60.0
            intervals = [(timestamps[i + 1] - timestamps[i]).total_seconds() / 60.0 for i in range(len(timestamps) - 1)]
            min_interval = 0.0 if (len(intervals) == 0) else min(intervals)
            max_interval = 0.0 if (len(intervals) == 0) else max(intervals)
            mean_interval = 0.0 if (len(intervals) == 0) else np.mean(intervals)
            cv_interval = 0.0 if (duration < ZERO_THRESH) else np.std(intervals) / mean_interval  # coefficient of variation
            mid_time = timestamps[0] + (timestamps[-1] - timestamps[0]) / 2
            items_clicked = self._items_clicked_by_session[sessionID]
            num_clicks = len(items_clicked)
            num_items_clicked = len(set(items_clicked))
            counts = [items_clicked.count(item) for item in set(items_clicked)]
            max_repeat_clicks = max(counts)  # max number of times the same item was clicked
            num_repeat_items = len([count for count in counts if (count > 1)])  # number of items clicked more than once
            ratio_repeat_items = float(num_repeat_items) / len(counts)  # ratio of items clicked more than once
            items_bought = items_bought_by_session[sessionID]
            num_buys = len(items_bought)
            num_items_bought = len(set(items_bought))
            self._session_features.set_value(i, 'numClicks', num_clicks)
            self._session_features.set_value(i, 'numBuys', num_buys)
            self._session_features.set_value(i, 'numItemsClicked', num_items_clicked)
            self._session_features.set_value(i, 'numItemsBought', num_items_bought)
            self._session_features.set_value(i, 'didBuy', num_buys > 0)
            self._session_features.set_value(i, 'buyRatio', float(num_buys) / num_clicks)
            self._session_features.set_value(i, 'boughtItemRatio', float(num_items_bought) / num_items_clicked)
            self._session_features.set_value(i, 'duration', duration)
            self._session_features.set_value(i, 'minInterval', min_interval)
            self._session_features.set_value(i, 'maxInterval', max_interval)
            self._session_features.set_value(i, 'meanInterval', mean_interval)
            self._session_features.set_value(i, 'cvInterval', cv_interval)
            self._session_features.set_value(i, 'midTime', mid_time)
            self._session_features.set_value(i, 'monthOfYear', mid_time.month)
            self._session_features.set_value(i, 'dayOfWeek', mid_time.dayofweek)
            self._session_features.set_value(i, 'dayOfYear', mid_time.dayofyear)
            self._session_features.set_value(i, 'hourOfDay', mid_time.hour)
            self._session_features.set_value(i, 'maxRepeatClicks', max_repeat_clicks)
            self._session_features.set_value(i, 'numRepeatItems', num_repeat_items)
            self._session_features.set_value(i, 'ratioRepeatItems', ratio_repeat_items)
            self._session_features.set_value(i, 'clickedSpecial', clicked_special_dict[sessionID])
        for (i, itemID) in enumerate(self.clicks.itemIDs):
            sessions_clicking = self.sessions_by_item_clicked[itemID]
            sessions_buying = sessions_by_item_bought[itemID]
            num_clicks = len(sessions_clicking)
            num_buys = len(sessions_buying)
            num_clickers = len(set(sessions_clicking))
            num_buyers = len(set(sessions_buying))
            self._item_features.set_value(i, 'numClicks', num_clicks)
            self._item_features.set_value(i, 'numClickers', num_clickers)
            self._item_features.set_value(i, 'numBuys', num_buys)
            self._item_features.set_value(i, 'numBuyers', num_buyers)
            self._item_features.set_value(i, 'wasBought', num_buys > 0)
            self._item_features.set_value(i, 'buyRatio', float(num_buys) / num_clicks)
            self._item_features.set_value(i, 'buyerRatio', float(num_buyers) / num_clickers)
    @autoreadwrite(['nx_click_graph'], ['nx.edges'])
    def make_nx_click_graph(self, save = False):
        """Makes bipartite graph of the click data using iGraph. An edge exists between a session and an item if a click occurred for that session/item pair."""
        assert hasattr(self, 'items_clicked_by_session')
        self._nx_click_graph = nx.Graph()
        for (i, sessionID) in enumerate(self.clicks.sessionIDs):
            self._nx_click_graph.add_edges_from(((sessionID, itemID) for itemID in self._items_clicked_by_session[sessionID]))
    @autoreadwrite(['ig_click_graph', 'ig_to_nx_labels'], ['ig.edges', 'csv'])
    def make_ig_click_graph(self, save = False):
        assert hasattr(self, '_nx_click_graph')
        self._ig_to_nx_labels = pd.DataFrame(list(self._nx_click_graph.nodes_iter()), columns = ['nxLabel'])
        nx_to_ig_labels = dict((ID, i) for (i, ID) in enumerate(self._ig_to_nx_labels['nxLabel']))
        with tempfile.TemporaryFile(mode = 'w+') as f:
            for edge in self._nx_click_graph.edges_iter():
                f.write("%d %d\n" % (nx_to_ig_labels[edge[0]], nx_to_ig_labels[edge[1]]))
            self._ig_click_graph = ig.Graph.Read_Edgelist(f)
    @autoreadwrite(['louvain_memberships'], ['csv'])
    def louvain(self, save = False):
        """Computes cluster memberships returned by the Louvain method (implemented in C++ via louvain-igraph package)."""
        assert hasattr(self, '_ig_click_graph')
        self._louvain_memberships = pd.DataFrame(columns = ['louvainMembership'])
        self._louvain_memberships['louvainMembership'] = louvain.find_partition(self._ig_click_graph, method = 'Modularity').membership
    @autoreadwrite(['community_features', 'session_community_features', 'item_community_features'], ['csv', 'csv', 'csv'])
    def louvain_statistics(self, save = False):
        assert all([hasattr(self, attr) for attr in ['_session_features', '_item_features', '_ig_to_nx_labels', '_louvain_memberships']])
        memberships = self._louvain_memberships['louvainMembership']
        num_communities = max(memberships) + 1
        nx_to_ig_labels = dict((int(ID), i) for (i, ID) in enumerate(self._ig_to_nx_labels['nxLabel']))
        self._community_features = pd.DataFrame()
        self._community_features['louvainCommunity'] = list(range(num_communities))
        num_sessions, num_buyers, num_items, num_bought_items = np.zeros(num_communities, dtype = int), np.zeros(num_communities, dtype = int), np.zeros(num_communities, dtype = int), np.zeros(num_communities, dtype = int)
        sessionID_index = list(self._session_features.columns).index('sessionID') + 1
        did_buy_index = list(self._session_features.columns).index('didBuy') + 1
        for row in self._session_features.itertuples():
            sessionID, did_buy = row[sessionID_index], row[did_buy_index]
            community = memberships[nx_to_ig_labels[sessionID]]
            num_sessions[community] += 1
            if did_buy:
                num_buyers[community] += 1
        self._community_features['numSessions'] = num_sessions
        self._community_features['numBuyers'] = num_buyers
        itemID_index = list(self._item_features.columns).index('itemID') + 1
        was_bought_index = list(self._item_features.columns).index('wasBought') + 1
        for row in self._item_features.itertuples():
            itemID, was_bought = row[itemID_index], row[was_bought_index]
            community = memberships[nx_to_ig_labels[itemID]]
            num_items[community] += 1
            if was_bought:
                num_bought_items[community] += 1
        self._community_features['numItems'] = num_items
        self._community_features['numBoughtItems'] = num_bought_items
        self._community_features['communitySize'] = num_sessions + num_items
        self._community_features['buyerRatio'] = np.vectorize(safe_divide)(num_buyers, num_sessions)
        self._community_features['boughtItemRatio'] = np.vectorize(safe_divide)(num_bought_items, num_items)
    def __len__(self):
        return len(self.vs)
    # def item_buy_chisq(self):
    #     """Performs chi^2 test for buy status within each session clique (each of which corresponds to an item)."""
    #     assert hasattr(self, 'item_features')
    #     f_obs = pd.DataFrame(columns = ['numBuyers', 'numNonbuyers'])
    #     f_obs['numBuyers'] = self._item_features['numBuyers']
    #     f_obs['numNonbuyers'] = self._item_features['numClickers'] - self.item_features['numBuyers']
    #     f_obs = np.asarray(f_obs, dtype = int)
    #     rowsums = np.matrix(np.sum(f_obs, axis = 1)).transpose()
    #     p = np.sum(f_obs[:, 0]) / np.sum(f_obs)  # proportion bought in overall population
    #     p_mat = np.matrix([p, 1 - p])
    #     f_exp = rowsums * p_mat
    #     chisq = np.sum(np.power(f_obs - f_exp, 2) / f_exp)
    #     dof = len(f_obs) - 1
    #     pval = chisqprob(chisq, dof)
    #     print("chi^2 = %f\ndof = %d\npval = %f" % (chisq, dof, pval))
    #     return (chisq, dof, pval)
    @classmethod
    def from_data(cls, folder = 'yoochoose/data'):
        """Loads clicks and buys from training data files."""
        print("Loading Yoochoose data...")
        start_time = time.time()
        y = cls(YoochooseClicks.from_csv(folder + '/clicks.dat'), YoochooseBuys.from_csv(folder + '/buys.dat'), folder)
        print(time_format(time.time() - start_time))
        return y
    @classmethod
    def from_solution(cls, folder = 'yoochoose/data'):
        """Loads clicks and buys from heldout test set."""
        print("Loading Yoochoose solution data...")
        start_time = time.time()
        y = cls(YoochooseClicks.from_csv(folder + '/heldout.dat'), YoochooseBuys.from_solution(folder + '/heldout_solution.dat'), folder)
        print(time_format(time.time() - start_time))
        return y
