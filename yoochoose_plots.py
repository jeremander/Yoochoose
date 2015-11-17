from yoochoose import *
from ggplot import *

click_comp_sizes = pd.read_csv('data/yoochoose-click_comp_sizes.csv')
buy_comp_sizes = pd.read_csv('data/yoochoose-buy_comp_sizes.csv')
session_features = pd.read_csv('data/yoochoose-session_features.csv')
item_features = pd.read_csv('data/yoochoose-item_features.csv')
session_features['Bought'] = session_features['numBuys'] > 0
item_features['Bought'] = item_features['numBuys'] > 0
session_features['Buy Ratio'] = session_features['numBuys'] / session_features['numClicks']
item_features['Buy Ratio'] = item_features['numBuys'] / item_features['numClicks']

num_sessions = 9249729
num_buy_sessions = 509696

session_click_plot = ggplot(aes(x = 'numClicks'), data = session_features) + geom_histogram(binwidth = 5.) + scale_y_log10() + ggtitle("Clicks by session") + xlab("# of clicks")
session_buy_plot = ggplot(aes(x = 'numBuys'), data = session_features) + geom_histogram(binwidth = 2.5) + scale_y_log10() + ggtitle("Buys by session") + xlab("# of buys")
session_buy_ratio_plot = ggplot(aes(x = 'Buy Ratio'), data = session_features) + geom_histogram() + scale_y_log10() + ggtitle("Session buy ratio") + xlab("Buys/clicks")
item_click_plot = ggplot(aes(x = 'numClicks'), data = item_features) + geom_histogram(binwidth = 2000.) + scale_y_log10() + ggtitle("Clicks by item") + xlab("# of clicks")
item_buy_plot = ggplot(aes(x = 'numBuys'), data = item_features) + geom_histogram(binwidth = 250.) + scale_y_log10() + ggtitle("Buys by item") + xlab("# of buys")
item_buy_ratio_plot = ggplot(aes(x = 'Buy Ratio'), data = item_features) + geom_histogram() + scale_y_log10() + ggtitle("Item buy ratio") + xlab("Buys/clicks")

session_duration_plot = ggplot(aes(x = 'Duration', fill = 'Bought'), data = session_features) + geom_histogram(binwidth = 75.) + scale_y_log10() + ggtitle("Session durations") + xlab("Duration (mins.)")
duration_df = session_features[['Duration', 'Buy Ratio']]
bins = 10 ** np.linspace(-6.0, 6.0, 500)
points = duration_df.groupby(np.digitize(duration_df['Duration'], bins)).mean()
points['log10Duration'] = points['Duration'].map(lambda x : max(-5.0, np.log10(x)))
session_duration_plot2 = ggplot(aes(x = 'log10Duration', y = 'Buy Ratio'), data = points) + geom_point() + ggtitle("Buy ratio vs. session duration") + scale_x_continuous(breaks = list(np.arange(-5.0, 5.0, 1.0)), labels = [0] + [10 ** x for x in range(-4, 5)]) + xlim(low = -5.5, high = 4.5) + xlab("Duration (mins.)") + ylab("Buys/clicks")

weekdays = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
day_of_week_plot = ggplot(aes(x = 'Day of Week', fill = 'Bought'), data = session_features) + geom_histogram(width = 0.8) + scale_x_discrete(breaks = range(7), labels = weekdays) + xlim(low = -0.5, high = 6.5) + ggtitle("Days of the week")

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
days_of_year_by_month = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]  # first of month in normal 365-day year
day_of_year_plot = ggplot(aes(x = 'Day of Year', fill = 'Bought'), data = session_features) + geom_histogram(binwidth = 7.) + ggtitle("Weeks of the year") + scale_x_continuous(breaks = days_of_year_by_month, labels = months) + xlab("Week of year") + xlim(low = 80, high = 280)

time_of_day_plot = ggplot(aes(x = 'Hour of Day', fill = 'Bought'), data = session_features) + geom_histogram(binwidth = 1.) + ggtitle("Time of day") + xlab("Hour")
