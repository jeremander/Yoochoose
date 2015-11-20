from yoochoose import *
from ggplot import *

data_prefix = 'data/yoochoose'

click_comp_sizes = pd.read_csv(data_prefix + '-click_comp_sizes.csv')
buy_comp_sizes = pd.read_csv(data_prefix + '-buy_comp_sizes.csv')
session_features = pd.read_csv(data_prefix + '-session_features.csv')
item_features = pd.read_csv(data_prefix + '-item_features.csv')

num_sessions = 9249729
num_buy_sessions = 509696

session_click_plot = ggplot(aes(x = 'numClicks'), data = session_features) + geom_histogram(binwidth = 5.) + scale_y_log10() + ggtitle("Clicks by session") + xlab("# of clicks")
session_buy_plot = ggplot(aes(x = 'numBuys'), data = session_features) + geom_histogram(binwidth = 2.5) + scale_y_log10() + ggtitle("Buys by session") + xlab("# of buys")
session_buy_ratio_plot = ggplot(aes(x = 'buyRatio'), data = session_features) + geom_histogram() + scale_y_log10() + ggtitle("Session buy ratio") + xlab("Buys/clicks")
item_click_plot = ggplot(aes(x = 'numClicks'), data = item_features) + geom_histogram(binwidth = 2000.) + scale_y_log10() + ggtitle("Clicks by item") + xlab("# of clicks")
item_buy_plot = ggplot(aes(x = 'numBuys'), data = item_features) + geom_histogram(binwidth = 250.) + scale_y_log10() + ggtitle("Buys by item") + xlab("# of buys")
item_buy_ratio_plot = ggplot(aes(x = 'buyRatio'), data = item_features) + geom_histogram() + scale_y_log10() + ggtitle("Item buy ratio") + xlab("Buys/clicks")

session_duration_plot = ggplot(aes(x = 'duration', fill = 'bought'), data = session_features) + geom_histogram(binwidth = 75.) + scale_y_log10() + ggtitle("Session durations") + xlab("Duration (mins.)")
duration_df = session_features[['duration', 'buyRatio']]
bins = 10 ** np.linspace(-6.0, 6.0, 500)
points = duration_df.groupby(np.digitize(duration_df['duration'], bins)).mean()
points['log10Duration'] = points['duration'].map(lambda x : max(-5.0, np.log10(x)))
session_duration_plot2 = ggplot(aes(x = 'log10Duration', y = 'buyRatio'), data = points) + geom_point() + ggtitle("Buy ratio vs. session duration") + scale_x_continuous(breaks = list(np.arange(-5.0, 5.0, 1.0)), labels = [0] + [10 ** x for x in range(-4, 5)]) + xlim(low = -5.5, high = 4.5) + xlab("Duration (mins.)") + ylab("Buys/clicks")

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
days_of_year_by_month = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]  # first of month in normal 365-day year

month_of_year_plot = ggplot(aes(x = 'monthOfYear', fill = 'bought'), data = session_features) + geom_histogram(width = 1.0) + scale_x_discrete(breaks = range(1, 13), labels = months) + xlim(low = 3.5, high = 9.5) + ggtitle("Months of the year")
month_of_year_plot2 = ggplot(aes(x = 'monthOfYear', y = 'buyRatio', width = 0.8), data = session_features) + geom_bar(stat = "summary", fun_y = np.mean) + scale_x_discrete(breaks = range(1, 13), labels = months) + xlim(low = 3.5, high = 9.5) + ggtitle("Months of the year") + ylab("mean buys/clicks")

day_of_week_plot = ggplot(aes(x = 'dayOfWeek', fill = 'bought'), data = session_features) + geom_histogram(width = 0.8) + scale_x_discrete(breaks = range(7), labels = weekdays) + xlim(low = -0.5, high = 6.5) + ggtitle("Days of the week")
day_of_week_plot2 = ggplot(aes(x = 'dayOfWeek', y = 'buyRatio', width = 0.8), data = session_features) + geom_bar(stat = "summary", fun_y = np.mean) + scale_x_discrete(breaks = range(7), labels = weekdays) + xlim(low = -0.5, high = 6.5) + ggtitle("Days of the week") + ylab("mean buys/clicks")

week_of_year_plot = ggplot(aes(x = 'dayOfYear', fill = 'bought'), data = session_features) + geom_histogram(binwidth = 7.) + ggtitle("Weeks of the year") + scale_x_continuous(breaks = days_of_year_by_month, labels = months) + xlab("") + xlim(low = 80, high = 280)
bins = np.arange(89.0, 277.0, 7.0)
points = session_features[['buyRatio', 'dayOfYear']].groupby(np.digitize(session_features['dayOfYear'], bins)).mean()
week_of_year_plot2 = ggplot(aes(x = 'dayOfYear', y = 'buyRatio'), data = points) + geom_line() + scale_x_continuous(breaks = days_of_year_by_month, labels = months) + xlab("") + xlim(low = 80, high = 280) + ylab("mean buys/clicks")

day_of_year_plot = ggplot(aes(x = 'dayOfYear', fill = 'bought'), data = session_features) + geom_histogram(binwidth = 1) + ggtitle("Days of the year") + scale_x_continuous(breaks = days_of_year_by_month, labels = months) + xlab("") + xlim(low = 80, high = 280)
day_of_year_plot2 = ggplot(aes(x = 'dayOfYear', y = 'buyRatio'), data = session_features) + geom_line(stat = "summary", fun_y = np.mean) + scale_x_continuous(breaks = days_of_year_by_month, labels = months) + xlab("") + ylab("mean buys/clicks") + xlim(low = 80, high = 280) + ggtitle("Days of the year")

time_of_day_plot = ggplot(aes(x = 'hourOfDay', fill = 'bought'), data = session_features) + geom_histogram(binwidth = 1) + ggtitle("Time of day") + xlab("Hour")
time_of_day_plot2 = ggplot(aes(x = 'hourOfDay', y = 'buyRatio', width = 0.8), data = session_features) + geom_bar(stat = "summary", fun_y = np.mean) + scale_x_discrete(breaks = range(24), labels = range(24)) + xlim(-1, 24) + xlab("Hour") + ggtitle("Time of day") + ylab("mean buys/clicks")

num_clicks_plot = ggplot(aes(x = 'numClicks', fill = 'bought'), data = session_features) + geom_histogram(binwidth = 1) + xlab("number of clicks") + scale_y_log10()
num_clicks_plot2 = ggplot(aes(x = 'numClicks', y = 'buyRatio', width = 0.8), data = session_features) + geom_line(stat = "summary", fun_y = np.mean) + xlab("number of clicks") + ylab("mean buys/clicks")

ggsave('plots/session_click_hist', session_click_plot)
ggsave('plots/session_buy_hist', session_buy_plot)
ggsave('plots/session_buy_ratio_hist', session_buy_ratio_plot)
ggsave('plots/item_click_hist', item_click_plot)
ggsave('plots/item_buy_hist', item_buy_plot)
ggsave('plots/item_buy_ratio_hist', item_buy_ratio_plot)
ggsave('plots/session_duration_hist', session_duration_plot)
ggsave('plots/session_buy_ratio_vs_duration', session_duration_plot2)
ggsave('plots/month_hist', month_of_year_plot)
ggsave('plots/session_buy_ratio_vs_month', month_of_year_plot2)
ggsave('plots/weekday_hist', day_of_week_plot)
ggsave('plots/session_buy_ratio_vs_weekday', day_of_week_plot2)
ggsave('plots/week_hist', week_of_year_plot)
ggsave('plots/session_buy_ratio_vs_week', week_of_year_plot2)
ggsave('plots/day_hist', day_of_year_plot)
ggsave('plots/session_buy_ratio_vs_day', day_of_year_plot2)
ggsave('plots/time_hist', time_of_day_plot)
ggsave('plots/session_buy_ratio_vs_time', time_of_day_plot2)
ggsave('plots/num_clicks_hist', num_clicks_plot)
ggsave('plots/session_buy_ratio_vs_num_clicks', num_clicks_plot2)
