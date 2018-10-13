from bayesian_regression import *
from pymongo import MongoClient
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
client = MongoClient()
database = client['okcoindb']
collection = database['historical_data']

# Retrieve price, v_ask, and v_bid data points from the database.
prices = []
dates = []
v_ask = []
v_bid = []
num_points = 777600
mini_num_points = 3 * 821
current_num_points = collection.count()
print('record num :', current_num_points)
if(current_num_points < mini_num_points):
    print('not engough data in data base')
for doc in collection.find().limit(num_points):
    # print(doc)
    d1 = doc['date']
    if(len(dates) > 0):
        d2 = dates[-1]
        d = (d1 -d2).seconds
        # time series interval（second）
        if(d > 60):
            # print(doc['date'])
            dates.append(doc['date'])
            prices.append(doc['price'])
            v_ask.append(doc['v_ask'])
            v_bid.append(doc['v_bid'])
    else:
         dates.append(doc['date'])
         prices.append(doc['price'])
         v_ask.append(doc['v_ask'])
         v_bid.append(doc['v_bid'])
print(len(dates))
[train_dates, test_dates] = np.array_split(dates, 2)
[dates1, dates2] = np.array_split(train_dates, 2)
# Divide prices into three, roughly equal sized, periods:
# prices1, prices2, and prices3.
[train_prices, test_prices] = np.array_split(prices, 2)
[prices1, prices2] = np.array_split(train_prices, 2)

# Divide v_bid into three, roughly equal sized, periods:
# v_bid1, v_bid2, and v_bid3.
[train_vbids, test_vbids] = np.array_split(v_bid, 2)
[v_bid1, v_bid2] = np.array_split(train_vbids, 2)

# Divide v_ask into three, roughly equal sized, periods:
# v_ask1, v_ask2, and v_ask3.
[train_vasks, test_vasks] = np.array_split(v_ask, 2)
[v_ask1, v_ask2] = np.array_split(train_vasks, 2)

# Use the first time period (prices1) to generate all possible time series of
# appropriate length (180, 360, and 720).
timeseries180 = generate_timeseries(prices1, 180)
timeseries360 = generate_timeseries(prices1, 360)
timeseries720 = generate_timeseries(prices1, 720)


# Cluster timeseries180 in 100 clusters using k-means, return the cluster
# centers (centers180), and choose the 20 most effective centers (s1).

centers180 = find_cluster_centers(timeseries180, 100)
s1 = choose_effective_centers(centers180, 20)

centers360 = find_cluster_centers(timeseries360, 100)
s2 = choose_effective_centers(centers360, 20)

centers720 = find_cluster_centers(timeseries720, 100)
s3 = choose_effective_centers(centers720, 20)



# Use the second time period to generate the independent and dependent
# variables in the linear regression model:
# Δp = w0 + w1 * Δp1 + w2 * Δp2 + w3 * Δp3 + w4 * r.
Dpi_r, Dp = linear_regression_vars(prices2, v_bid2, v_ask2, s1, s2, s3)

# Find the parameter values w (w0, w1, w2, w3, w4).
w = find_parameters_w(Dpi_r, Dp)

# Predict average price changes over the third time period.
dps = predict_dps(test_prices, test_vbids, test_vasks, s1, s2, s3, w)

#
evaluate_performance(test_prices, test_dates, dps, t=0.0001, step=1)

