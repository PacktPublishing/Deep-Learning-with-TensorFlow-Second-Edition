import tensorflow as tf
import pandas as pd
from collections import Counter
from tffm import TFFMClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import os
from tensorflow.python.framework import ops
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Loading datasets
# Download and mention the respective paths

buys = open('yoochoose-buys.dat', 'r')
clicks = open('yoochoose-clicks.dat', 'r')

# Reading datasets
print("Reading datasets...")
initial_buys_df = pd.read_csv(buys, names=['Session ID', 'Timestamp', 'Item ID', 'Category', 'Quantity'],
                              dtype={'Session ID': 'float32', 'Timestamp': 'str', 'Item ID': 'int32',
                                     'Category': 'str'}) # read file into dataframe by column names

initial_clicks_df = pd.read_csv(clicks, names=['Session ID', 'Timestamp', 'Item ID', 'Category'],
                                dtype={'Item ID': 'int32','Category': 'str'})

print("Preprocessing data..")
# Make 'Session ID' column as index
initial_buys_df.set_index('Session ID', inplace=True)
initial_clicks_df.set_index('Session ID', inplace=True)

# We won't use timestamps in this example, remove 'Timestamp' column from dataframe(df)
initial_buys_df = initial_buys_df.drop('Timestamp', 1)
initial_clicks_df = initial_clicks_df.drop('Timestamp', 1)

# For illustrative purposes, we will only use a subset of the data: top 10000 buying users,
x = Counter(initial_buys_df.index).most_common(10000) # count top 10000 most common session ID's
top_k = dict(x).keys()                                # find respective keys

initial_buys_df = initial_buys_df[initial_buys_df.index.isin(top_k)]  # Assign the most common to df
initial_clicks_df = initial_clicks_df[initial_clicks_df.index.isin(top_k)]

# Create a copy of the index, since we will also apply one-hot encoding on the index
initial_buys_df['_Session ID'] = initial_buys_df.index

# One-hot encode all columns for buys
transformed_buys = pd.get_dummies(initial_buys_df)
transformed_clicks = pd.get_dummies(initial_clicks_df)

# Aggregate historical data for Items and Categories for buys
filtered_buys = transformed_buys.filter(regex="Item.*|Category.*")
filtered_clicks = transformed_clicks.filter(regex="Item.*|Category.*")

# groupby index for buy data and click data
historical_buy_data = filtered_buys.groupby(filtered_buys.index).sum()
historical_buy_data = historical_buy_data.rename(columns=lambda column_name: 'buy history:' + column_name)

historical_click_data = filtered_clicks.groupby(filtered_clicks.index).sum()
historical_click_data = historical_click_data.rename(columns=lambda column_name: 'click history:' + column_name)

# Merge historical buy and click data of every session id
merged1 = pd.merge(transformed_buys, historical_buy_data, left_index=True, right_index=True)
merged2 = pd.merge(merged1, historical_click_data, left_index=True, right_index=True)

# took Quantity as target and converted into binary
y = np.array(merged2['Quantity'].as_matrix())

# converting y into binary [if buying happens 1 else 0]
for i in range(y.shape[0]):
    if y[i]!=0:
        y[i]=1
    else:
        y[i]=0

# split the data into train and test
X_tr, X_te, y_tr, y_te = train_test_split(merged2, y, test_size=0.2)

# split the test data into half each for normal and cold start testing
X_te, X_te_cs, y_te, y_te_cs = train_test_split(X_te, y_te, test_size=0.5)

# taking session id and item id into dataframe
test_x = pd.DataFrame(X_te, columns = ['Item ID'])
test_x_cs = pd.DataFrame(X_te_cs, columns = ['Item ID'])

# Removing unwanted features from datasets
X_tr.drop(['Item ID', '_Session ID', 'click history:Item ID', 'buy history:Item ID', 'Quantity'], 1, inplace=True)
X_te.drop(['Item ID', '_Session ID', 'click history:Item ID', 'buy history:Item ID', 'Quantity'], 1, inplace=True)
X_te_cs.drop(['Item ID', '_Session ID', 'click history:Item ID', 'buy history:Item ID', 'Quantity'], 1, inplace=True)

# converting dataframes into array
ax_tr = np.array(X_tr)
ax_te = np.array(X_te)
ax_te_cs = np.array(X_te_cs)

# replacing NaN with zeros
ax_tr = np.nan_to_num(ax_tr)
ax_te = np.nan_to_num(ax_te)
ax_te_cs = np.nan_to_num(ax_te_cs)

# defining the model with optimized hyper parameters
model = TFFMClassifier(
        order=2, 
        rank=7, 
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001), 
        n_epochs=100, 
        batch_size=1024,
        init_std=0.001,
        reg=0.01,
        input_type='dense',
        log_dir = '/home/asif/01_tffm/logs/',
        verbose=1,
        seed=12345
    )

# preparing the data for cold start
cold_start = pd.DataFrame(ax_te_cs, columns=X_tr.columns)

# What happens if we only have access to categories and no historical click/purchase data?
# Let's delete historical click and purchasing data for the cold_start test set
for column in cold_start.columns:
    if ('buy' in column or 'click' in column) and ('Category' not in column):
        cold_start[column] = 0

# training the model
model.fit(ax_tr, y_tr, show_progress=True)

# predicting the buying events in the sessions
predictions = model.predict(ax_te)
print('accuracy: {}'.format(accuracy_score(y_te, predictions)))

cold_start_predictions = model.predict(ax_te_cs)
print('Cold-start accuracy: {}'.format(accuracy_score(y_te_cs, cold_start_predictions)))

# adding predicted values to test data
test_x["Predicted"] = predictions
test_x_cs["Predicted"] = cold_start_predictions

# finding all buy events for each session_id in test data and then retrieve the respective item id's
sess = list(set(test_x.index))
fout = open("solution.dat", "w")
print("writing the results into .dat file....")
for i in sess:
    if test_x.loc[i]["Predicted"].any()!= 0:
        fout.write(str(i)+";"+','.join(s for s in str(test_x.loc[i]["Item ID"].tolist()).strip('[]').split(','))+'\n')

fout.close()

# finding all buy events for each session id in cold start test data then retrieve the respective item id's
sess_cs = list(set(test_x_cs.index))
fout = open("solution_cs.dat", "w")
print("writing the cold start results into .dat file....")
for i in sess_cs:
    if test_x_cs.loc[i]["Predicted"].any()!= 0:
        fout.write(str(i)+";"+','.join(s for s in str(test_x_cs.loc[i]["Item ID"].tolist()).strip('[]').split(','))+'\n')

fout.close()
print("completed..!!")
