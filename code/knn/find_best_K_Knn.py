import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
import seaborn as sns

warnings.filterwarnings('ignore')

train_file = os.path.join('KDDTrain+.txt')
test_file = os.path.join('KDDTest+.txt')
          
# Original KDD dataset feature names obtained from 
# http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names
# http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

header_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
                'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type',
                'success_pred']

# Differentiating between nominal, binary, and numeric features

# root_shell is marked as a continuous feature in the kddcup.names 
# file, but it is supposed to be a binary feature according to the 
# dataset documentation

col_names = np.array(header_names)

nominal_idx = [1, 2, 3]
binary_idx = [6, 11, 13, 14, 20, 21]
numeric_idx = list(set(range(41)).difference(nominal_idx).difference(binary_idx))

nominal_cols = col_names[nominal_idx].tolist()
binary_cols = col_names[binary_idx].tolist()
numeric_cols = col_names[numeric_idx].tolist()

# training_attack_types.txt maps each of the 22 different attacks to 1 of 4 categories
# file obtained from http://kdd.ics.uci.edu/databases/kddcup99/training_attack_types

category = defaultdict(list)
category['benign'].append('normal')

with open('training_attack_types.txt', 'rb') as f:
    for line in f.readlines():
        attack, cat = line.decode().strip().split(' ')
        category[cat].append(attack)
        
attack_mapping = dict((v,k) for k in category for v in category[k])

###############
#Generating and analyzing train and test sets
###############


train_df = pd.read_csv(train_file, names=header_names)
train_df['attack_category'] = train_df['attack_type'].map(lambda x: attack_mapping[x])
train_df.drop(['success_pred'], axis=1, inplace=True)
test_df = pd.read_csv(test_file, names=header_names)
test_df['attack_category'] = test_df['attack_type'].map(lambda x: attack_mapping[x])
test_df.drop(['success_pred'], axis=1, inplace=True)

train_attack_types = train_df['attack_type'].value_counts()
train_attack_cats = train_df['attack_category'].value_counts()

test_attack_types = test_df['attack_type'].value_counts()
test_attack_cats = test_df['attack_category'].value_counts()

""" train_attack_types.plot(kind='barh', figsize=(20,10), fontsize=20)
train_attack_cats.plot(kind='barh', figsize=(20,10), fontsize=30)
test_attack_types.plot(kind='barh', figsize=(20,10), fontsize=15)
test_attack_cats.plot(kind='barh', figsize=(20,10), fontsize=30) """

# By definition, all of these features should have a min of 0.0 and a max of 1.0

train_df[binary_cols].describe().transpose()

train_df.groupby(['su_attempted']).size()

# Let's fix this discrepancy and assume that su_attempted=2 -> su_attempted=0

train_df['su_attempted'].replace(2, 0, inplace=True)
test_df['su_attempted'].replace(2, 0, inplace=True)

train_df.groupby(['su_attempted']).size()

# Now, that's not a very useful feature - let's drop it from the dataset

train_df.drop('num_outbound_cmds', axis = 1, inplace=True)
test_df.drop('num_outbound_cmds', axis = 1, inplace=True)
numeric_cols.remove('num_outbound_cmds')

#############
#Data preparation
#############

train_Y = train_df['attack_category']
train_x_raw = train_df.drop(['attack_category','attack_type'], axis=1)
test_Y = test_df['attack_category']
test_x_raw = test_df.drop(['attack_category','attack_type'], axis=1)

combined_df_raw = pd.concat([train_x_raw, test_x_raw])
combined_df = pd.get_dummies(combined_df_raw, columns=nominal_cols, drop_first=True)

train_x = combined_df[:len(train_x_raw)]
test_x = combined_df[len(train_x_raw):]

# Store dummy variable feature names
dummy_variables = list(set(train_x)-set(combined_df_raw))

#print(train_x.describe())

# Example statistics for the 'duration' feature before scaling
#print(train_x['duration'].describe())

# Experimenting with StandardScaler on the single 'duration' feature
from sklearn.preprocessing import StandardScaler

durations = train_x['duration'].values.reshape(-1, 1)
standard_scaler = StandardScaler().fit(durations)
scaled_durations = standard_scaler.transform(durations)
#print(pd.Series(scaled_durations.flatten()).describe())

""" # Experimenting with MinMaxScaler on the single 'duration' feature
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler().fit(durations)
min_max_scaled_durations = min_max_scaler.transform(durations)
#print(pd.Series(min_max_scaled_durations.flatten()).describe())

# Experimenting with RobustScaler on the single 'duration' feature
from sklearn.preprocessing import RobustScaler

min_max_scaler = RobustScaler().fit(durations)
robust_scaled_durations = min_max_scaler.transform(durations)
#print(pd.Series(robust_scaled_durations.flatten()).describe()) """

# Let's proceed with StandardScaler- Apply to all the numeric columns

standard_scaler = StandardScaler().fit(train_x[numeric_cols])

train_x[numeric_cols] = standard_scaler.transform(train_x[numeric_cols])
test_x[numeric_cols] = standard_scaler.transform(test_x[numeric_cols])
#print(train_x.describe())

train_Y_bin = train_Y.apply(lambda x: 0 if x is 'benign' else 1)
test_Y_bin = test_Y.apply(lambda x: 0 if x is 'benign' else 1)

# Find best K value in KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def find_k():
    # find value for parameter n_neighbors value between 1 to 20 where accuracy is higher
    for i in range(1, 21):
        # KNN classifier
        clf_knn = KNeighborsClassifier(n_neighbors=i)

        # train data
        clf_knn = clf_knn.fit(train_x, train_Y)

        # predict
        pred_y = clf_knn.predict(test_x)

        print('accuracy for k value ', i, ': ', accuracy_score(test_Y, pred_y))
        
if __name__ == "__main__":
          find_k()
          
""" accuracy for k value  1 :  0.7579400283889283
accuracy for k value  2 :  0.7505766501064585
accuracy for k value  3 :  0.7551898509581263
accuracy for k value  4 :  0.7511089425124201
accuracy for k value  5 :  0.7619322214336409
accuracy for k value  6 :  0.7601579134137686
accuracy for k value  7 :  0.7631742370475515
accuracy for k value  8 :  0.7519073811213627
accuracy for k value  9 :  0.7532824698367636
accuracy for k value  10 :  0.7511976579134138
accuracy for k value  11 :  0.752971965933286
accuracy for k value  12 :  0.752750177430802
accuracy for k value  13 :  0.7558552164655784
accuracy for k value  14 :  0.7553229240596168
accuracy for k value  15 :  0.755722143364088
accuracy for k value  16 :  0.7556777856635912
accuracy for k value  17 :  0.7563431511710433
accuracy for k value  18 :  0.7560770049680624
accuracy for k value  19 :  0.7567423704755145
accuracy for k value  20 :  0.7564762242725337 """