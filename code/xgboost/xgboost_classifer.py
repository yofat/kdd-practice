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

# xgboost classifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import timeit
from termcolor import colored
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, zero_one_loss,ConfusionMatrixDisplay

def xgboost_clf():
    print(colored("------XGBoost Classification-------", 'red'))

    """ xgb_model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                  colsample_bynode=1, colsample_bytree=1, gamma=1,
                                  learning_rate=0.2, max_delta_step=0, max_depth=3,
                                  min_child_weight=1, missing=np.nan, n_estimators=490, n_jobs=-1,
                                  nthread=None, objective='multi:softprob', random_state=0,
                                  reg_alpha=0, reg_lambda=1, seed=0,tree_method='gpu_hist',
                                  silent=None, subsample=1, verbosity=1,gpu_id=0) """
    
    xgb_model = xgb.XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, gpu_id=0, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=0.3, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=8, max_leaves=None,
              min_child_weight=None, missing=np.nan, monotone_constraints=None,
              n_estimators=100, n_jobs=None, num_parallel_tree=None,
              objective='multi:softprob', predictor=None)

    print("Training the XGBoost Classifier.......")

    # start timer
    starttime = timeit.default_timer()  # start timer
    
    label_encoder = LabelEncoder()
    train_Y_encoded = label_encoder.fit_transform(train_Y)
    test_Y_encoded = label_encoder.transform(test_Y)

    xgb_model.fit(train_x, train_Y_encoded)

    print("The time difference is :", timeit.default_timer() - starttime)

    print("Predicting test data.......")

    # print(xgb_model.feature_importances_)

    xgb_pred = xgb_model.predict(test_x)

    # plot
    # plot_importance(xgb_model, height=0.9)
    # pyplot.show()

    # Feature importance
    '''selector = RFE(xgb_model, 40, step=1)
    selector = selector.fit(train_x, train_Y)
    print(selector.support_)
    print(selector.ranking_)'''

    # results
    c_matrix = confusion_matrix(test_Y_encoded, xgb_pred)
    error = zero_one_loss(test_Y_encoded, xgb_pred)
    score = accuracy_score(test_Y_encoded, xgb_pred)

    # display results
    print('Confusion Matrix\n---------------------------\n', c_matrix)
    print('---------------------------')
    print("Error: {:.4f}%".format(error * 100))
    print("Accuracy Score: {:.4f}%".format(score * 100))
    print(classification_report(test_Y_encoded, xgb_pred))
    print('accuracy: ', c_matrix.diagonal() / c_matrix.sum(axis=1))

    # Plot non-normalized confusion matrix
    y_pred = xgb_model.predict(test_x)
    cm = confusion_matrix(test_Y_encoded, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=xgb_model.classes_)
    disp.plot(cmap=plt.cm.Greens, values_format='.0f', xticks_rotation='horizontal')

    plt.title("Confusion Matrix for XGBoost")

    plt.show()
    
if __name__ == "__main__":
          xgboost_clf()
          
          
#original
""" Training the XGBoost Classifier.......
The time difference is : 9.2222414
Predicting test data.......
Confusion Matrix
---------------------------
 [[9443   71  196    0    1]
 [1479 6085   72    0    0]
 [ 751  163 1472   37    0]
 [2294    0    4  274    2]
 [ 187    0    1    3    9]]
---------------------------
Error: 23.3366%
Accuracy Score: 76.6634%
              precision    recall  f1-score   support

           0       0.67      0.97      0.79      9711
           1       0.96      0.80      0.87      7636
           2       0.84      0.61      0.71      2423
           3       0.87      0.11      0.19      2574
           4       0.75      0.04      0.08       200

    accuracy                           0.77     22544
   macro avg       0.82      0.51      0.53     22544
weighted avg       0.81      0.77      0.73     22544

accuracy:  [0.97240243 0.79688318 0.60751135 0.10644911 0.045     ] """

#myself

""" Training the XGBoost Classifier.......
The time difference is : 34.3229839
Predicting test data.......
Confusion Matrix
---------------------------
 [[9434   67  207    1    2]
 [1250 6229  157    0    0]
 [ 721  165 1537    0    0]
 [2370    0    2  200    2]
 [ 192    0    0    0    8]]
---------------------------
Error: 22.7821%
Accuracy Score: 77.2179%
              precision    recall  f1-score   support

           0       0.68      0.97      0.80      9711
           1       0.96      0.82      0.88      7636
           2       0.81      0.63      0.71      2423
           3       1.00      0.08      0.14      2574
           4       0.67      0.04      0.08       200

    accuracy                           0.77     22544
   macro avg       0.82      0.51      0.52     22544
weighted avg       0.82      0.77      0.74     22544

accuracy:  [0.97147565 0.81574123 0.6343376  0.07770008 0.04      ] """