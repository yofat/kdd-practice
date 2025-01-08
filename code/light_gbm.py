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

unique_labels = np.unique(train_Y)
#print("Unique labels in train_Y:", unique_labels)

# lightgbm
import timeit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from lightgbm import log_evaluation , early_stopping
from termcolor import colored
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, zero_one_loss,ConfusionMatrixDisplay

seed = 42

params = {
    'num_leaves': 499,
    'min_child_weight': 0.01,
    'feature_fraction': 0.89,
    'bagging_fraction': 0.83,
    'bagging_freq': 1,
    'min_data_in_leaf': 120,
    'objective': 'binary',
    'max_depth': 47,
    'learning_rate': 0.1,
    'boosting_type': 'gbdt',
    'bagging_seed': seed,
    'metric': 'auc',
    'verbosity': -1,
    'reg_alpha': 1.17,
    'reg_lambda': 1.12,
    'random_state': seed,
}

""" le = LabelEncoder()
encoded_train_Y = le.fit_transform(train_Y) """

""" param_cv2 = {'max_depth': [5, 10, 20, 30, 40, 47, 50], 'num_leaves': [50, 100, 200, 300, 400, 499, 600]}
model = lgb.LGBMClassifier(boosting_type='gbdt', objective='multiclass', metric='auc', learning_rate=0.1, n_estimators=345,
                           min_child_samples=120, min_child_weight=0.009, feature_fraction=0.89, bagging_fraction=0.83,
                           bagging_freq=1, random_state=seed,verbosity = -1)

gsearch1 = GridSearchCV(estimator=model, param_grid=param_cv2, scoring='roc_auc', cv=3, verbose=2, error_score='raise')
gsearch1.fit(train_x, train_Y)

print('best-params : ', gsearch1.best_params_)
print('best-score : ', gsearch1.best_score_) """

def lightgbm_clf():
    print(colored("------Lightgbm Classification-------", 'red'))

    # model
    model = lgb.LGBMClassifier(objective='multiclass', boosting_type='goss', n_estimators=10000, class_weight='balanced', verbosity = -1)

    print("Training the Lightgbm classification.......")

    # start timer
    starttime = timeit.default_timer()  # start timer

    model.fit(train_x, train_Y)

    print("The time difference is :", timeit.default_timer() - starttime)

    print("Predicting test data.......")

    # predict
    y_pred = model.predict(test_x)

    # results
    c_matrix = confusion_matrix(test_Y, y_pred)
    error = zero_one_loss(test_Y, y_pred)
    score = accuracy_score(test_Y, y_pred)

    # display results
    print('Confusion Matrix\n---------------------------\n', c_matrix)
    print('---------------------------')
    print("Error: {:.4f}%".format(error * 100))
    print("Accuracy Score: {:.4f}%".format(score * 100))
    print(classification_report(test_Y, y_pred))
    print('accuracy: ', c_matrix.diagonal() / c_matrix.sum(axis=1))

    # Plot non-normalized confusion matrix
    y_pred = model.predict(test_x)
    cm = confusion_matrix(test_Y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Greens, values_format='.0f', xticks_rotation='horizontal')

    plt.title("Confusion Matrix for Lightgbm")

    plt.show()

if __name__ == "__main__":
          lightgbm_clf()
        
        
""" The time difference is : 86.3775812
Predicting test data.......
Confusion Matrix
---------------------------
 [[9431   71  205    2    2]
 [1444 6075  117    0    0]
 [ 690  163 1570    0    0]
 [2335    0    1  235    3]
 [ 182    0    2    0   16]]
---------------------------
Error: 23.1414%
Accuracy Score: 76.8586%
              precision    recall  f1-score   support

      benign       0.67      0.97      0.79      9711
         dos       0.96      0.80      0.87      7636
       probe       0.83      0.65      0.73      2423
         r2l       0.99      0.09      0.17      2574
         u2r       0.76      0.08      0.14       200

    accuracy                           0.77     22544
   macro avg       0.84      0.52      0.54     22544
weighted avg       0.82      0.77      0.74     22544

accuracy:  [0.97116672 0.7955736  0.64795708 0.09129759 0.08      ] """