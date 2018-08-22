import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report,  roc_curve


def plot_confusion_matrix(y_true, y_pred):

    y_true_normal = y_true.value_counts()[0]
    y_true_fraud = y_true.value_counts()[1]

    cfn_matrix = confusion_matrix(y_true, y_pred)
    cfn_norm_matrix = np.array([[1.0 / y_true_normal, 1.0 / y_true_normal], [1.0 / y_true_fraud, 1.0 / y_true_fraud]])
    norm_cfn_matrix = cfn_matrix * cfn_norm_matrix

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1)
    sns.heatmap(cfn_matrix, cmap='coolwarm_r', linewidths=0.5, annot=True, ax=ax)
    plt.title('Confusion Matrix')
    plt.ylabel('True values')
    plt.xlabel('Predicted values')

    ax = fig.add_subplot(1, 2, 2)
    sns.heatmap(norm_cfn_matrix, cmap='coolwarm_r', linewidths=0.5, annot=True, ax=ax)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True values')
    plt.xlabel('Predicted values')

    plt.show()

    print('Classification Report')
    print(classification_report(y_true, y_pred))


def plot_roc_curve(fpr, tpr, model):
    plt.figure(figsize=(16, 8))
    plt.title("ROC curve of " + model)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

    plt.show()

data = pd.read_csv("creditcard.csv")
print(data.head())
print(data.info())

colors = ['b', 'r']

print('Distribution of the Classes in the unsampled dataset')
print(data.Class.value_counts())
print('Normal transactions: ', round(data.Class.value_counts()[0] / len(data) * 100, 2), '% of the dataset')
print('Fraudulent transactions: ', round(data.Class.value_counts()[1] / len(data) * 100, 2), '% of the dataset')


sns.countplot('Class', data=data, palette=colors)
plt.title('Normal vs Fraud counts')



f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 4))
plt.suptitle('Distribution of transaction Time')
sns.distplot(data.Time[data.Class == 1], ax=ax1, color='r', bins=50)
ax1.set_title('Fraud')
sns.distplot(data.Time[data.Class == 0], ax=ax2, color='b', bins=50)
ax2.set_title('Normal')


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 4))
plt.suptitle('Distribution of transaction Amount')
sns.distplot(data.Amount[data.Class == 1], ax=ax1, color='r', bins=50)
ax1.set_title('Fraud')
sns.distplot(data.Amount[data.Class == 0], ax=ax2, color='b', bins=50)
ax2.set_title('Normal')


import matplotlib.gridspec as gridspec

plt.figure(figsize=(10, 30))
gs = gridspec.GridSpec(28, 1)
for n in range(1, 29):
    ax = plt.subplot(gs[n-1])
    sns.distplot(data['V'+str(n)][data.Class == 1], bins=50)
    sns.distplot(data['V'+str(n)][data.Class == 0], bins=50)
    ax.set_xlabel("")
    ax.set_title('V'+str(n))

plt.tight_layout()


f, ax = plt.subplots(figsize=(24, 20))

corr = data.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax)
ax.set_title("Correlation Matrix", fontsize=14)
ax.set_yticklabels(data.columns, rotation=45)



from sklearn.preprocessing import RobustScaler

rob_scaler = RobustScaler()

data['scaled_amount'] = rob_scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data['scaled_time'] = rob_scaler.fit_transform(data['Time'].values.reshape(-1, 1))

data.drop(['Time', 'Amount'], axis=1, inplace=True)




X = data.drop(['Class'], axis=1)
y = data.Class

X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = LinearSVC(dual=False, class_weight={1:1000, 0:1})
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
plot_confusion_matrix(y_test, y_pred)


# Under-sampling

df = data.sample(frac=1)

fraud = df.loc[df['Class'] == 1]
normal = df.loc[df['Class'] == 0][:1000]

eq_dist_df = pd.concat([fraud, normal])

df_us = eq_dist_df.sample(frac=1, random_state=42)

print('Distribution of the Classes after undersampling')
print(df_us.Class.value_counts())
print(df_us['Class'].value_counts()/len(df_us))


fig = plt.figure()
sns.countplot('Class', data=df_us, palette=colors)
plt.title('Equally Distributed Classes', fontsize=14)


f, ax = plt.subplots(figsize=(24, 20))

sub_sample_corr = df_us.corr()
sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax)
ax.set_title('Correlation Matrix after undersampling', fontsize=14)
ax.set_yticklabels(df_us.columns, rotation=45)

plt.show()

X = df_us.drop('Class', axis=1)
y = df_us['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "KNearestNeighbors": KNeighborsClassifier(),
    "Support Vector Classifier": SVC()
}

for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    print("Classifier ", classifier.__class__.__name__, "has accuracy score", round(training_score.mean(), 2) * 100, "%")



log_reg_params = {"penalty": ['l1', 'l2'],
                  'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(X_train, y_train)
log_reg = grid_log_reg.best_estimator_
print('Best parameters for logit: ', grid_log_reg.best_params_)


knn_params = {"n_neighbors": list(range(2, 5, 1)),
              "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute']}

grid_knn = GridSearchCV(KNeighborsClassifier(), knn_params)
grid_knn.fit(X_train, y_train)
knn = grid_knn.best_estimator_
print('Best parameters for kNN: ', grid_knn.best_params_)

svc_params = {'C': np.logspace(-3, 2, 6),
              'kernel': ['linear'],
              'gamma': np.logspace(-3, 2, 6)}

grid_svc = GridSearchCV(SVC(), svc_params)
grid_svc.fit(X_train, y_train)
svc = grid_svc.best_estimator_
print("Best parameters for SVC: ", grid_svc.best_params_)

log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)
print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')

knears_score = cross_val_score(knn, X_train, y_train, cv=5)
print('Knears Neighbors Cross Validation Score', round(knears_score.mean() * 100, 2).astype(str) + '%')

svc_score = cross_val_score(svc, X_train, y_train, cv=5)
print('Support Vector Classifier Cross Validation Score', round(svc_score.mean() * 100, 2).astype(str) + '%')

log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5, method="decision_function")
print("ROC_AUC score of logit: ", roc_auc_score(y_train, log_reg_pred))
log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_pred)
plot_roc_curve(log_fpr, log_tpr, "logit model")
y_pred = log_reg.predict(X_test)
plot_confusion_matrix(y_test, y_pred)

knn_pred = cross_val_predict(knn, X_train, y_train, cv=5)
print("ROC_AUC score of kNN: ", roc_auc_score(y_train, knn_pred))
knn_fpr, knn_tpr, knn_thresold = roc_curve(y_train, knn_pred)
plot_roc_curve(knn_fpr, knn_tpr, "kNN")
y_pred = knn.predict(X_test)
plot_confusion_matrix(y_test, y_pred)

svc_pred = cross_val_predict(svc, X_train, y_train, cv=5, method="decision_function")
print("ROC_AUC score of svc: ", roc_auc_score(y_train, svc_pred))
svc_fpr, svc_tpr, svc_thresold = roc_curve(y_train, svc_pred)
plot_roc_curve(svc_fpr, svc_tpr, "SVC")
y_pred = svc.predict(X_test)
plot_confusion_matrix(y_test, y_pred)



# over_sampling

data = pd.read_csv("creditcard.csv")

from sklearn.utils import shuffle

fraud = shuffle(data[data.Class == 1])
normal = shuffle(data[data.Class == 0])

X_train = fraud.sample(frac=0.8)
X_train = pd.concat([X_train, normal.sample(frac=0.8)], axis=0)

remainder = data.loc[~data.index.isin(X_train.index)]
X_test = remainder.sample(frac=1)

from imblearn.over_sampling import ADASYN

ada = ADASYN()
X_train_resampled, X_train_labels_resampled = ada.fit_sample(np.array(X_train.ix[:, X_train.columns != 'Class']),
                                                       np.array(X_train.Class))


from collections import Counter
print(Counter(X_train_labels_resampled))

X_train_resampled = pd.DataFrame(X_train_resampled)
X_train_labels_resampled = pd.DataFrame(X_train_labels_resampled)
X_train_resampled = pd.concat([X_train_resampled, X_train_labels_resampled], axis=1)
X_train_resampled.columns = X_train.columns


X_train = shuffle(X_train_resampled)
X_test = shuffle(X_test)

df_os = pd.concat([X_train, X_test])
print(df_os.head())
print(df_os.info())

for feature in X_train.columns.values[:-1]:
    mean, std = df_os[feature].mean(), df_os[feature].std()
    X_train.loc[:, feature] = (X_train[feature] - mean) / std
    X_test.loc[:, feature] = (X_test[feature] - mean) / std



y_train = X_train.Class
y_test = X_test.Class
X_train = X_train.drop(['Class'], axis=1)
X_test = X_test.drop(['Class'], axis=1)


datadict = {
    'X_train' : X_train,
    "X_test" : X_test,
    'y_train' : y_train,
    'y_test' : y_test
}


for k, v in datadict.items():
    if 'y' in k:
        print(k, Counter(v))



lsvm = LinearSVC(C=1.0, dual=False)
lsvm.fit(X_train, y_train)
y_pred = lsvm.predict(X_test)
plot_confusion_matrix(y_test, y_pred)


fpr, tpr, _ = roc_curve(y_test, y_pred)
plot_roc_curve(fpr, tpr, "linear SVM") 


