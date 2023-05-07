import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
import os

class LinearSVM(BaseEstimator):
    def __init__(self, C=1.0, n_epochs=1000, learning_rate=0.001):
        self.C = C
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.classifiers = []

    def _train_binary_classifier(self, X, y):
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0

        for _ in range(self.n_epochs):
            for idx, x_i in enumerate(X):
                if y[idx] * (np.dot(x_i, w) - b) >= 1:
                    w -= self.learning_rate * (2 * (1 / self.n_epochs) * w)
                else:
                    w -= self.learning_rate * (2 * (1 / self.n_epochs) * w - np.dot(x_i, y[idx]))
                    b -= self.learning_rate * y[idx]

        return w, b

    def fit(self, X, y):
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)

        for i in range(n_classes):
            y_binary = np.where(y == unique_classes[i], 1, -1)
            w, b = self._train_binary_classifier(X, y_binary)
            self.classifiers.append((w, b))

        return self

    def predict(self, X):
        scores = []

        for w, b in self.classifiers:
            scores.append(np.dot(X, w) - b)

        return np.argmax(np.vstack(scores).T, axis=1)

def run_experiment(seed, n_samples, n_classes, n_informative, n_features, n_redundant, C, n_epochs):
    random_state = np.random.default_rng(seed)

    X, y = make_classification(n_samples=n_samples, n_classes=n_classes, n_informative=n_informative,
                               n_features=n_features, n_redundant=n_redundant, random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LinearSVM(C=C, n_epochs=n_epochs)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)

    return acc_score

X, y = make_classification(n_samples=500, n_classes=3, n_informative=4, n_features=4, n_redundant=0, random_state=None)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=None)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = LinearSVM(C=1.0)
knn = KNeighborsClassifier(n_neighbors=3)
dt = DecisionTreeClassifier(random_state=None)
sklearn_svm = LinearSVC(C=1.0, multi_class='ovr', random_state=None)

def save_results(dir_name, file_name, results):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    result_filepath = os.path.join(dir_name, file_name)
    np.save(result_filepath, results)


clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc_score = accuracy_score(y_test, y_pred)
prec_score = precision_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
rec_score = recall_score(y_test, y_pred, average='macro')
clf_results = f'''--- Linear SVM (one vs all) --- 
accuracy = {acc_score}
precision = {prec_score}
f1 = {f1}
recall = {rec_score}
'''
print(clf_results)
save_results('results', 'linear_svm.npy', clf_results)

knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_score_knn = accuracy_score(y_test, y_pred_knn)
prec_score_knn = precision_score(y_test, y_pred, average='macro')
f1_knn = f1_score(y_test, y_pred, average='macro')
rec_score_knn = recall_score(y_test, y_pred, average='macro')
knn_results = f'''--- kNN --- 
accuracy = {acc_score_knn}
precision = {prec_score_knn}
f1 = {f1_knn}
recall = {rec_score_knn}
'''
print(knn_results)
save_results('results', 'knn.npy', knn_results)

dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
acc_score_dt = accuracy_score(y_test, y_pred_dt)
prec_score_dt = precision_score(y_test, y_pred, average='macro')
f1_dt = f1_score(y_test, y_pred, average='macro')
rec_score_dt = recall_score(y_test, y_pred, average='macro')
dt_results = f'''--- Decision tree --- 
accuracy = {acc_score_dt}
precision = {prec_score_dt}
f1 = {f1_dt}
recall = {rec_score_dt}
'''
print(dt_results)
save_results('results', 'decision_tree.npy', dt_results)

sklearn_svm.fit(X_train, y_train)
y_pred_sklearn_svm = sklearn_svm.predict(X_test)
acc_score_sklearn_svm = accuracy_score(y_test, y_pred_sklearn_svm)
prec_score_sklearn_svm = precision_score(y_test, y_pred, average='macro')
f1_sklearn_svm = f1_score(y_test, y_pred, average='macro')
rec_score_sklearn_svm = recall_score(y_test, y_pred, average='macro')
sklearn_svm_results = f'''--- SVM z biblioteki Sklearn --- 
accuracy = {acc_score_sklearn_svm}
precision = {prec_score_sklearn_svm}
f1 = {f1_sklearn_svm}
recall = {rec_score_sklearn_svm}
'''
print(sklearn_svm_results)
save_results('results', 'sklearn_svm.npy', sklearn_svm_results)

