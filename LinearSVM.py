from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
import os
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import tabulate

class LinearSVM(BaseEstimator):
    def __init__(self, C=1.0, n_epochs=200, learning_rate=0.001, multi_class='ovr'):
        self.C = C
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.multi_class = multi_class
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

        if self.multi_class == 'ovr':
            for i in range(n_classes):
                y_binary = np.where(y == unique_classes[i], 1, -1)
                w, b = self._train_binary_classifier(X, y_binary)
                self.classifiers.append((w, b))
        elif self.multi_class == 'ovo':
            class_combinations = list(combinations(unique_classes, 2))
            for class_comb in class_combinations:
                class_1, class_2 = class_comb

                mask = np.logical_or(y == class_1, y == class_2)
                X_pair = X[mask]
                y_pair = y[mask]
                y_binary = np.where(y_pair == class_1, 1, -1)

                w, b = self._train_binary_classifier(X_pair, y_binary)
                self.classifiers.append(((class_1, class_2), w, b))

        return self

    def predict(self, X):
        if self.multi_class == 'ovr':
            scores = []

            for w, b in self.classifiers:
                scores.append(np.dot(X, w) - b)

            return np.argmax(np.vstack(scores).T, axis=1)
        elif self.multi_class == 'ovo':
            scores = np.zeros((X.shape[0], len(self.classifiers)))

            for i, (class_comb, w, b) in enumerate(self.classifiers):
                class_1, class_2 = class_comb
                scores[:, i] = np.dot(X, w) - b
            
            predictions = np.argmax(scores, axis=1)

            return predictions

X, y = make_classification(n_samples=500, n_classes=3, n_informative=4, n_features=4, n_redundant=0, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

real_data = np.loadtxt('irys.csv', delimiter=',', skiprows=1)
real_X = real_data[:, :-1]
real_y = real_data[:, -1]

X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(real_X, real_y, train_size=0.8, test_size=0.2, random_state=42)

X_real_train = scaler.fit_transform(X_real_train)
X_real_test = scaler.transform(X_real_test)

clf = LinearSVM(C=1.0, multi_class='ovr')
clf1 = LinearSVM(C=1.0, multi_class='ovo')
knn = KNeighborsClassifier(n_neighbors=3)
dt = DecisionTreeClassifier(random_state=42)
sklearn_svm = LinearSVC(C=1.0, multi_class='ovr', random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def save_results(dir_name, file_name, results):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    result_filepath = os.path.join(dir_name, file_name)
    np.save(result_filepath, results)

def load_results(dir_name, file_name):
        result_filepath = os.path.join(dir_name, file_name)
        return np.load(result_filepath)

def metrics(X_train, y_train, classifier, data_type):
    acc_Array = []
    prec_Array = []
    f1_Array = []
    rec_Array = []
    if classifier == clf:
        description = "Linear SVM (one vs all)"
    elif classifier == clf1:
        description = "Linear SVM (one vs one)"
    elif classifier == knn:
        description = "kNN"
    elif classifier == dt:
        description = "Decision tree"
    elif classifier == sklearn_svm:
        description = "SVM (one vs all) z biblioteki Sklearn"
    for i, (train_index, test_index) in enumerate(kf.split(X_train, y_train)):
        X_train_fold, y_train_fold = X_train[train_index], y_train[train_index]
        X_test_fold, y_test_fold = X_train[test_index], y_train[test_index]
        classifier.fit(X_train_fold, y_train_fold)
        y_pred_general = classifier.predict(X_test_fold)
        acc_score = accuracy_score(y_test_fold, y_pred_general)
        prec_score = precision_score(y_test_fold, y_pred_general, average='macro')
        f1 = f1_score(y_test_fold, y_pred_general, average='macro')
        rec_score = recall_score(y_test_fold, y_pred_general, average='macro')
        acc_Array.append(acc_score)
        prec_Array.append(prec_score)
        f1_Array.append(f1)
        rec_Array.append(rec_score)
    acc_std = np.std(acc_Array)
    prec_std = np.std(prec_Array)
    f1_std = np.std(f1_Array)
    rec_std = np.std(rec_Array)
    #classifier_results = f'''--- {description} --- 
    #accuracy = {acc_Array} +- {acc_std}
    #precision = {prec_Array} +- {prec_std}
    #f1 = {f1_Array} +- {f1_std}
    #recall = {rec_Array} +- {rec_std}
    #'''
    #print(classifier_results)
    save_results(f'{data_type}_results', f'{data_type}_{classifier}_acc.npy', acc_Array)
    save_results(f'{data_type}_results', f'{data_type}_{classifier}_prec.npy', prec_Array)
    save_results(f'{data_type}_results', f'{data_type}_{classifier}_f1.npy', f1_Array)
    save_results(f'{data_type}_results', f'{data_type}_{classifier}_rec.npy', rec_Array)

    return {
        'description': description,
        'accuracy': acc_Array,
        'accuracy_std': acc_std,
        'precision': prec_Array,
        'precision_std': prec_std,
        'f1': f1_Array,
        'f1_std': f1_std,
        'recall': rec_Array,
        'recall_std': rec_std
    }

results_clf = metrics(X_train, y_train, clf, "synt")
results_clf1 = metrics(X_train, y_train, clf1, "synt")
results_knn = metrics(X_train, y_train, knn, "synt")
results_dt = metrics(X_train, y_train, dt, "synt")
results_sklearn_svm = metrics(X_train, y_train, sklearn_svm, "synt")

results_list = [results_clf, results_clf1, results_knn, results_dt, results_sklearn_svm]

#print(tabulate(results_list, headers="keys"))
# Wywołujemy funkcję metrics dla wszystkich klasyfikatorów i przechowujemy wyniki
results = []
classifiers = [clf, clf1, knn, dt, sklearn_svm]
for classifier in classifiers:
    result = metrics(X_train, y_train, classifier, "synt")
    results.append(result)

#lista słowników do utworzenia tabeli
table_data = []
for classifier in classifiers:
    result = metrics(X_train, y_train, classifier, "synt")
    table_data.append({
        'description': result['description'],
        'accuracy': np.mean(result['accuracy']),
        'accuracy_std': result['accuracy_std'],
        'precision': np.mean(result['precision']),
        'precision_std': result['precision_std'],
        'f1': np.mean(result['f1']),
        'f1_std': result['f1_std'],
        'recall': np.mean(result['recall']),
        'recall_std': result['recall_std']
    })

df = pd.DataFrame(table_data)
df.to_excel('results.xlsx', index=False)


# Tworzymy tabelę z danych
headers = ['description', 'accuracy', 'accuracy_std', 'precision', 'precision_std', 'f1', 'f1_std', 'recall', 'recall_std']
print(tabulate(table_data, headers, tablefmt='fancy_grid'))
#print("SYNT")
#metrics(X_train, y_train, clf, "synt")
#metrics(X_train, y_train, clf1, "synt")
#metrics(X_train, y_train, knn, "synt")
#metrics(X_train, y_train, dt, "synt")
#metrics(X_train, y_train, sklearn_svm, "synt")

#print("REAL")
#metrics(X_real_train, y_real_train, clf, "real")
#metrics(X_real_train, y_real_train, clf1, "real")
#metrics(X_real_train, y_real_train, knn, "real")
#metrics(X_real_train, y_real_train, dt, "real")
#metrics(X_real_train, y_real_train, sklearn_svm, "real")

# Wczytaj zapisane wyniki
def load_results(dir_name, file_name):
    result_filepath = os.path.join(dir_name, file_name)
    return np.load(result_filepath)

classifiers = [clf, clf1, knn, dt, sklearn_svm]
metrics = ['acc', 'prec', 'f1', 'rec']
result_type = ['synt', 'real']

for type in result_type:
    results = {}

    # Ładowanie wyników z plików
    for classifier in classifiers:
        for metric in metrics:
            results[(classifier, metric)] = load_results(f'{type}_results', f'{type}_{classifier}_{metric}.npy')

    # Test T-Studenta dla każdej pary klasyfikatorów
    for metric in metrics:
        print(f'--- {metric} ---')
        for i in range(len(classifiers)):
            for j in range(i+1, len(classifiers)):
                t_statistic, p_value = ttest_ind(results[(classifiers[i], metric)], results[(classifiers[j], metric)])
                print(f'{classifiers[i]} vs {classifiers[j]}: T-statistic = {t_statistic}, p-value = {p_value}')

    # Wykresy słupkowe dla każdej metryki
    for metric in metrics:
        mean_scores = [results[(classifier, metric)].mean() for classifier in classifiers]
        std_scores = [results[(classifier, metric)].std() for classifier in classifiers]

        plt.figure(figsize=(20, 6))
        sns.barplot(x=[str(classifier) for classifier in classifiers], y=mean_scores, yerr=std_scores)
        plt.title(f'Mean {metric} scores with standard deviation error bars. Data: {type}')
        plt.ylabel(f'{metric} score')
        
plt.show()