from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, \
    f1_score, classification_report
from sklearn.tree import export_graphviz
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
import warnings


def unsw():
    train_file = pd.read_csv("unsw-nb15-train/UNSW_NB15_training-set.csv")
    test_file = pd.read_csv("unsw-nb15-test/UNSW_NB15_testing-set.csv")
    warnings.filterwarnings("ignore")

    columns = train_file.columns
    train_categorial_columns = train_file.columns[train_file.dtypes == 'object']
    test_categorial_columns = test_file.columns[test_file.dtypes == 'object']
    labellencoder = LabelEncoder()
    for column in train_categorial_columns:
        train_file[column] = labellencoder.fit_transform(train_file[column])
    for column in test_categorial_columns:
        test_file[column] = labellencoder.fit_transform(test_file[column])

    labels = 'normal', 'abnormal'
    plt.figure(figsize=(8, 6))
    plt.title("The number of normal and abnormal samples in the UNSW-NB15 training set", fontsize=13)
    train_file["label"].value_counts().plot(kind="pie", labels=None, autopct="%.2f")
    plt.legend(labels, loc="lower center", bbox_to_anchor=(1, 0))
    plt.show()

    X_train, y_train = train_file.drop('label', axis=1), train_file['label']
    X_test, y_test = test_file.drop('label', axis=1), test_file['label']

    rf_classifier = RandomForestClassifier(n_estimators=90, max_depth=10)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print("\nTP (Normal traffic): ", cm[0, 0])
    print("FP: ", cm[1, 0])
    print("FN: ", cm[0, 1])
    print("TN (Abnormal traffic): ", cm[1, 1], "\n")

    print(classification_report(y_test, y_pred, target_names=['normal', 'abnormal']))

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall(Sensivity): ", recall)
    print("FP - rate: ", (cm[1, 0]) / (cm[1, 0] + cm[1, 1]))
    print("F-measure: ", f1score)

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax);
    ax.yaxis.set_ticklabels(['Normal traffic', 'Abnormal traffic']);
    ax.xaxis.set_ticklabels(['Normal traffic', 'Abnormal traffic']);
    plt.show()

    tree = rf_classifier.estimators_[0]
    dot_data = export_graphviz(tree,
                               feature_names=X_train.columns,
                               filled=True,
                               max_depth=10,
                               impurity=False,
                               proportion=True)
    graph = graphviz.Source(dot_data)
    graph.view()


def kdd():
    kdd = pd.read_csv("kdd/kdd99cup.csv")
    warnings.filterwarnings("ignore")
    columns = kdd.columns

    kdd_categorial_columns = kdd.columns[kdd.dtypes == 'object'][:-1]
    labellencoder = LabelEncoder()
    for column in kdd_categorial_columns:
        kdd[column] = labellencoder.fit_transform(kdd[column])
    outcomes_kdd = {'normal.': 0, 'buffer_overflow.': 1, 'loadmodule.': 2, 'perl.': 3, 'neptune.': 4, 'smurf.': 5,
                    'guess_passwd.': 6, 'pod.': 7,
                    'teardrop.': 8, 'portsweep.': 9, 'ipsweep.': 10, 'land.': 11, 'ftp_write.': 12, 'back.': 13,
                    'imap.': 14, 'satan.': 15, 'phf.': 16,
                    'nmap.': 17, 'multihop.': 18, 'warezmaster.': 19, 'warezclient.': 20, 'spy.': 21, 'rootkit.': 22}
    outcome = kdd["outcome"].value_counts()
    kdd.outcome = [outcomes_kdd[i] for i in kdd.outcome]

    normal_elements = outcome.tolist()[2]
    abnormal_elements = (outcome.sum() - normal_elements)
    df = pd.DataFrame({'amount': [normal_elements, abnormal_elements]})
    df.plot(kind="pie", y='amount', labels=None, autopct="%.2f", figsize=(8, 6))
    labels = 'normal', 'abnormal'
    plt.legend(labels, loc="lower center", bbox_to_anchor=(1, 0))
    plt.title("The number of normal and abnormal samples in the KDD99 dataset", fontsize=13)
    plt.show()

    X, y = kdd.drop('outcome', axis=1), kdd['outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
    rf_classifier = RandomForestClassifier(n_estimators=50)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax);
    ax.yaxis.set_ticklabels(
        ['normal', 'buffer_overflow', 'loadmodule', 'perl', 'neptune', 'smurf', 'guess_passwd', 'pod',
         'teardrop', 'portsweep', 'ipsweep', 'land', 'ftp_write', 'back', 'imap', 'satan', 'phf',
         'nmap', 'multihop', 'warezmaster', 'warezclient']);
    ax.set_xticklabels(['normal', 'buffer_overflow', 'loadmodule', 'perl', 'neptune', 'smurf', 'guess_passwd', 'pod',
                        'teardrop', 'portsweep', 'ipsweep', 'land', 'ftp_write', 'back', 'imap', 'satan', 'phf',
                        'nmap', 'multihop', 'warezmaster', 'warezclient'], rotation=45)

    print("\nTP for normal traffic: ", cm[0, 0])
    print("FP for normal traffic: ", cm[1:, 0].sum())
    print("FN for normal traffic: ", cm[0, 1:].sum())
    print("TN for normal traffic: ", cm[1:, 1:].sum())

    print(classification_report(y_test, y_pred,
                                target_names=['normal', 'buffer_overflow', 'loadmodule', 'perl', 'neptune', 'smurf',
                                              'guess_passwd', 'pod',
                                              'teardrop', 'portsweep', 'ipsweep', 'land', 'ftp_write', 'back', 'imap',
                                              'satan', 'phf',
                                              'nmap', 'multihop', 'warezmaster', 'warezclient']))

    tp_abn = 0
    for i in range(1, 21, 1):
        tp_abn += cm[i, i]
    print("TP for all types of abnormal traffic: ", tp_abn)
    print("FP for all types of abnormal traffic: ", cm[0:, 1:].sum() - tp_abn)
    print("FN for all types of abnormal traffic: ", cm[1:, 0:].sum() - tp_abn)
    print("TN for all types of abnormal traffic: ", cm[0, 0], "\n")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1score = f1_score(y_test, y_pred, average='weighted')
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall(Sensivity): ", recall)
    print("FP - rate:", cm[1:, 0].sum() / (cm[1:, 0].sum() + cm[1:, 1:].sum()))
    print("F-measure: ", f1score, "\n")
    plt.show()

    tree = rf_classifier.estimators_[0]
    dot_data = export_graphviz(tree,
                               feature_names=X_train.columns,
                               filled=True,
                               max_depth=10,
                               impurity=False,
                               proportion=True)
    graph = graphviz.Source(dot_data)
    graph.view()


if __name__ == "__main__":
    print("-UNSW-NB15-dataset-------------------------------------\n")
    unsw()
    print("\n-KDD99-dataset---------------------------------------\n")
    kdd()


