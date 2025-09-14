import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score
import matplotlib.pyplot as plt

scores = {
    "decision_tree" : {
        "accuracy" : None,
        "recall" : None,
        "f1" : None
    },
    "kneighbors" : {
        "accuracy" : None,
        "recall" : None,
        "f1" : None
    },
    "logistic_regression" : {
        "accuracy" : None,
        "recall" : None,
        "f1" : None
    }
}


df = pd.read_excel("Thyroid_Diff.xlsx")

print("=" * 40 + " SOBRE O DATASET " + "=" * 40)
print(df.head())
x = input()
print("\n" + "=" * 40 + " SOBRE O DATASET " + "=" * 40)
print(df.info())
x = input()
print("\n" + "=" * 40 + " SOBRE O DATASET " + "=" * 40)
print(df.describe())
x = input()


X = df.drop("Recurred", axis=1) # features (age, gender, smoking, ...)
y = df["Recurred"] # target (recurred? -> yes/no)

X_enc = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size = 0.3, random_state = 42)

# (1) treinamento usando Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
scores['decision_tree']['accuracy'] = accuracy_score(y_test, y_predict)
scores['decision_tree']['recall'] = recall_score(y_test, y_predict, average="macro")
scores['decision_tree']['f1'] = f1_score(y_test, y_predict, average="macro")

conf_matrix = confusion_matrix(y_test, y_predict)
display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = model.classes_)
display.plot()
plt.title("Matriz de Confusão - Decision Tree Classifier")
plt.show()

# (2) treinamento usando KNeighbors Classifier
model = KNeighborsClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
scores['kneighbors']['accuracy'] = accuracy_score(y_test, y_predict)
scores['kneighbors']['recall'] = recall_score(y_test, y_predict, average="macro")
scores['kneighbors']['f1'] = f1_score(y_test, y_predict, average="macro")

conf_matrix = confusion_matrix(y_test, y_predict)
display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = model.classes_)
display.plot()
plt.title("Matriz de Confusão - KNeighbors Classifier")
plt.show()

# (3) treinamento usando Logistic Regression
model = LogisticRegression(max_iter=400)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
scores['logistic_regression']['accuracy'] = accuracy_score(y_test, y_predict)
scores['logistic_regression']['recall'] = recall_score(y_test, y_predict, average="macro")
scores['logistic_regression']['f1'] = f1_score(y_test, y_predict, average="macro")

conf_matrix = confusion_matrix(y_test, y_predict)
display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = model.classes_)
display.plot()
plt.title("Matriz de Confusão - Logistic Regression")
plt.show()


score_labels = [
    "Decision Tree C. \n(ACURÁCIA)", "KNeighbors C. \n(ACURÁCIA)", "Logistic Regression \n(ACURÁCIA)",
    "Decision Tree C. \n(RECALL)", "KNeighbors C. \n(RECALL)", "Logistic Regression \n(RECALL)",
    "Decision Tree C. \n(F1)", "KNeighbors C. \n(F1)", "Logistic Regression \n(F1)"
]

score_numbers = [
    scores["decision_tree"]["accuracy"] * 100, scores["kneighbors"]["accuracy"] * 100, scores["logistic_regression"]["accuracy"] * 100,
    scores["decision_tree"]["recall"] * 100, scores["kneighbors"]["recall"] * 100, scores["logistic_regression"]["recall"] * 100,
    scores["decision_tree"]["f1"] * 100, scores["kneighbors"]["f1"] * 100, scores["logistic_regression"]["f1"] * 100
]

score_colors = [
    "#BA5B49", "#BA5B49", "#BA5B49",
    "#1C2E5C", "#1C2E5C", "#1C2E5C",
    "#295C1C", "#295C1C", "#295C1C"
]

plt.bar(score_labels, score_numbers, color=score_colors)

#plt.text(-0.15, score_numbers[0] + 0.25, round(score_numbers[0], 2))

for i in range(len(score_numbers)):
    plt.text(-0.15 + i, score_numbers[i] + 0.25, round(score_numbers[i], 2))
    

plt.ylabel("Percentual (%)")
plt.title("Métricas de Desempenho")
plt.show()