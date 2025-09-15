import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score
import matplotlib.pyplot as plt


def trainning(X_train, X_test, y_train, y_test, md="decision_tree", max_iterator=None, show_matrix=False):
    
    selected_model = None
    if(max_iterator != None):
        selected_model = select_model(md, max_iterator)
    else:
        selected_model = select_model(md)
    model = selected_model[0]
    description = selected_model[1]

    if model == None:
        print("ERRO - Nenhum modelo válido foi selecionado")
        return None
    
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict, average="macro")
    f1 = f1_score(y_test, y_predict, average="macro")

    if(show_matrix):
        show_confusion_matrix(y_test, y_predict, model, description)

    return {
        "model" : description,
        "accuracy" : accuracy * 100,
        "recall" : recall * 100,
        "f1" : f1 * 100
    }

def show_metrics(scores):

    score_labels = []
    score_values = []
    for score in scores:
        for type in ["\n(ACURÁCIA)", "\n(RECALL)", "\n(F1)"]:
            score_labels.append(score["model"] + type)
        score_values.extend([score["accuracy"], score["recall"], score["f1"]])

    score_colors = [
        "#BA5B49", "#BA5B49", "#BA5B49",
        "#1C2E5C", "#1C2E5C", "#1C2E5C",
        "#295C1C", "#295C1C", "#295C1C"
    ]

    plt.bar(score_labels, score_values, color=score_colors)
    for i in range(len(score_values)):
        plt.text(-0.15 + i, score_values[i] + 0.25, round(score_values[i], 2))    
    plt.ylabel("Percentual (%)")
    plt.title("Métricas de Desempenho")
    plt.show()

def select_model(md, max_iterator=400):

    if md == "decision_tree":
        return [DecisionTreeClassifier(), "Decision Tree C."]
    elif md == "kneighbors":
        return [KNeighborsClassifier(), "KNeighbors C."]
    elif md == "logistic_regression":
        return [LogisticRegression(max_iter=max_iterator), "Logistic Regress."]
    else:
        return None

def show_confusion_matrix(y_test, y_predict, model, md):

    conf_matrix = confusion_matrix(y_test, y_predict)
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
    display.plot()
    plt.title("Matriz de Confusão - " + md)
    plt.show()


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

scores = []
for model in ["decision_tree", "kneighbors", "logistic_regression"]:
    results = trainning(X_train, X_test, y_train, y_test, model, show_matrix=True)
    scores.append(results)
print(scores)
show_metrics(scores)


