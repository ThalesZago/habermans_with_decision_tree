from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

dataFrame = pd.read_csv('haberman.data', names=[
                        'Age', 'year_P', 'N_nodes', 'Class'])

X = dataFrame.drop('Class', axis=1)
y = dataFrame['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)


y_pred = dtree.predict(X_test)

matrix = confusion_matrix(y_test, y_pred)

vn, fp, fn, vp = confusion_matrix(y_test, y_pred).ravel()

print(matrix)

totalRespostas = vn + fp + fn + vp
respostasCorretas = vn + vp

print(f"Total de dados analisados: {totalRespostas}")
print(f"vn, fp, fn, vp = {vn}, {fp}, {fn}, {vp} ")
print(f"Precisão: {round(((100*respostasCorretas)/totalRespostas), 2)}%")
