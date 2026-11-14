"""Lab 5: Decision Tree classification and model evaluation."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)
model = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification report:\n",
      classification_report(y_test, pred, target_names=data.target_names))
plot_tree(model, feature_names=data.feature_names,
          class_names=data.target_names, filled=True, rounded=True)
plt.tight_layout()
plt.savefig("lab5_decision_tree.png")
plt.show()
