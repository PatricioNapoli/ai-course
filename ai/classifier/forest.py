from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

iris = load_iris()

classifier = RandomForestClassifier(n_estimators=100, max_depth=5)

classifier.fit(iris.data, iris.target)

print(classifier.feature_importances_)


prediction = classifier.predict_proba([[4.9, 2.5, 1.2, 1]])

print(prediction)