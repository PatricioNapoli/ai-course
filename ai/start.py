import graphviz
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

data = iris.data
target = iris.target
target_names = iris.target_names

classifier = tree.DecisionTreeClassifier()

trained_classifier = classifier.fit(data, target)

test1 = [[4, 2, 4, 2]]

prediction1 = trained_classifier.predict(test1)

print(test1[0], '-->', target_names[prediction1[0]])

dot_data = tree.export_graphviz(trained_classifier, class_names=target_names)
graph = graphviz.Source(dot_data)
graph.render('resources/iris', view=True)
