# Load all the required libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

...
# Load the dataset
url = "https://github.com/prahul/capstone-ai-production/blob/master/iris.csv"
sepalname = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=sepalname)

...
# draw box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

...
# draw histograms
dataset.hist()
pyplot.show()

...
# draw scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

...
# Split the validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)


...
# Check Algorithms
mlmodels = []
mlmodels.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
mlmodels.append(('LDA', LinearDiscriminantAnalysis()))
mlmodels.append(('KNN', KNeighborsClassifier()))
mlmodels.append(('CART', DecisionTreeClassifier()))
mlmodels.append(('NB', GaussianNB()))
mlmodels.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
modelresults = []
sepalname = []
for name, model in mlmodels:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_modelresults = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	modelresults.append(cv_modelresults)
	sepalname.append(name)
	print('%s: %f (%f)' % (name, cv_modelresults.mean(), cv_modelresults.std()))
    
...
# Compare Algorithms
pyplot.boxplot(modelresults, labels=sepalname)
pyplot.title('Algorithm Comparison')
pyplot.show()

...
# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

...
# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))