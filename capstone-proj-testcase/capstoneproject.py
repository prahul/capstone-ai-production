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
import logging

def model_predict(test,loglevel):
    
    logger = logging.getLogger()
    if loglevel=="debug":
        logger.setLevel(logging.DEBUG)
    elif loglevel=="info":
        logger.setLevel(logging.INFO)
    elif loglevel=="warn":
        logger.setLevel(logging.WARNING)
    elif loglevel=="error":
        logger.setLevel(logging.ERROR)
    elif loglevel=="critical":
        logger.setLevel(logging.CRITICAL)
    else:
        logger.setLevel(logging.INFO)
        
    logger.info("Model compare and prediction START!")
    if test is None:
        logger.warning("Testmode is empty")
    elif loglevel is None:
        logger.warning("Kog level not set default to info")
    
    # Load the dataset
    url = "https://github.com/prahul/capstone-ai-production/blob/master/iris.csv"
    logger.debug("Loading data from: %s", url )
    if url == "":
        logger.critical("No location specfied for data file")
    sepalname = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = read_csv(url, names=sepalname)
    logger.info("Dataset loaded successfully")
    

    # draw box and whisker plots
    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    pyplot.show()
    logger.info("Plot drawn successfully")


    # draw histograms
    dataset.hist()
    pyplot.show()
    logger.info("Plot drawn successfully")


    # draw scatter plot matrix
    scatter_matrix(dataset)
    pyplot.show()
    logger.info("Scatter plot matrix drawn successfully")


    # Split the validation dataset
    array = dataset.values
    X = array[:,0:4]
    y = array[:,4]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
    logger.debug("X_train: %s", X_train)
    logger.debug("X_validation: %s", X_validation)
    logger.debug("Y_train: %s", Y_train)
    logger.debug("Y_validation: %s", Y_validation)
    if X_train is None:
        logger.error("X_train value empty")
    elif X_validation is None:
        logger.error("X_validation value empty")
    elif Y_train is None:
        logger.error("Y_train value empty")
    elif Y_validation is None:
        logger.error("Y_validation value empty")
    else:
        logger.info("Validation dataset split successfully")
    

    # Check Algorithms
    mlmodels = []
    mlmodels.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    mlmodels.append(('LDA', LinearDiscriminantAnalysis()))
    mlmodels.append(('KNN', KNeighborsClassifier()))
    mlmodels.append(('CART', DecisionTreeClassifier()))
    mlmodels.append(('NB', GaussianNB()))
    mlmodels.append(('SVM', SVC(gamma='auto')))
    
    logger.debug("Models: %s", mlmodels)
    logger.info("Check algorithms successfully")
    # evaluate each model in turn
    modelresults = []
    sepalname = []
    for name, model in mlmodels:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_modelresults = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        modelresults.append(cv_modelresults)
        sepalname.append(name)
        print('%s: %f (%f)' % (name, cv_modelresults.mean(), cv_modelresults.std()))
    

    # Compare Algorithms
    logger.debug("Models results: %s", modelresults)
    if modelresults is None:
        logger.debug("No data found for model results")
        
    pyplot.boxplot(modelresults, labels=sepalname)
    pyplot.title('Algorithm Comparison')
    pyplot.show()
    logger.info("Algorithm compare success")

    # Make predictions on validation dataset
    logger.debug("X_train: %s", X_train)
    logger.debug("Y_train: %s", Y_train)
    model = SVC(gamma='auto')
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    logger.info("Make prediction success")


    # Evaluate predictions
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
    logger.info("Prediction eveluation complete")
    
    logger.info("Model compare and prediction COMPLETE!")