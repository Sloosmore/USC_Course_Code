""" Stan Loosmore
    ITP-449
    Final
    NASA Kepler Exoplanet Candidate Classification
"""

'''
IMPORTANT: the all in one pipeline code is inconsistant on if it times out or not 75% of the time it runs but it occationaly does not. 
I have also split all of the model tests to have there own indivdual pipelines as technically allowed in the instructions.  
If the code does not run with the pipeline unstring the extra code in the main function and it will run an search manually.
I talked with Prof Aldenderfer about this. 
'''

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import math
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance


def preprocess(file_path, pca = False):
    df = pd.read_csv(file_path, skiprows=41)

    #Drop all columns with 0 rows 

    #drop_col = ["koi_eccen", "koi_eccen_err1", "koi_eccen_err2", "koi_sma_err1", "koi_sma_err2", 'koi_incl_err1', 'koi_incl_err2', 'koi_teq_err1', 'koi_teq_err2']
    #for x in range(len(drop_col)):
        #df = df.drop(columns=drop_col[x])
    '''koi_eccen is all zeros'''
    df = df[['koi_period', 'koi_duration', 'koi_prad', 'koi_sma', 'koi_incl', 'koi_teq', 'koi_dor', 'koi_steff', 'koi_srad', 'koi_smass', 'koi_pdisposition']]

    #drop dupes and drop nas 

    df = df.dropna()
    X = df.drop_duplicates()
    #print(df.info())
    


    """okay so target will be COLUMN koi_pdisposition: Exoplanet Archive Disposition because it includes the most data.
    koi_pdisposition uses just kepler so I will drop it because it uses all the other data"""

    X = df.drop(columns=['koi_pdisposition'])
    #stan = StandardScaler()
    #scale = stan.fit_transform(X)
    #X = pd.DataFrame(scale, columns=X.columns)
    '''  this is handled in the pipleline now
    '''
    #print(X.shape)   
    if pca:
        pca = PCA(n_components=6)
        X = pca.fit_transform(X)
        X = pd.DataFrame(X) 
    
    

    y = df['koi_pdisposition'] 
    unique_classes = np.unique(y)
    print(unique_classes)
    scale_mapper = {'FALSE POSITIVE': 0, 'CANDIDATE':1}
    y = y.replace(scale_mapper)
    corr = pd.concat((X,y), axis=1).corr()
    print(corr)
    corr.to_csv('corr.csv')
    
    return X, y

'''

these are non cross validated operations!

'''


def Logreg(X_train, X_test, y_train, y_test):
    lgs_model = LogisticRegression(max_iter=200)
    lgs_model.fit(X_train, y_train)
    y_pred_lgs = lgs_model.predict(X_test)
    print(classification_report(y_test, y_pred_lgs))

def KNN(X_train, X_test, y_train, y_test):
    num_ex, atr = X_train.shape
    ex_sqr = math.isqrt(num_ex)
    rng = ex_sqr*1.5-ex_sqr
    acc = []
    rng = range(ex_sqr, math.ceil(ex_sqr*1.5))
    for i in rng:
        kn_model = KNeighborsClassifier(n_neighbors=i)
        kn_model.fit(X_train, y_train)
        #y_pred_kn = lgs_model.predict(X_test)
        acc.append(kn_model.score(X_test, y_test))
    fig, ax = plt.subplots(1,1)
    ax.plot(rng, acc)
    print
    fig.savefig('KNN')

def Dtree(X_train, X_test, y_train, y_test):    
    criter = ['gini', 'entropy']
    max_dep = range(3, 16)
    min_samp_leaf = range(1, 11)
    gin_mx = np.zeros((13,10))
    crit_mx = gin_mx.copy()
    #gin 
    #criter
    for c in criter:
        for d in max_dep:
            for l in min_samp_leaf:
                model_tree = DecisionTreeClassifier(criterion = c, max_depth =d, min_samples_leaf=l, random_state=42)
                model_tree.fit(X_train, y_train)
                if c == 'gini':
                    gin_mx[d-3][l-1] = model_tree.score(X_test, y_test)
                else:
                    crit_mx[d-3][l-1] = model_tree.score(X_test, y_test)
        if c == 'gini':
            gin_df = pd.DataFrame(gin_mx, columns = min_samp_leaf)
        else:
            crit_df = pd.DataFrame(crit_mx, columns = min_samp_leaf)
    print(gin_df)    
    print(crit_df)
    #upon visual inspection the most accurate decition tree uses, gini, 7 for max depth, and 2 for the min leaf samples

def Sc(X_train, X_test, y_train, y_test):
    svc_mx = np.zeros((4, 3))
    svc_c_list = [.1, 1, 10, 100]
    svc_gamma_list = [.1, 1, 10]
    for c, cval in enumerate(svc_c_list):
        for g, gval in enumerate(svc_gamma_list):
            svc_model = SVC(kernel='rbf', C=cval, gamma=gval, random_state=42)
            svc_model.fit(X_train, y_train)
            svc_mx[c][g] = svc_model.score(X_test, y_test)
    svc_df = pd.DataFrame(svc_mx, columns=svc_gamma_list)
    print(svc_df)

'''

lets now cross validate in these functions with pipelines

'''

#LogisticRegression
def cross_Logreg(X, y):
    pipe = Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(max_iter=200))])
    cross = cross_val_score(pipe, X, y)
    acc = cross.mean()
    print(acc)
    return acc

#KNN
def cross_KNN(X, y):
    num_ex, atr = X.shape
    ex_sqr = math.isqrt(math.ceil(num_ex*.8))
    rng = ex_sqr*1.5-ex_sqr
    acc = []
    rng = range(ex_sqr, math.ceil(ex_sqr*1.5))
    for i in rng:
        pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=i))])
        cross = cross_val_score(pipe, X, y)
        #y_pred_kn = lgs_model.predict(X_test)
        acc.append(cross.mean())
    fig, ax = plt.subplots(1,1)
    ax.plot(rng, acc)
    fig.savefig('KNN')
    np_acc = np.asanyarray(acc)
    index = np.argmax(np_acc)
    print(f'best k is {rng[index]} with accuracy {max(acc)}')
    return (rng[index], max(acc))

#DecisionTree
def cross_Dtree(X, y):    
    criter = ['gini', 'entropy']
    max_dep = range(3, 16)
    min_samp_leaf = range(1, 11)
    gin_mx = np.zeros((13,10))
    crit_mx = gin_mx.copy()
    #gin 
    #criter
    for c in criter:
        for d in max_dep:
            for l in min_samp_leaf:
                pipe = Pipeline([('scaler', StandardScaler()), ('dtree', DecisionTreeClassifier(criterion = c, max_depth =d, min_samples_leaf=l, random_state=42))])
                cross = cross_val_score(pipe, X, y)
                if c == 'gini':
                    gin_mx[d-3][l-1] = cross.mean()
                else:
                    crit_mx[d-3][l-1] = cross.mean()
        if c == 'gini':
            gin_df = pd.DataFrame(gin_mx, columns = min_samp_leaf)
        else:
            crit_df = pd.DataFrame(crit_mx, columns = min_samp_leaf)
    print(gin_df)    
    print(crit_df)

def cross_Sc(X, y):
    svc_mx = np.zeros((4, 3))
    svc_c_list = [.1, 1, 10, 100]
    svc_gamma_list = [.1, 1, 10]
    for c, cval in enumerate(svc_c_list):
        for g, gval in enumerate(svc_gamma_list):
            pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf', C=cval, gamma=gval, random_state=42))])
            cross = cross_val_score(pipe, X, y)
            svc_mx[c][g] = cross.mean()
    svc_df = pd.DataFrame(svc_mx, columns=svc_gamma_list)
    print(svc_df)

def cross_classify(X, y):
    print('Logistic Regression')
    cross_Logreg(X, y)
    print('KNN')
    cross_KNN(X, y)
    print('Decision Tree')
    cross_Dtree(X, y)
    print('SVC')
    cross_Sc(X, y)

'''
now lets do everything in one pipeline
'''

def modelpipe(X, y, pca):
    
    #info for KNN
    num_ex, atr = X.shape
    ex_sqr = math.isqrt(math.ceil(num_ex*.8))

    
    pipe = Pipeline([
        ('estimator', DummyClassifier())   
    ])

    estimator_list = [
        {
            'estimator': [KNeighborsClassifier()],
            'estimator__n_neighbors': range(ex_sqr, math.ceil(ex_sqr*1.5))
        },
        {
            'estimator': [LogisticRegression()],
            'estimator__max_iter': [500],
        },
        {
            'estimator': [SVC()],
            'estimator__kernel': ['rbf', 'linear', 'poly'],
            'estimator__C': [0.1, 1, 10, 100],
            'estimator__gamma': [0.1, 1, 10]
        },
        {
            'estimator': [DecisionTreeClassifier()],
            'estimator__max_depth': range(3, 16),
            'estimator__min_samples_leaf': range(1, 11),
            'estimator__criterion': ['entropy', 'gini']
        }

    ]

    # create the cross validator
    gscv = RandomizedSearchCV(
        pipe,  
        param_distributions=estimator_list,
        scoring='accuracy',
        #n_jobs=-1,
        n_iter=7,
    )
    
    print('now fitting')

    gscv.fit(X, y)  # fit 
    
    print('done fitting')
    
    best_model = gscv.best_estimator_
    print(best_model)
    
    #creating conf matrix
    
    cm = confusion_matrix(y, best_model.predict(X))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['FALSE POSITIVE', 'CANDIDATE',])
    disp.plot()
    plt.tight_layout()
    plt.savefig(f'confusion_matrix with pca {pca}.png')
    
    #classification report
    
    print(classification_report(y, best_model.predict(X)))
    
    #looking at most valible atributes assuming pca is false
    
    if pca == False:
        res = permutation_importance(best_model, X, y, n_repeats=10, random_state=42, )
        series = pd.Series(res.importances_mean, index=X.columns)
        print(series.sort_values(ascending=False))
        
    
    


def main():
    #renamed csv because long name was annoying
    file_path = 'kepler.csv'
    
    normal = input('do you want to see the non concat pipeline results too (you should probably say n unless the server is having timeout issues)? y/n: ')
    
    bool = [False, True]
    
    
    for p in bool:
        
        print(f'pca is {p}')
        
        
        X, y = preprocess(file_path, pca=p)
        
        if normal == 'y':

            print('-----------------------------------------NORMAL--------------------------------------------')
            
            cross_classify(X, y)
        
        
        print('----------------------------------------PIPELINE------------------------------------------')
        
        modelpipe(X, y, p)
        
        print('------------------------------------------------------------------------------------------')

        '''
        "My final model is the non PCA model I just wanted to be more thourough and produce classification reports and a confusion matrix for both"

        QUESTIONS:
        1. Did PCA improve upon or not improve upon the results from when you did not use it?

        PCA performed slighly worse classifying both positives and negatives.  

        2. Why might this be?

        There were likley enough training exsamples for the curse of dimentionality to be minimal and the information in was present albiet minimal.

        3. Was your model able to classify objects equally across labels? How can you tell?

        It was not. It was significantly better at predicting the absolute exsitance or denial of an exoplanet.
        I was able to assess this from the classification matrix and report. 

        4. Based on your results, which attribute most significantly influences whether or not an object is an exoplanet?

        Based off the permutation importance, koi_prad is the most important attribute.        

        5.Describe that attribute in your own words and why you believe it might be most influential. (This is an opinion question so the only way to get it wrong is to not actually reflect.)

        Based on the definition given about exoplanets (earth-sized planents) if the raduis of the planet is not similar
        to earths it is less likely to be an exoplanet. Also potentially the larger planets in theroy would make the 
        luminocity of the starts deviate more when passing in front making them more detectable. 

        '''
    
    
    
if __name__ == '__main__':
    main()