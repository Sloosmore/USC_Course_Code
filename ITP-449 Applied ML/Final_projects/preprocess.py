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

def main():
    #renamed csv because long name was annoying
    file_path = 'kepler.csv'
    
    #normal = input('do you want to see the non concat pipeline results too? y/n: ')
    
    bool = [False, True]
    
    
    for p in bool:
        
        print(f'pca is {p}')
        
        
        X, y = preprocess(file_path, pca=p)
        
        '''if normal == 'y':

            print('-----------------------------------------NORMAL--------------------------------------------')
            
            cross_classify(X, y)'''
        
        
        print('----------------------------------------PIPELINE------------------------------------------')
        
        
        print('------------------------------------------------------------------------------------------')

        '''
        QUESTIONS:
        1. Did PCA improve upon or not improve upon the results from when you did not use it?

        PCA didn't have a significant impact on the results based off observing the confustion matrix 
        and classification_report. While some areas of innacurate guessing imporved others got worse. 
        It produced similar results.  

        2. Why might this be?

        In my case I would guess PCA captured a very significant amount of the important information 
        such that entropy was minimized. It may be work looking into further expanding my pca investagation
        through less componets to deturmine if the models performance. 

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