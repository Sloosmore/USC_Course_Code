""" Stan Loosmore
    ITP-449
    H12 - Clustering Wine Quality
    In this assignment, you will analyze the WineQualityReds.csv dataset. (Make sure to take a look inside this file before importing to make sure you understand the structure!)
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


 
def main():
    #Adding and wrangling data
    path = 'Homeworks/csv_files/WineQualityReds.csv'
    wine_df = pd.read_csv(path)
    wine_df = wine_df.drop_duplicates()
    wine_df = wine_df.dropna()
    #droping quality 
    wine_quality = wine_df['quality']
    wine_df = wine_df.drop('quality', axis=1)

    #getting familiar with data
    print(wine_df.info())
    print(wine_df.columns)

    #normalize data from 0 to 1 and create new df
    norm = MinMaxScaler(feature_range=(0, 1))
    scale = norm.fit_transform(wine_df)
    norm_wine = pd.DataFrame(scale, columns=wine_df.columns)
    #defining interations for clusters 
    num_clusters = 11
    cluster_num = range(1, num_clusters)
    initia_list = []
    #running knn 10 times
    for x in cluster_num:
        cluster = KMeans(x, random_state=42)
        cluster.fit(norm_wine)
        initia_list.append(cluster.inertia_)
    

    #ploting results
    fig, ax = plt.subplots() 
    ax.plot(cluster_num, initia_list)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia')
    ax.set_title('Red Wines: Inertia vs Number of clusters')
    fig.savefig('Homeworks/outputs/Initia vs Num clusters')


    #the seems the most 'elbowish' to me from my graph so using that
    Optimial_cluster = KMeans(3, random_state=42)
    Optimial_cluster.fit(norm_wine)
    print(wine_df.info())

    #combining winedf wth results 
    results = pd.concat([wine_df, wine_quality], axis=1)
    results = results.rename(columns={'0':'quality'})
    print(results.info())

    #adding in lables
    results['ClusterID'] = Optimial_cluster.labels_
    #creating cross
    cross = pd.crosstab(results['ClusterID'],results['quality'])
    print(cross)

    #The wine quality does not seem to represent my clusters. 
    #Quality is more subjective and based on 'expert' ratings so it makes sense 

    


if __name__ == '__main__':
    main()