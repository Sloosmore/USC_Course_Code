"""
    Stan Loosmore
    ITP-449
    Week 05 in-class code
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    file_path = 'Inclass_work/csv_folders/titanic.csv'
    df_titan = pd.read_csv(file_path)
    df_titan = df_titan.drop_duplicates()
    print(df_titan.head())
    
    #df_titan = df_titan.loc[:, ['pclass', 'sex','age','survived']]
    df_titan = df_titan[['pclass', 'sex','age','survived']]
    df_titan = df_titan.fillna(df_titan['age'].mean())
    print(df_titan['pclass'].unique())

    fig, ax = plt.subplots()
    ax.boxplot(df_titan['age'], vert = False)
    ax.set_title('Age of dataset')
    ax.set_xlabel('Age')
    #ax.set_ylabel('# of people')
    #plt.savefig('Titanic_age_boxplot.png')



    plt.savefig('Inclass_work/figs/Age_boxplot.png')
    
    #print(df_titan)
    

if __name__ == '__main__':
    main()