""" Stan Loosmore
    ITP-449
    H08 - Diabetes Regression
    The practical goal of this assignment is to create a line of best fit of quantitative diabetes progression.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#using tucies fences to calc range
def calc_nonoutlier_range(series):
	Q3 = series.quantile(0.75)
	Q1 = series.quantile(0.25)
	IQR = Q3 - Q1
	lower_value = Q1 - 1.5 * IQR
	upper_value = Q3 + 1.5 * IQR
	return [lower_value, upper_value]


def main():
    #read in csv, and drop dupes
    dia = pd.read_csv('Homeworks/csv_files/diabetes.csv', skiprows=1)
    dia = dia.drop_duplicates()
    print(dia.info())
    print(dia.describe())

    #remove outliers using code from class
    lower, upper = calc_nonoutlier_range(dia['BMI'])
    filt_lower = dia['BMI'] >= lower
    filt_upper = dia['BMI'] <= upper
    dia = dia[filt_lower & filt_upper ]

    #isolate x and y data
    X = dia['BMI']
    #reshape data so it's 2D
    X = np.array(X).reshape(-1, 1)
    y = dia['Y']
    #create linear Regression
    model = LinearRegression()
    model.fit(X, y)
    #create line to put in graph
    pred = model.predict(X)

    #plot it
    fig, ax = plt.subplots(1,1, figsize=(15,10))
    ax.scatter(X, y, label='Diabetes Data')
    ax.plot(X, pred, label='Line of best fit', color='y')
    ax.set_xlabel('BMI')
    ax.set_ylabel('Progression')
    ax.set_title('Diabetes data: Progression vs BMI (Linear Regression)')
    ax.legend()
    fig.savefig('Homeworks/outputs/Diabetes Regression.png')

if __name__ == '__main__':
    main()