""" Stan Loosmore
    ITP-449
    H05 
    The practical goal of this assignment is to produce visualizations from the data contained within the Avocados dataset.
"""

def main():
    #importing needed libriries
    import numpy as np
    import matplotlib.pyplot as plt 
    import pandas as pd
    from datetime import datetime as dt
    #read in csv file
    file_path = 'ITP-449/Homeworks/csv_files/avocado.csv'
    vacado = pd.read_csv(file_path)


    #convert data to datetime by using built in pandas functionality
    vacado['Date'] = pd.to_datetime(vacado['Date'], format='%m/%d/%Y')
    #sort by date
    vacado.sort_values(by='Date')
    print(vacado.head())

    #Creating additional col tot rev
    vacado['TotalRevenue'] = vacado['AveragePrice'] * vacado['Total Volume']
    #Grouping data and summing by data to remove dupes
    ag_vag = vacado.groupby(by='Date', as_index=False).sum()
    #Dropping and then re-adding average price
    ag_vag_drop = ag_vag.drop(['AveragePrice'], axis=1)
    ag_vag_drop['AveragePrice'] = ag_vag_drop['TotalRevenue']/ag_vag_drop['Total Volume']
    print(ag_vag_drop.info())   

    #Smoothing data

    
    #Creating plots of each set
    fig, ax = plt.subplots(2, 3)
    ax[0,0].scatter(vacado['Date'],vacado['AveragePrice'])
    ax[1,0].scatter(vacado['Date'],vacado['Total Volume'])
    ax[0,1].plot(ag_vag_drop['Date'], ag_vag_drop['AveragePrice'], marker='o')
    ax[1,1].plot(ag_vag_drop['Date'],ag_vag_drop['Total Volume'], marker='o')
    #Using rolling/mean to smooth out data of last graphs
    ax[0,2].plot(ag_vag_drop['Date'], ag_vag_drop['AveragePrice'].rolling(20).mean(), marker='o')
    ax[1,2].plot(ag_vag_drop['Date'], ag_vag_drop['Total Volume'].rolling(20).mean(), marker='o')
    

    #setting y labels
    ax[0,0].set_ylabel('Average Price (USD)')
    ax[1,0].set_ylabel('Total Volume (millions)')

    #Adding in xlables, removing first row ticks, turning 2nd row ticks 90 deg, setting titles for 1st row
    top_title = ['Raw','Aggregated','Smoothed']
    for i in range(3):
        j = i-1
        ax[1,j].tick_params(axis='x', labelrotation = 90)
        ax[0,j].tick_params('x', labelbottom=False)
        ax[1,j].set_xlabel('Time')
        ax[0,j].set_title(top_title[j])

    #sizing graphs and global title setting
    fig.set_figheight(12)
    fig.set_figwidth(12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle('Avocado Prices and Volume Time Series')
    fig.savefig('ITP-449/Homeworks/outputs/vacado')

if __name__ == '__main__':
    main()