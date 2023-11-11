"""
    Stan Loosmore
    ITP-449
    Week 04 in-class code
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main1():
    x = np.arange(0, 100)
    y = x + 10*(np.random.rand(x.shape[0]))
    print(x.shape[0])
    plt.plot(x,y, color = 'green')
    plt.savefig('ITP-449 first plot.png')
def challenge_one():
    years = [2016, 2017, 2018, 2019, 2020]
    enrollment_ITP_115 = [180, 250, 390, 540, 720]
    enrollment_ITP_449 = [70, 150, 130, 180, 220]
    plt.bar(years-.4, enrollment_ITP_115, color='y', label= 'ITP-115', alpha =.5)
    plt.bar(years+.4, years, enrollment_ITP_449, color='b', label = 'ITP-449', alpha = .5)

    plt.xlabel("year")
    plt.ylabel("enrolment")
    plt.legend()
    plt.title('Years to enroll')
    plt.savefig('ITP-449 2nd plot.png')

def challenge_pieplot():
    # browser market share as of 2022 Dec
    # https://gs.statcounter.com/browser-market-share#monthly-200901-202212
    browsers = ['Chrome', 'Safari', 'Edge', 'Samsung Internet', 'Firefox', 'Opera']
    percentage = [64.68, 18.29, 4.23, 3.05, 3.01, 2.25]
    pernp = np.array(percentage)
    sumper = (100 - np.sum(pernp))
    percentage.append(sumper)
    browsers.append('other')
    plt.pie(percentage, labels=browsers, autopct='%.2f')

    plt.legend()
    plt.title('browser marketshare')
    plt.savefig('Inclass_work/figs/ITP-449 3rd plot.png')

if __name__ == '__main__':
    challenge_pieplot()