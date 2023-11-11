


from datetime import datetime
import pandas as pd

def main():
    # get the current system time
    now = datetime.now()

    # check out the format of what is printed
    print(now)

    # construct a new datetime object
    birthday = datetime(970, 1, 1)

    # notice how this prints
    print(birthday)

    # here's a String timestamp
    date_string = '..6-21?1970'
    # here's the format of the String
    fmt = '..%m-%d?%Y'

    # convert to a datetime object
    dt_new = datetime.strptime(date_string, fmt) 

    # notice how this prints
    print(dt_new)
    # notice how these print
    print('Original:', dt_new)
    print('Shortened:', dt_new.strftime('%Y.%m.%d'))
    print('Wackiness:', dt_new.strftime('%d!!!%Y?%d-,%m'))


# birthday challenge
def challenge_one():

    # get users birthday data
    # construct a datetime object from that data
    # get the current datetime
    # subtract birthday datetime from current datetime
    # display results
    year = (input('Year: '))
    month = (input('Mouth: '))
    day = (input('Day: '))
    aage = year + '-' + month + "-" + day
    fmt = '%Y-%m-%d'
    age = datetime.now() - datetime.strptime(aage, fmt)

    print(f'you are {age.days/365.25} days old')
    

def challenge_two():
    file_name = '/home/global_temp_anomaly_1880-2005.csv'
    noa = pd.read_csv(file_name, skiprows=5)
    #print(noa.head)
    #noa.drop([0,1,2,3,4], axis = 0)
    #noa.rename(columns=noa.iloc[0])
    noa['Year'] = pd.to_datetime(noa['Year'], format = '%Y')
    noa = noa.set_index('Year')
    print(noa.head())

    

if __name__ == '__main__':
    # main()
    challenge_two()