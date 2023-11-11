""" Samuel Stanley Loosmore
    ITP-449
    H01
    Write a program which, given a user's name and birth date, will calculate and show how old they are.
"""

def main():
    # write your code here

    cont = True
    #Here are the inputs for the age calculator
    name = input('Hello! Enter your name: ')
    birth_year = input('Enter the year you were born: ')
    birth_month = input('Enter the month of the year you were born: ')
    birth_day = input('Enter the day of the month you were born: ')


    #Make sure that the input year makes is valid
    if not birth_year.isdigit():
        raise Exception('Your year input needs to be a whole number') 
    birth_year = int(birth_year)

    if birth_year >= 2024 :
        print()
        print(f'You have entered a birthdate which has not yet happened.')
        print('Please restart the program.')
        cont = False    
    elif birth_year < 1800 :
        raise Exception('If you were born in then you would be dead aleady')
   
    if cont:

    #Make sure month is valid and asign month number to month name to be called later
        if not birth_month.isdigit():
            raise Exception('Your month input needs to be a whole number')
        birth_month = int(birth_month)
    
        #assigning varible to each month and making sure that it is 1-12
        if int(birth_month) > 12 or int(birth_month) < 0:
            raise Exception('That month is too large make sure it is between 1 and 12')
        elif birth_month == 1:
            month = 'January'
        elif birth_month == 2:
            month = 'February'  
        elif birth_month == 3:
            month = 'March'
        elif birth_month == 4:
            month = 'April'
        elif birth_month == 5:
            month = 'May'
        elif birth_month == 6: 
            month = 'June'
        elif birth_month == 7:
            month = 'July'
        elif birth_month == 8:
            month = 'August'
        elif birth_month == 9:
            month = 'September'  
        elif birth_month == 10:
            month = 'October'
        elif birth_month == 11:
            month = 'November'
        elif birth_month == 12:
            month = 'December'
        else:
            print()
            print(f'{birth_month} is an invalid month. (It doesn\'t exist.)')
            print('Please restart the program.')
            cont = False


        if cont:    
            #Make sure day input is valid
            if not birth_day.isdigit():
                raise Exception('Your input needs to be a whole number')
            birth_day = int(birth_day)
            #Day overflow checking based on each month
            if month == "February":
                if birth_day > 28:
                    print()
                    print(f'{birth_day} is an invalid day. (It doesn\'t exist in {month}.)')
                    print('Please restart the program.')
                    cont = False


            elif month == "April" or month == "June" or month == "September" or month == "November":
                if birth_day > 30:
                    print()
                    print(f'{birth_day} is an invalid day. (It doesn\'t exist in {month}.)')
                    print('Please restart the program.')
                    cont = False

            elif month == "January" or month == "March" or month == "May" or month == "July" or month == "August" or month == "October" or month == "December":
                if birth_day > 31:
                    print()
                    print(f'{birth_day} is an invalid day. (It doesn\'t exist in {month}.)')

                    print('Please restart the program.')
                    cont = False

            #check endings per number

            #Finding ending based off divisors
            if cont:
                if birth_day % 10 == 1:
                    end = 'st'
                elif birth_day % 10 == 2:
                    end = 'nd'
                elif birth_day % 10 == 3:
                    end = 'rd'
                else: 
                    end = 'th'
                
                #9/1/2023 is the date we are weighing this against
                year_diff = 2023 - birth_year
                month_diff = 9 - birth_month
                day_diff = 1 - birth_day
                #print(day_diff)
                
                # this will deal with the months and days to find the exact age in years
                if month_diff < 0:
                    year_diff = year_diff - 1
                elif month_diff == 0 and day_diff < 0:
                    year_diff = year_diff - 1
                print('')
                print(f"Hello {name}!")
                print(f"You were born in {birth_year} on the {birth_day}{end} of {month}.")
                print(f'You are {year_diff} years old.')


if __name__ == '__main__':
    main()