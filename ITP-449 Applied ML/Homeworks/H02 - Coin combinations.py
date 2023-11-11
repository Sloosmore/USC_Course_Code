""" Stan Loosmore
    ITP-449
    H02
    Write a program to compute and print all possible combinations of change for a given amount of money. Denominations to be considered: quarter, dime, nickel, and penny.
"""


def main():
    # this will start the opening section and give a description 
    print("""This program calculates the number of coin combinations
you can create from a given amount of money.
""")
    #this will be our program input
    money_input = input('Enter an amount of money: ')

    #this is a floor function that rounds down to the nearest number which will be helpful for the range of the for loop
    def floor(num):
        return int(num // 1)
    #this fuction will calculate the combinations 
    def innerf(money_input): 

        combinations = 0
        tot_pennies = money_input * 100
        # This will go through every combination and see if that can sum to the input amount
        for tfc in range(floor(tot_pennies/25) + 1):
            for tc in range(floor(tot_pennies/10) + 1):
                for fvc in range(floor(tot_pennies/5) + 1):
                    for c in range(floor(tot_pennies/1) + 1):
                        if tot_pennies - (25*tfc + 10*tc + 5*fvc + c) == 0:
                        #this will add one to the combinations if true
                            combinations += 1
        print(f'The total number of combinations for ${money_input} is {combinations}.')
    #this fucntion will take an input and return it if it can be a float 
    def is_float(value):
        try:
            return float(value)
        #if it can't be a float then it will call the function again untill it is
        except ValueError:
            retry_money_input = input('That is not a valid number. Enter an amount of money: ')
            return is_float(retry_money_input)
    
    #call the float function from the original input
    float_money = is_float(money_input)
    #after it's a float then use function to find the combinaions
    innerf(float_money)



if __name__ == '__main__':
    main()