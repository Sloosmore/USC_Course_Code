""" Stan Loosmore
    ITP-449
    H04
    The practical goal of this assignment is to produce visualizations from the data contained within the Palmer Penguins dataset
"""


def main():
    # write your code here
    import pandas as pd 
    import numpy as np 
    import matplotlib.pyplot as plt
    #read in csv
    file_name = 'ITP-449/Homeworks/csv_files/penguins.csv'
    df_peng = pd.read_csv(file_name)


    #Numerical atributes: bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g
    #get individual date on all numerical series 
    bill_length = df_peng.loc[:, 'bill_length_mm']
    bill_depth = df_peng.loc[:, 'bill_depth_mm']
    flipper_length = df_peng.loc[:, 'flipper_length_mm']
    body_mass = df_peng.loc[:, 'body_mass_g']    
    body_mass = body_mass.to_frame()


    '''
    Used to gen scaterplots
    Num_atributes = ['bill_length', 'bill_depth', 'flipper_length', 'body_mass_g']
    for x in range(len(Num_atributes)):
        for y in range(len(Num_atributes)):
            print(f'ax.scatter({Num_atributes[x]}, {Num_atributes[y]})')
    '''
    #create plot matrix
    nbin = 10
    fig, ax = plt.subplots(4, 4)
    ax[0,0].hist(bill_length, bins = nbin)
    ax[0,0].set_xlabel('Bill Length')
    ax[0,1].scatter(bill_length, bill_depth)
    ax[0,2].scatter(bill_length, flipper_length)
    ax[0,3].scatter(bill_length, body_mass)
    ax[1,0].scatter(bill_depth, bill_length)
    ax[1,1].hist(bill_depth, bins = nbin)
    ax[1,1].set_xlabel('Bill Depth')
    ax[1,2].scatter(bill_depth, flipper_length)
    ax[1,3].scatter(bill_depth, body_mass)
    ax[2,0].scatter(flipper_length, bill_length)
    ax[2,1].scatter(flipper_length, bill_depth)
    ax[2,2].hist(flipper_length, bins = nbin)
    ax[2,2].set_xlabel('Flipper length')
    ax[2,3].scatter(flipper_length, body_mass)
    ax[3,0].scatter(body_mass, bill_length)
    ax[3,1].scatter(body_mass, bill_depth)
    ax[3,2].scatter(body_mass, flipper_length)
    ax[3,3].hist(body_mass, bins = nbin)
    ax[3,3].set_xlabel('body mass')
    fig.set_figheight(15)
    fig.set_figwidth(15)
    fig.suptitle('Numerical Comparisons')
    plt.savefig('ITP-449/Homeworks/outputs/penguins_attributes_scatterplot_matrix.png')
    


    #Species Adelie, Chinstrap, Gentoo. Filter each into own df
    df_adelie = df_peng[df_peng['species'] == 'Adelie']
    df_chinstrap = df_peng[df_peng['species'] == 'Chinstrap']
    df_gentoo = df_peng[df_peng['species'] == 'Gentoo']

    #grab the bill length from each new df
    bill_adelie = df_adelie.loc[:, 'bill_length_mm']
    bill_chinstrap = df_chinstrap.loc[:, 'bill_length_mm']
    bill_gentoo = df_gentoo.loc[:, 'bill_length_mm']

    #grab the flipper length from each new df
    flipper_adelie = df_adelie.loc[:, 'flipper_length_mm']
    flipper_chinstrap = df_chinstrap.loc[:, 'flipper_length_mm']
    flipper_gentoo = df_gentoo.loc[:, 'flipper_length_mm']

    #graph the Bill vs Flipper length
    fig, ax = plt.subplots(1, 1)
    ax.scatter(bill_adelie, flipper_adelie, label = 'Adelie')
    ax.scatter(bill_chinstrap, flipper_chinstrap, label = 'Chinstrap')
    ax.scatter(bill_gentoo, flipper_gentoo, label = 'Gentoo')
    ax.legend()
    ax.set_title('Bill vs Flipper length')
    ax.set_xlabel('Bill length (mm)')
    ax.set_ylabel('Flipper length (mm)')
    plt.savefig('ITP-449/Homeworks/outputs/penguins_bill_flipper_by_species.png')

if __name__ == '__main__':
    main()