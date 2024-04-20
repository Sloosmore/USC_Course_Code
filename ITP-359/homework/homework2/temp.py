import pandas as pd
import sklearn as skt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from hmmlearn.hmm import GaussianHMM
import seaborn as sns


#read in file to DF (Step 2)
df = pd.read_csv('homework/homework2/Temperature data.csv', delimiter=',')
"""      
Year  Anomaly
0  1850-01    -0.36
1  1850-02    -0.12
2  1850-03    -0.23
3  1850-04    -0.19
"""
#to DateTime (Step 3)
df['Time'] = pd.to_datetime(df['Year'], format='%Y-%m')

df = df.drop('Year', axis=1)

#Plot the data (Anonmaly vs Year Step 4) 
fig, ax = plt.subplots()
ax.plot(df['Time'], df['Anomaly'])
ax.set_xlabel('Year')
ax.set_ylabel('Temp Anomaly')
ax.set_title('Temperature Anomaly Over Time')
fig.savefig('temp_anomaly.png')

#Numpy array of anon (Step 5)
np_Anom = np.array(df['Anomaly']).reshape(-1,1)


def nn(np_Anom):
    #scale the data (Step 6)
    scaler = MinMaxScaler()
    np_Anom = scaler.fit_transform(np_Anom)
    
    #create the time window for the data (Step 7)
    window_size = 10
    max_length = np_Anom.shape[0]
    y = np_Anom[window_size:max_length+1].squeeze()
    X = []
    #time window
    for i in range (y.shape[0]):
        appendlist = np_Anom[i:i+window_size]
        X.append(appendlist)
    X = np.array(X).squeeze()
    print(X)
    print(y)
    
    #create the neural network model (Step 8)
    nn_model = tf.keras.Sequential()
    nn_model.add(tf.keras.Input(shape=(window_size,)))
    nn_model.add(tf.keras.layers.Dense(200, activation='relu'))
    nn_model.add(tf.keras.layers.Dense(1000, activation='relu'))
    nn_model.add(tf.keras.layers.Dense(5000, activation='relu'))
    nn_model.add(tf.keras.layers.Dense(2500, activation='relu'))
    nn_model.add(tf.keras.layers.Dense(500, activation='relu'))
    nn_model.add(tf.keras.layers.Dense(1, activation='linear'))
    #mse loss function (step 9)
    nn_model.compile(
        optimizer='adam',
        loss="mse",
        metrics=['accuracy']
    )
    #train the model (Step 10)
    nn_model.fit(X, y, batch_size=40, epochs=30)
    #predict the data and inverse scale temp (Step 11)
    prediction = nn_model.predict(X)
    scale_pred = scaler.inverse_transform(prediction)
    #evaluate the model (Step 12)
    loss, acc = nn_model.evaluate(X, y)
    mse = mean_squared_error(y, scale_pred)
    print("MSE: ", mse)
    print("loss: ", loss)
    print("Acc: ", acc)
    """
    MSE:  0.1503026114530207
    loss:  0.0023833110462874174
    Acc:  0.0009708738070912659
    """
    #plot prediction (Step 13)
    fig, ax = plt.subplots()
    ax.plot(df['Time'], df['Anomaly'], label='Original')
    ax.plot(df['Time'].iloc[window_size::], scale_pred, label='Prediciton')
    ax.set_xlabel('Year')
    ax.set_ylabel('Temp Anomaly')
    ax.set_title('Temperature Anomaly Over Time')
    ax.legend()
    fig.savefig('nn model')

    #predict the next 24 months in the future and plot (Step 14) 
    pred_months = 24
    future_pred_list = y[-window_size:].tolist()
    for i in range(pred_months):
        last_window = np.array(future_pred_list[-window_size:]).reshape(1, -1)  # reshape for prediction
        new_prediction = nn_model.predict(last_window)
        future_pred_list.extend(new_prediction.flatten().tolist())  # flatten and extend list

    future_pred = future_pred_list[-pred_months:]
    future_pred = np.array(future_pred)
    future_pred = scaler.inverse_transform(future_pred.reshape((-1, 1)))
    dates = pd.date_range(start='2023-09', end='2025-09', freq='M')
    future_dates = np.array(dates)

    fig, ax = plt.subplots()
    ax.plot(df['Time'], df['Anomaly'], label='Original')
    ax.plot(df['Time'].iloc[window_size::], scale_pred, label='Prediciton')
    ax.plot(future_dates, future_pred, label='Future_Prediciton')
    ax.set_xlabel('Year')
    ax.set_ylabel('Temp Anomaly')
    ax.set_title('Temperature Anomaly Over Time')
    ax.legend()
    fig.savefig('nn model future')
    
    


def hmm(X, time):
    #train the hmm (Step 15)
    model = GaussianHMM( n_iter=100, n_components=3)
    model.fit(X)
    #predict the states (Step 16)
    pred = model.predict(X)
    state = np.array(pred)
    
    #plot the states (Step 17)
    d = {"state": state, "Anom": X.flatten(), 'time': time}
    hmm_df = pd.DataFrame(data=d)  
    print(hmm_df)
    statelist = []
    datelist = []
    for i in range(3):
        state = hmm_df[hmm_df['state'] == i]["Anom"]
        statelist.append(state)
        date = hmm_df[hmm_df['state'] == i]["time"]
        datelist.append(date)
        
    
    fig, ax = plt.subplots(figsize=(20, 8))
    for i in range(3):
        ax.scatter(datelist[i], statelist[i], label=f'state {i}')
    ax.set_xlabel('Year')
    ax.set_ylabel('Temp Anomaly')
    ax.set_title('Temperature Anomaly Over Time')
    ax.legend()
    fig.tight_layout()
    fig.savefig('hmm')
    
    #plot the transition matrix (Step 18)
    plt.figure(figsize=(10, 8))
    sns.heatmap(model.transmat_, annot=True, cmap='coolwarm')
    plt.title('Heatmap of Transition Matrix')
    plt.xlabel('To State')
    plt.ylabel('From State')
    plt.savefig('hmm heatmap')
    

    #bonus, predict the next 24 months in the future and plot (Step 19)
    last_hidden_state = hmm_df['state'].iloc[-1]

    fut_pred_list = [last_hidden_state]

    mean, std = model.means_, np.sqrt(model.covars_)

    # Loop for future predictions
    for i in range(24):
        trans_probs = model.transmat_[last_hidden_state]

        next_state = np.random.choice(a=len(trans_probs), p=trans_probs)

        # Update last_hidden_state
        last_hidden_state = next_state

        # Generate observable output for the next state
        future_observation = np.random.normal(mean[last_hidden_state][0], std[last_hidden_state][0])
        fut_pred_list.append(future_observation)
        
    dates = pd.date_range(start='2023-09', end='2025-09', freq='M')
    
    fut_pred = np.array(fut_pred_list[1::]).squeeze()

    fig, ax = plt.subplots()
    ax.plot(df['Time'], df['Anomaly'], label='Original')
    ax.plot(dates, fut_pred, label='Future_Prediciton')
    ax.set_xlabel('Year')
    ax.set_ylabel('Temp Anomaly')
    ax.set_title('Temperature Anomaly Over Time')
    ax.legend()
    fig.savefig('hmm model future')
    print('end of hmm model future')

    #plt.tight_layout()

    
    
    
    
    


if __name__ == "__main__":
    #hmm(np_Anom, np.array(df['Time']))
    nn(np_Anom)