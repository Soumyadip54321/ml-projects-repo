'''
This workbook demonstrates working of Gradient Descent to find the best-fit line that captures underline data.
'''

# import necessary packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import math

# setup plotting function
def plot_original_distribution(df: pd.DataFrame):
    '''
    This function plots the distribution of underline prices.
    :param df:
    :return:
    '''
    plt.scatter(x=df['area_sqr_ft'],y=df['price_lakhs'],label='Original distribution')
    plt.xlabel('area_sqr_ft')
    plt.ylabel('price_lakhs')
    plt.title('Distribution of house prices in Lakhs basis area in sq ft.')
    plt.legend()
    plt.show()

def gradient_descent(df: pd.DataFrame, epochs = 10, lr = 0.01):
    '''
    This function implements the gradient descent algorithm. It then finds
    the best fit line that captures underline prices.
    :param df:
    :return: list of losses iteration-wise, final params
    '''

    # set seed for random number generator
    np.random.seed(42)

    # set parameters
    m = np.random.rand(1,1)
    b = np.random.rand(1,1)
    n = df.shape[0]

    # feature scaling
    df['area_sqr_ft'] = (df['area_sqr_ft'] - df['area_sqr_ft'].mean())/df['area_sqr_ft'].std()
    df['price_lakhs'] = (df['price_lakhs'] - df['price_lakhs'].mean())/df['price_lakhs'].std()

    # setup input and ground truth vectors
    x = df['area_sqr_ft'].values.reshape(-1,1)
    y = df['price_lakhs'].values.reshape(-1,1)

    # capture loss per epoch
    loss_history = []
    prev_loss = math.inf
    final_params = np.zeros(2)

    for iter in tqdm(range(epochs)):
        # prediction
        y_hat = np.matmul(x, m) + b

        # compute loss
        l = (np.sum(np.square(y - y_hat)))/n
        loss_history.append((l, iter))

        # update params
        dl_dm = -(2/n)*np.matmul(x.T, (y-y_hat))
        dl_db = -(2/n)*np.sum(y-y_hat)
        m = m - lr*dl_dm
        b = b - lr*dl_db

        if prev_loss > l:
            final_params = (m,b)
            prev_loss = l

    return loss_history, final_params

def plot_distribution(df: pd.DataFrame, final_params: tuple):
    '''
    Function to plot original distribution along with the best-fit line that captures the distribution.
    :param df:
    :return:
    '''

    # plot original distribution
    plt.scatter(x=df['area_sqr_ft'], y=df['price_lakhs'], label='Original distribution')

    x = df['area_sqr_ft'].values.reshape(-1,1)
    m,b = final_params

    # overlay best-fit line
    y_hat = np.matmul(x, m) + b

    plt.scatter(x=x,y=y_hat,label='Best-fit line')
    plt.xlabel('area_sqr_ft')
    plt.ylabel('price_lakhs')
    plt.title('Original distribution and best-fit line')
    plt.legend()
    plt.show()

def plot_loss_curve(loss_history: list):
    '''
    Function to plot the loss curve.
    :param loss_history:
    :return:
    '''

    # fetch losses and iterations
    losses = [loss for loss,_ in loss_history]
    epochs = [epoch for _,epoch in loss_history]

    plt.plot(epochs,losses)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(f'Loss curve with {epochs[-1]+1} epochs')
    plt.show()

if __name__ == '__main__':
    # import data
    home_prices_df = pd.read_csv('Datasets/home_prices.csv')
    loss_history, final_params = gradient_descent(home_prices_df,epochs = 100)
    print(final_params)
    plot_loss_curve(loss_history)
    plot_distribution(home_prices_df, final_params)




