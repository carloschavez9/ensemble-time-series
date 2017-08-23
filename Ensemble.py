# =============================================================================
#
# Copyright 2017 Carlos Alberto Chavez
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# =============================================================================


# Uncomment these lines to use THEANO as the backend instead of tensorflow
# # Use Theano
# import os
# os.environ["KERAS_BACKEND"] = "theano"
# import keras; import keras.backend
# if keras.backend.backend() != 'theano':
#     raise BaseException("This script uses other backend")
# else:
#     keras.backend.set_image_dim_ordering('th')
#     print("Backend ok")

import numpy as np
import pandas as pd
import math
import random
import time
import gc
from keras.models import Sequential
from keras.layers import LSTM, GRU, SimpleRNN, Dense
from keras.utils import plot_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz  # with pydot
import warnings

warnings.filterwarnings("ignore")

# ===============================
# Hyperparameters
# ===============================

train_size_percentage = 0.9  # Training size
mutation_rate = 0.1  # Mutation rate for GA
min_mutation_momentum = 0.0001  # Min mutation momentum
max_mutation_momentum = 0.1  # Max mutation momentum
min_population = 20  # Min population for GA
max_population = 50  # Max population for GA
num_Iterations = 3  # Number of iterations to evaluate GA
look_back = 1  # Num of timespaces to look back for training and testing
max_dropout = 0.2  # Maximum percentage of dropout
min_num_layers = 1  # Min number of hidden layers
max_num_layers = 10  # Max number of hidden layers
min_num_neurons = 10  # Min number of neurons in hidden layers
max_num_neurons = 100  # Max number of neurons in hidden layers
min_num_estimators = 100  # Min number of random forest trees
max_num_estimators = 500  # Max number of random forest trees
force_gc = True  # Forces garbage collector
rnn_epochs = 1  # Epochs for RNN

# ===============================
# Constants and variables
# ===============================

datasets = ['data/daily-births.csv', 'data/tesco-stock.csv', 'data/weather-madrid.csv']
optimisers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam']
rnn_types = ['LSTM', 'GRU', 'SimpleRNN']


# fix random seed for reproducibility
# np.random.seed(0)

def create_dataset(dataset, look_back=1):
    """
    Converts an array of values into a dataset matrix
    :param dataset:
    :param look_back:
    :return:
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])

    collect_gc()

    return np.array(dataX), np.array(dataY)


def collect_gc():
    """
    Forces garbage collector
    :return:
    """
    if force_gc:
        gc.collect()


def load_dataset(dataset_path):
    """
    Loads a dataset with training and testing arrays
    :param dataset_path:
    :return:
    """
    # Load dataset
    dataset = pd.read_csv(dataset_path, parse_dates=True, index_col=0)
    dataset = dataset.values  # as numpy array
    dataset = dataset.astype('float64')
    # Normalise the dataset
    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * train_size_percentage)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    # reshape into X=t and Y=t+1
    train_x, train_y = create_dataset(train, look_back)
    test_x, test_y = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    train_x_stf = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    test_x_stf = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
    train_x_st = np.reshape(train_x, (train_x.shape[0], 1))
    test_x_st = np.reshape(test_x, (test_x.shape[0], 1))

    return dataset, scaler, train_x_stf, train_x_st, train_y, test_x_stf, test_x_st, test_y


def generate_rnn(hidden_layers):
    """
    Generates a RNN using an array of hidden layers including the number of neurons for each layer
    :param hidden_layers:
    :return:
    """
    # Create and fit the RNN
    model = Sequential()
    # Add input layer
    model.add(Dense(1, input_shape=(1, look_back)))

    # Add hidden layers
    for i in range(len(hidden_layers)):
        neurons_layer = hidden_layers[i]
        # Randomly select rnn type of layer
        rnn_type_index = random.randint(0, len(rnn_types) - 1)
        rnn_type = rnn_types[rnn_type_index]

        dropout = random.uniform(0, max_dropout)  # dropout between 0 and max_dropout
        return_sequences = i < len(hidden_layers) - 1  # Last layer cannot return sequences when stacking

        # Select and add type of layer
        if rnn_type == 'LSTM':
            model.add(LSTM(neurons_layer, dropout=dropout, return_sequences=return_sequences))
        elif rnn_type == 'GRU':
            model.add(GRU(neurons_layer, dropout=dropout, return_sequences=return_sequences))
        elif rnn_type == 'SimpleRNN':
            model.add(SimpleRNN(neurons_layer, dropout=dropout, return_sequences=return_sequences))

    collect_gc()

    # Add output layer
    model.add(Dense(1))
    return model


def evaluate_rnn(model, train_x, test_x, train_y, test_y, scaler, optimiser):
    """
    Evaluates the RNN model using the training and testing data
    :param model:
    :param train_x:
    :param test_x:
    :param train_y:
    :param test_y:
    :param scaler:
    :param optimiser:
    :return:
    """
    model.compile(loss='mean_squared_error', optimizer=optimiser)
    model.fit(train_x, train_y, epochs=rnn_epochs, batch_size=1, verbose=2)
    # Forecast
    train_predict = model.predict(train_x)
    test_predict = model.predict(test_x)
    # Invert forecasts
    train_predict = scaler.inverse_transform(train_predict)
    train_y = scaler.inverse_transform([train_y])
    test_predict = scaler.inverse_transform(test_predict)
    test_y = scaler.inverse_transform([test_y])
    # Calculate RMSE for train and test
    train_score = math.sqrt(mean_squared_error(train_y[0], train_predict[:, 0]))
    # print('Train Score: %.2f RMSE' % (train_score))
    test_score = math.sqrt(mean_squared_error(test_y[0], test_predict[:, 0]))
    # print('Test Score: %.2f RMSE' % (test_score))
    model.train_score = train_score
    model.test_score = test_score

    return train_score, test_score, train_predict, test_predict


def crossover_rnn(model_1, model_2):
    """
    Executes crossover for the RNN in the GA for 2 models, modifying the first model
    :param model_1:
    :param model_2:
    :return:
    """
    # new_model = copy.copy(model_1)
    new_model = model_1

    # Probabilty of models depending on their RMSE test score
    # Lower RMSE score has higher prob
    test_score_total = model_1.test_score + model_2.test_score
    model_1_prob = 1 - (model_1.test_score / test_score_total)
    model_2_prob = 1 - model_1_prob
    # Probabilities of each item for each model (all items have same probabilities)
    model_1_prob_item = model_1_prob / (len(model_1.layers) - 2)
    model_2_prob_item = model_2_prob / (len(model_2.layers) - 2)

    # Number of layers of new generation depend on probability of each model
    num_layers_new_gen = int(model_1_prob * (len(model_1.layers) - 1) + model_2_prob * (len(model_2.layers) - 1))

    # Create list of int with positions of the layers of both models.
    cross_layers_pos = []
    # Create list of weights
    weights = []
    # Add positions of layers for model 1. Input and ouput layer are not added.
    for i in range(2, len(model_1.layers)):
        mod_item = type('', (), {})()
        mod_item.pos = i
        mod_item.model = 1
        cross_layers_pos.append(mod_item)
        weights.append(model_1_prob_item)

    # Add positions of layers for model 2. Input and ouput layer are not added.
    for i in range(2, len(model_2.layers)):
        mod_item = type('', (), {})()
        mod_item.pos = i
        mod_item.model = 2
        cross_layers_pos.append(mod_item)
        weights.append(model_2_prob_item)

    collect_gc()

    # If new num of layers are larger than the num crossover layers, keep num of crossover layers
    if num_layers_new_gen > len(cross_layers_pos):
        num_layers_new_gen = len(cross_layers_pos)

    # Randomly choose num_layers_new_gen layers of the new list
    cross_layers_pos = list(np.random.choice(cross_layers_pos, size=num_layers_new_gen, replace=False, p=weights))

    # Add both group of hidden layers to new group of layers using previously chosen layer positions of models
    cross_layers = []
    for i in range(len(cross_layers_pos)):
        mod_item = cross_layers_pos[i]
        if mod_item.model == 1:
            cross_layers.append(model_1.layers[mod_item.pos])
        else:
            cross_layers.append(model_2.layers[mod_item.pos])

    collect_gc()

    # Add input layer randomly from parent 1 or parent 2
    bit_random = random.randint(0, 1)
    if bit_random == 0:
        cross_layers.insert(0, model_1.layers[0])
    else:
        cross_layers.insert(0, model_2.layers[0])

    bit_random = random.randint(0, 1)
    if bit_random == 0:
        cross_layers.append(model_1.layers[len(model_1.layers) - 1])
    else:
        cross_layers.append(model_2.layers[len(model_2.layers) - 1])

    # Set new layers
    new_model.layers = cross_layers

    return new_model


def mutate_rnn(model):
    """
    Mutates the RNN model
    :param model:
    :return:
    """
    for i in range(len(model.layers)):
        # Mutate randomly each layer
        bit_random = random.uniform(0, 1)

        if bit_random <= mutation_rate:
            weights = model.layers[i].get_weights()  # list of weights as numpy arrays
            # calculate mutation momentum
            mutation_momentum = random.uniform(min_mutation_momentum, max_mutation_momentum)
            new_weights = [x * mutation_momentum for x in weights]
            model.layers[i].set_weights(new_weights)

    collect_gc()


def save_plot_model_rnn(model):
    """
    Saves the plot of the RNN model
    :param model:
    :return:
    """
    plot_model(model, show_shapes=True)


def generate_rf(estimators):
    """
    Generates a Random Forest with the number of estimators to use
    :param estimators:
    :return:
    """
    # Create and fit the RF
    model = RandomForestRegressor(n_estimators=estimators, criterion='mse', max_depth=None, min_samples_split=2,
                                  min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=True,
                                  oob_score=False, n_jobs=1, random_state=None, verbose=0)

    return model


def evaluate_rf(model, train_x, test_x, train_y, test_y, scaler):
    """
    Evaluates the Random Forest with training and testing data
    :param model:
    :param train_x:
    :param test_x:
    :param train_y:
    :param test_y:
    :param scaler:
    :return:
    """
    model.fit(train_x, train_y)
    # Forecast
    train_predict = model.predict(train_x)
    test_predict = model.predict(test_x)
    # Invert forecasts
    train_predict = scaler.inverse_transform(train_predict)
    train_y = scaler.inverse_transform([train_y])
    test_predict = scaler.inverse_transform(test_predict)
    test_y = scaler.inverse_transform([test_y])
    # Calculate RMSE for train and test
    train_score = math.sqrt(mean_squared_error(train_y[0], train_predict[:]))
    # print('Train Score: %.2f RMSE' % (train_score))
    test_score = math.sqrt(mean_squared_error(test_y[0], test_predict[:]))
    # print('Test Score: %.2f RMSE' % (test_score))
    model.train_score = train_score
    model.test_score = test_score

    return train_score, test_score, train_predict, test_predict


def crossover_rf(model_1, model_2):
    """
    Executes crossover for the RF in the GA for 2 models, modifying the first model
    :param model_1:
    :param model_2:
    :return:
    """
    # new_model = copy.copy(model_1)
    new_model = model_1

    # Probabilty of models depending on their RMSE test score
    test_score_total = model_1.test_score + model_2.test_score
    model_1_prob = 1 - model_1.test_score / test_score_total
    model_2_prob = 1 - model_1_prob

    # New estimator is the sum of both estimators times their probability
    new_model.n_estimators = math.ceil(model_1.n_estimators * model_1_prob + model_2.n_estimators * model_2_prob)

    return new_model


def mutate_rf(model):
    """
    Mutates the Random Forest
    :param model:
    :return:
    """
    # Mutate randomly the estimator
    bit_random = random.uniform(0, 1)

    if bit_random <= mutation_rate:
        # calculate mutation momentum
        mutation_momentum = random.uniform(min_mutation_momentum, max_mutation_momentum)
        # Mutate estimators
        model.n_estimators = model.n_estimators + math.ceil(model.n_estimators * mutation_momentum)


def save_plot_model_rf(model):
    """
    Saves the plot of the Random Forest model
    :param model:
    :return:
    """
    for i in range(len(model.estimators_)):
        estimator = model.estimators_[i]
        out_file = open("trees/tree-" + str(i) + ".dot", 'w')
        export_graphviz(estimator, out_file=out_file)
        out_file.close()


def ensemble_stacking(model_1_values, model_2_values, test, scaler):
    """
    Ensemble result of 2 models using stacking and averaging.
    Takes both model predictions, averages them and calculates the new RMSE
    :param model_1_values:
    :param model_2_values:
    :return:
    """
    # Generates the stacking values by averaging both predictions
    stacking_values = []
    for i in range(len(model_1_values)):
        stacking_values.append((model_1_values[i][0] + model_2_values[i]) / 2)

    test = scaler.inverse_transform([test])
    rmse = math.sqrt(mean_squared_error(test[0], stacking_values))
    return stacking_values, rmse


def evaluate_ga(dataset):
    """
    Evaluates and generates the ensemble model using Genetic Algorithms
    :param dataset:
    :return:
    """
    print('#-----------------------------------------------')
    print('  ', dataset)
    print('#-----------------------------------------------')

    dataset, scaler, train_x_stf, train_x_st, train_y, test_x_stf, test_x_st, test_y = load_dataset(dataset)
    start = time.clock()  # Start Timer
    num_population = random.randint(min_population, max_population)  # Number of RNN to evaluate
    # == 1) Generate initial population for RNN and Random Forest
    population_rnn = []
    population_rf = []
    start_ga_1 = time.clock()  # Start Timer
    for i in range(num_population):
        # -- RNN
        # Generate random topology configuration
        num_layers = random.randint(min_num_layers, max_num_layers)
        hidden_layers = []
        for j in range(num_layers):
            num_neurons = random.randint(min_num_neurons, max_num_neurons)
            hidden_layers.append(num_neurons)

        collect_gc()

        # Generate and add rnn model to population
        model_rnn = generate_rnn(hidden_layers)
        population_rnn.append(model_rnn)

        # -- RF
        # Generate random number of estimators for RF
        num_estimators = random.randint(min_num_estimators, max_num_estimators)

        # Generate and add rf model to population
        model_rf = generate_rf(num_estimators)
        population_rf.append(model_rf)

    end_ga_1 = time.clock() - start_ga_1  # End Timer
    print('Generate Initial population Time_Taken:%.3f' % end_ga_1)

    collect_gc()
    # print(len(population))

    best_rmse_rnn = float("inf")
    best_rmse_rf = float("inf")
    best_rnn_model = None
    best_test_predict_rnn = None
    best_rf_model = None
    best_test_predict_rf = None
    # Evaluate fitness for
    for i in range(num_Iterations):
        print('=================================================================================================')
        print(' iteration: %d, total iterations: %d, population size: %d ' % (i + 1, num_Iterations, num_population))
        print('=================================================================================================')
        # train_score, test_score = float("inf"), float("inf")
        # == 2)  Evaluate fitness for population
        start_ga_2 = time.clock()  # Start Timer
        for j in range(num_population):
            # Evaluate fitness for RNN
            rnn_model = population_rnn[j]
            train_score_rnn, test_score_rnn, train_predict_rnn, test_predict_rnn = evaluate_rnn(rnn_model, train_x_stf,
                                                                                                test_x_stf, train_y,
                                                                                                test_y, scaler,
                                                                                                optimisers[0])
            # print('test predictions RNN: ', test_predict_rnn)
            print('test_score RMSE RNN:%.3f ' % test_score_rnn)

            if test_score_rnn < best_rmse_rnn:
                best_rmse_rnn = test_score_rnn
                # best_rnn_model = copy.copy(rnn_model)
                best_rnn_model = rnn_model
                best_test_predict_rnn = test_predict_rnn

            # Evaluate fitness for RF
            rf_model = population_rf[j]
            train_score_rf, test_score_rf, train_predict_rf, test_predict_rf = evaluate_rf(rf_model, train_x_st,
                                                                                           test_x_st, train_y, test_y,
                                                                                           scaler)
            # print('test predictions RF: ', test_predict_rf)
            print('test_score RMSE RF:%.3f ' % test_score_rf)

            if test_score_rf < best_rmse_rf:
                best_rmse_rf = test_score_rf
                # best_rf_model = copy.copy(rf_model)
                best_rf_model = rf_model
                best_test_predict_rf = test_predict_rf

        end_ga_2 = time.clock() - start_ga_2  # End Timer
        print('Evaluate Fitness population Time_Taken:%.3f' % end_ga_2)

        collect_gc()

        print('Temporal Best RMSE RNN:%.3f' % best_rmse_rnn)
        print('Temporal Best predictions: ', [x[0] for x in best_test_predict_rnn])
        print('Temporal Best RMSE RF:%.3f' % best_rmse_rf)
        print('Temporal Best predictions: ', [x for x in best_test_predict_rf])

        # == 3) Create new population with new generations
        # Every generation will use the current best RNN and best RF to mate
        start_ga_3 = time.clock()  # Start Timer
        for pop_index in range(num_population):
            # Select parents for mating
            # Element at pop_index as parent. This will be replaced with the new generation
            rnn_model_1 = population_rnn[pop_index]
            rf_model_1 = population_rf[pop_index]
            # 2 parent is the best found so far
            rnn_model_2 = best_rnn_model
            rf_model_2 = best_rf_model

            # == 4) Create new generation with crossover
            new_rnn_model = crossover_rnn(rnn_model_1, rnn_model_2)
            new_rf_model = crossover_rf(rf_model_1, rf_model_2)

            # == 5) Mutate new generation
            mutate_rnn(new_rnn_model)
            mutate_rf(new_rf_model)

            # Replace current model in population
            population_rnn[pop_index] = new_rnn_model
            population_rf[pop_index] = new_rf_model

        end_ga_3 = time.clock() - start_ga_3  # End Timer
        print('Generate new population Time_Taken:%.3f' % end_ga_3)

        collect_gc()

    collect_gc()

    end = time.clock() - start  # End Timer

    print('=============== BEST RNN ===============')
    print('Best predictions: ', [x[0] for x in best_test_predict_rnn])
    print('Best RMSE:%.3f Time_Taken:%.3f' % (best_rmse_rnn, end))
    save_plot_model_rnn(best_rnn_model)

    print('=============== BEST RF ===============')
    print('Best predictions: ', [x for x in best_test_predict_rf])
    print('Best RMSE:%.3f Time_Taken:%.3f' % (best_rmse_rf, end))

    save_plot_model_rf(best_rf_model)
    # print(best_rf_model.get_params(deep=True))

    # Ensemble
    print('=============== Ensemble ===============')
    averaging_values, rmse = ensemble_stacking(best_test_predict_rnn, best_test_predict_rf, test_y, scaler)
    print('Ensemble averaging_values: ', averaging_values)
    print('Ensemble rmse: ', rmse)


def evaluate_bptt(dataset):
    """
    Evaluates and generates a RNN model using BPTT
    :param dataset:
    :return:
    """
    print('#-----------------------------------------------')
    print('  ', dataset)
    print('#-----------------------------------------------')

    dataset, scaler, train_x_stf, train_x_st, train_y, test_x_stf, test_x_st, test_y = load_dataset(dataset)
    start = time.clock()  # Start Timer

    # Generate a 1 hidden layer configuration
    hidden_layers = [1]
    # Generate and add rnn model to population
    model_rnn = generate_rnn(hidden_layers)

    train_score_rnn, test_score_rnn, train_predict_rnn, test_predict_rnn = evaluate_rnn(model_rnn, train_x_stf,
                                                                                        test_x_stf, train_y,
                                                                                        test_y, scaler,
                                                                                        optimisers[0])

    end = time.clock() - start  # End Timer

    print('Predictions: ', [x[0] for x in test_predict_rnn])
    print('RMSE:%.3f Time_Taken:%.3f' % (test_score_rnn, end))
    save_plot_model_rnn(model_rnn)


def main():
    """
    Main execution
    :return:
    """
    evaluate_ga(datasets[0])
    # evaluate_ga(datasets[1])
    # evaluate_ga(datasets[2])

    # evaluate_bptt(datasets[0])
    # evaluate_bptt(datasets[1])
    # evaluate_bptt(datasets[2])


if __name__ == "__main__":
    main()
