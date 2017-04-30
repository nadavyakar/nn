import itertools
from random import shuffle
import os
from csv import reader
from numpy import exp
import math
from random import uniform
import time
import sys

LETTERS_LIST = ["Aleph", "Bet", "Gimmel", "Dalet", "He", "Vav", "Kaf", "Lamed"]

class Activation:
    def __call__(self, weights, input_vector):
        '''
        :param weights: weights, including the theta (bias) as the last weight
        :param input_vector: input to the neuron
        '''
        return sum(map(lambda p: p[0]*p[1], zip(weights,input_vector+[1.0])))

class Transfer:
    def __init__(self, c=10):
        self.c = c

    def __call__(self, val):
        try:
            x = 1.0 / (1.0 + exp(-self.c*val))
        except Exception:
            return 0 if val<0 else 1
        return x

    def derivative(self, output):
         return (1.0 - output) * output

class Neuron:
    def __init__(self, activation, transfer, weights, weights_from_prev_layer_neurons):
        '''
        :param weights: weights of the edges leading to the neuron, including the theta (bias) as the last weight
        '''
        self.activation = activation
        self.transfer = transfer
        self.weights = weights
        self.weights_from_prev_layer_neurons = weights_from_prev_layer_neurons
        self.output = None
        self.lmbda = None

    def get_weight_from(self, neuron):
        '''
        :param neuron: the neuron that has a weighted edge to this one
        '''
        return self.weights[neuron]
        pass

    def calc_output(self, input):
        '''
        calc_output for he current neuron with the input vector 'input',
        considering its weights and using its transfer method
        '''
        self.output = self.transfer(self.activation(self.weights, input))
        return self.output

    def calc_delta(self, **kwargs):
        '''
        calculate the lmbda error per the expected output,
        or per the next layers neurons' lmbda if no expected result is given
        :param kwargs: 'expected' = the expected result
                        'next_layer' = the next layer of neurons this neuron has edges to
        '''
        if 'expected' in kwargs:
            expected_result = kwargs['expected']
            err = expected_result - self.output
        else:
            # if its not an output neuron, calculate the weighted lambdas
            # that comes out of this neuron to the next layer's neurons
            next_layer_neurons = kwargs['next_layer']
            err = sum([ next_layer_neuron.weights_from_prev_layer_neurons[self] *
                        next_layer_neuron.lmbda
                        for next_layer_neuron in next_layer_neurons ])
        self.lmbda = self.transfer.derivative(self.output) * err

    def adapt_weights(self, input_vector, alpha):
        '''
        adapt each incomming weight to the neuron per its precalculated lmbda,
        the input input_vector from the previous layer, and the learning rate, alpha
        :param input_vector: the NN input / the outcome of the former layer
        '''
        input_and_bias_input = input_vector+[1.0]
        self.weights = [ weight + (alpha * self.lmbda * input_val)
                         for weight,input_val in zip(self.weights, input_and_bias_input)]

class NN:
    def __init__(self, neurons_activation, neurons_transfer, nn_weights,dump_file):
        '''
        create a network with len(layers) layers + a single output neutron the last layer leads to
        :param nn_weights: tuple of of incoming weights per neuron + its bias, in a tuple of neurons per layer, in a tuple of layers.
        eg. - given 2 layers with 2,3 neurons each where the input layer expects an input vector of size 4:
        (
         (
          (1,2,3,4,4.5)
          (5,6,7,8,8.5)
         ),
         (
          (9,10,10.5),
          (11,12,12.5),
          (13,14,14.5)
         ),
          (
           (15,16,17,17.5)
          )
        )
        '''
        self.layers = []
        for layer_weights in nn_weights:
            # here we initialize a layer with the cocrresponding neurons which in turn are initialized with their weights
            self.layers.append(
                [ Neuron(neurons_activation,
                         neurons_transfer,
                         neuron_weights,
                         # this is a mapping between the neurons from the previous layer to the one just initializeed,
                         #  and the weights on the corresponding edges, later used for the error correction process:
                         dict(zip(self.layers[-1], neuron_weights)) if self.layers else None)
                  for neuron_weights in layer_weights ])
        self.dump_file = dump_file

    def propagate_forward(self, input_vector):
        latest_layer_output = input_vector
        for i in range(len(self.layers)):   #   propagate input_vector through the network layers
            layer = self.layers[i]
            latest_layer_output = [ neuron.calc_output(latest_layer_output ) for neuron in layer ]
        return latest_layer_output

    def calc_net_lambdas(self, expected_outputs):
        '''
        calculate the lambdas of each neurons giver the expected output
        :param T:
        :return:
        '''
        i = len(self.layers)-1
        # calculate the error for the output neurons
        for neuron,expected_output in zip(self.layers[-1], expected_outputs):
            neuron.calc_delta(expected=expected_output)
        # back propagate the error using a one-at-a-time propagation method over the rest of the
        # non-output layer from the layer before the output back to the 1st one:
        next_layer = self.layers[-1]
        for layer in reversed(self.layers[:-1]):
            i-=1
            for neuron in layer:
                neuron.calc_delta(next_layer=next_layer)
            next_layer = layer

    def calc_net_weights(self, input_vector, alpha):
        '''
        recalibrate each edge's weight based upon the latest calculated lambda
        :param alpha: the learning rate
        '''
        input = input_vector
        for i in range(len(self.layers)):
            layer = self.layers[i]
            for neuron in layer:
                neuron.adapt_weights(input, alpha)
            # curr lalyes' neurons output PER THE FORMER WEIGHTS,
            # to be used to reclibrate the next layers' neurons' weights
            input = [ neuron.output for neuron in layer ]

    def correct_error_one_at_a_time(self, input_vector, T, alpha):
        '''
        recalibrate the network's weights per the training input vector, input_vector, and its expected output
        '''
        self.calc_net_lambdas(T)
        self.calc_net_weights(input_vector, alpha)

    def run_back_propagation(self, training_set, test_set, epochs_num, alpha, min_soft_max_limit):
        '''
        :param training_set: a tuple of a vector (tuple) input_vector of training row fields, and t - the actual result
        '''
        self.train(training_set, epochs_num, alpha, test_set,min_soft_max_limit)
        return self.propagate_forward_and_calc_err(test_set,min_soft_max_limit)

    def propagate_forward_and_calc_err(self, test_set, min_soft_max_limit):
        '''
        execute a forward propagate process overe the given test set, and report the average error -
        the percentage of test set for which the network did not give a proper identification
        '''
        calculated_rows=[]
        sum_correct = 0
        for input_vector ,expected_vector in test_set:
            actual_vector = self.propagate_forward(input_vector )
            max_actual_vector_val = max(actual_vector)
            if expected_vector.index(max(expected_vector)) == actual_vector.index(max_actual_vector_val) \
                    and max_actual_vector_val >= min_soft_max_limit:
                correct = 1
            else:
                correct = 0
            sum_correct += correct
            calculated_rows.append((input_vector, expected_vector, actual_vector, correct))
        avg_err = (len(calculated_rows) - sum_correct) / len(calculated_rows)
        return calculated_rows, avg_err

    def train(self, training_set, epochs_num, alpha, test_set, min_soft_max_limit):
        '''
        train the network for the given amount of epochs per fold or until 3 consecutives insufficient improvements
        (a gap lesser than 0.001 from the last one) are met.
        These tests of accuricy are done against the test set for each of the 10 first epochs, and then each 10 epochs
        :param epochs_num: the amount of epochs to train the network with for each fold
        :param min_soft_max_limit: a limit that only above it the maximum cell in the output
        vecotr is considered to be viable to compare against
        '''
        latest_avg = 9999
        # the amount of insufficient improvements the iteration will go on untill it will go on to the next fold
        allowed_time_for_insufficient_improvement = 3
        for epoch_num in range(epochs_num):
            epoch_start_time = time.time()
            for input_vector, expected_output in training_set:
                self.propagate_forward(input_vector)
                self.correct_error_one_at_a_time(input_vector, expected_output, alpha)

            rows, avg_err = self.propagate_forward_and_calc_err(test_set, min_soft_max_limit)
            log("epoch {} average error {:.2f}% took {:.2f} seconds".format(epoch_num, avg_err * 100, time.time() - epoch_start_time),dump_file=self.dump_file)
            if epoch_num<10 or epoch_num % round(epochs_num / 100)==0:
                if math.fabs(latest_avg - avg_err)<=0.001:
                    allowed_time_for_insufficient_improvement-=1
                else:
                    allowed_time_for_insufficient_improvement=3
                latest_avg = avg_err
                if allowed_time_for_insufficient_improvement==0:
                    break

def normalize(data):
    '''
    Rescale dataset columns to the range 0-1
    '''
    # find min and max
    min_max_list = [[min(column), max(column)] for column in zip(*data)]
    response_data = []
    for row_idx in range(len(data)):
        normalized_row = []
        for i in range(len(data[0])-1):
            normalized_row.append((float(data[row_idx][i]) - min_max_list[i][0]) /
                                  (float(min_max_list[i][1] - min_max_list[i][0])))
        response_data.append(normalized_row+[data[row_idx][-1]])
    return response_data

def load_csv(file_path):
    """
    :param file_path: The full path of file
    :return: Tha data in csv's file
    """
    dataset = []
    with open(file_path, 'r') as csv_file:
        csv_reader = reader(csv_file, delimiter=';')
        for row in csv_reader:
            dataset.append(row)
    # Remove the title columns in dataset
    if not str(dataset[0][1]).isnumeric():
        dataset.pop(0)
    return [ list(map(float,row)) for row in dataset]

def load_and_categorize_letters(folder_path):
    data = []
    for letter_folder_name in os.listdir(folder_path):
        for letter_file_name in os.listdir(os.path.join(folder_path,letter_folder_name)):
            vectorized_letter = []
            with open(os.path.join(folder_path, letter_folder_name, letter_file_name), 'r') as letter_file:
                for line in letter_file:
                    vectorized_letter += list(map(int,list(line[:-1])))
            letter_category_vector = [ 1 if letter==letter_folder_name else 0 for letter in LETTERS_LIST]
            data.append((vectorized_letter,letter_category_vector))
    return data

def log(s, dump_file, title=""):
    with open(dump_file,'a') as f:
        if title!="":
            title+='\n'
        f.write(title+str(s)+'\n')

def transfor_expected_column_to_vector(dataset,v_len):
    '''
    return a new dataset where the last column holding the expected data,
    it represented by a vector where the bit at the index equals to the expected value is 1 and the rest are zeros
    '''
    return [ (row[:-1], [ 1 if i==row[-1] else 0 for i in range(v_len)]) for row in dataset ]

def add_clasification_column(data,classification_value):
    '''
    adds to each row in the dataset a column which labels it with the given value of classification_value
    '''
    return list(map(lambda x: x+[classification_value],data))

def execute_setup(transfer_func_C, num_of_ten_folds_iterations, alpha, epochs_num,
                  architecture, data, dump_file, min_soft_max_limit):
    '''
    create a NN per the outlined setup and run it for the given amount of epochs specified
    '''
    try:
        learning_start_time = time.time()
        nn = NN(Activation(), Transfer(transfer_func_C), architecture, dump_file=dump_file)
        folders_errors=[]
        for ten_fold_iteration in range(num_of_ten_folds_iterations):
            log("fold #{}".format(ten_fold_iteration), dump_file=dump_file)
            train_set = list(itertools.chain.from_iterable(data[:ten_fold_iteration] + data[ten_fold_iteration + 1:]))
            test_set = data[ten_fold_iteration]
            rows, avg_err = nn.run_back_propagation(train_set, test_set, epochs_num, alpha, min_soft_max_limit)
            nn = NN(Activation(), Transfer(transfer_func_C), architecture, dump_file=dump_file)
            folders_errors.append(avg_err*100)
        folds_average_error = (sum(folders_errors)/num_of_ten_folds_iterations)
        folds_deviation = math.sqrt(sum(map(lambda x: (x-folds_average_error)**2, folders_errors))/num_of_ten_folds_iterations)
        total_time_sec = time.time() - learning_start_time
        log("average error: {:.2f}%\n"
            "average deviation over the folds: {:.2f}\n"
            "learning time for all folds: {}:{}".format(folds_average_error, folds_deviation,
                                                        int(total_time_sec/60),int(total_time_sec%60)),
             dump_file=dump_file)
    except Exception as e:
        log(e, dump_file=dump_file)
        raise e

def main():
    # here the user should supply all the needed parameters for the running of the network.
    # data_type: defines the type of data preprocessing needed to be done in order to work on the data - wines are categorized per the ,
    # and letters are categorized per the folder they reside in
    argvs = sys.argv[1:]
    argsdict = {}
    for argv in argvs:
        (arg, val) = argv.split("=")
        argsdict[arg] = val
    bad_input = False
    exception_ = None
    try:
        data_type = argsdict.get('data_type',None)
        data_path_white_wine = argsdict.get('data_path_white_wine',None)
        data_path_red_wine = argsdict.get('data_path_red_wine',None)
        data_path_letters = argsdict.get('data_path_letters',None)
        log_file=argsdict.get('log_file',None)
        num_of_folds=int(argsdict['num_of_folds'])
        epochs_num = int(argsdict['epochs'])
        alpha = float(argsdict['learning_rate'])
        num_of_layers = int(argsdict['layers_num'])
        neurons_hidden_1 = int(argsdict.get('neurons_hidden_1',-1))
        neurons_hidden_2 = int(argsdict.get('neurons_hidden_2',-1))
        neurons_out = int(argsdict['neurons_out'])
        num_weights_input_layer = int(argsdict['num_weights_input_layer'])
    except Exception as exception:
        bad_input = True
        exception_=exception
    if bad_input or len(argvs) == 0 :
        print("missing/too many arguments! please supply all input arguments, and in the following format:\n"
              "hw1.py "
              "\ndata_type=[WINE|LETTERS] "
              "\ndata_path_white_wine=<data_folder_for_white_wine (for data_type=WINE)> "
              "\ndata_path_red_wine=<data_folder_for_red_wine (for data_type=WINE)> "
              "\ndata_path_letters=<data_folder_for_letters. For data  categorization, this folder must contains the letters files sorted under each of the following subfolders: Aleph, Bet, Gimmel, Dalet, He, Vav, Kaf, Lamed (for data_type=LETTERS)> "
              "\nlog_file=<full path to the log file which would contains the output of the NN>"
              "\nnum_of_folds=<num_of_folds to divide the data by>"
              "\nepochs=num_epochs_per_fold "
              "\nlearning_rate=number "
              "\nlayers_num=<number_of_layers(1-3)> "
              "\nneurons_hidden_1=<positive_number_of_neurons_in_first_hidden_layer (for 2-3 layers)> "
              "\nneurons_hidden_2=<positive_number_of_neurons_in_second_hidden_layer (for 3 layers)> "
              "\nneurons_out=<positive_number_of_neurons_in_output_layer> "
              "\nnum_weights_input_layer=<number of weights per neuron in the input layer. must be equal to the number of columns the NN should learn from each input row from the data>"
              "\nfor your consideration: the command line can't parse empty spaces in the middle of path - spaces separates between command line arguments."
              "\n\nexample - wines:"
              "\npython3 hw1.py data_type=WINE data_path_white_wine='/home/nadav/Downloads/WineQualityDataset/winequality-white.csv' "
              "data_path_red_wine='/home/nadav/Downloads/WineQualityDataset/winequality-red.csv' log_file='/tmp/log.txt' "
              "epochs=1000 learning_rate=0.3 layers_num=3 neurons_hidden_1=20 neurons_hidden_2=6 neurons_out=2 num_weights_input_layer=12 num_of_folds=10"
              "\n\nexample - letters:"
              "\npython3 hw1.py data_type=LETTERS data_path_letters='/home/nadav/workspace/NN-BackPropagation/Images-Dataset' log_file='/tmp/log.txt' "
              "epochs=1000 learning_rate=0.3 layers_num=3 neurons_hidden_1=20 neurons_hidden_2=6 neurons_out=8 num_weights_input_layer=256 num_of_folds=4")
        if len(argvs) >= 11 and exception_:
            raise exception_
        return 2
    if data_type == 'WINE':
        WHITE_CLASSIFICATION = 0
        RED_CLASSIFICATION = 1
        data_white = load_csv(data_path_white_wine)
        data_white = list(map(lambda x: list(x), set(list(map(lambda x: tuple(x), data_white)))))
        data_white = add_clasification_column(data_white, WHITE_CLASSIFICATION)
        data_red = load_csv(data_path_red_wine)
        data_red = list(map(lambda x: list(x), set(list(map(lambda x: tuple(x), data_red)))))
        data_red = add_clasification_column(data_red, RED_CLASSIFICATION)
        data = data_white + data_red
        data = normalize(data)
        data = transfor_expected_column_to_vector(data, neurons_out)
    else:
        neurons_out = len(LETTERS_LIST)
        data=load_and_categorize_letters(data_path_letters)

    shuffle(data)
    # partition the shuffled data to folds
    data = [data[i:i + round(len(data) / num_of_folds)] for i in range(num_of_folds)]

    W_MIN = -0.5
    W_MAX = 0.5
    num_of_ten_folds_iterations = num_of_folds
    min_soft_max_limit = 0.8
    transfer_func_C = 1

    if num_of_layers == 1:
        architecture = (
            [
                [ uniform(W_MIN, W_MAX) for i in range(num_weights_input_layer) ] for j in range(neurons_out)
            ],
        )
    elif num_of_layers == 2:
        architecture = (
             [
                 [ uniform(W_MIN,W_MAX) for i in range(num_weights_input_layer) ] for j in range(neurons_hidden_1)
             ],
             [
                 [ uniform(W_MIN,W_MAX) for i in range(neurons_hidden_1) ] for j in range(neurons_out)
             ]
         )
    elif num_of_layers ==3:
        architecture = (
             [
                 [ uniform(W_MIN,W_MAX) for i in range(num_weights_input_layer) ] for j in range(neurons_hidden_1)
             ],
             [
                 [ uniform(W_MIN,W_MAX) for i in range(neurons_hidden_1) ] for j in range(neurons_hidden_2)
             ],
             [
                 [ uniform(W_MIN,W_MAX) for i in range(neurons_hidden_2) ] for j in range(neurons_out)
             ]
         )
    else:
        raise Exception("illegal num of layers! (1-3)")
    execute_setup(transfer_func_C, num_of_ten_folds_iterations, alpha, epochs_num,architecture, data, log_file, min_soft_max_limit)

if __name__ == "__main__":
    main()