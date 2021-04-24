import csv
import numpy as np
import math

def import_and_parse_data(file):
    datafile = open(file, 'r')
    datareader = csv.reader(datafile, delimiter=' ')
    data = []
    for row in datareader:
        clean_row = [float(a) for a in row if a != '']
        if len(clean_row) == 1:
            data.append(clean_row[0]) 
        elif len(clean_row) > 1:
            data.append(clean_row)   
    return np.array(data)

def import_and_parse_numbers(file):
    rows = import_and_parse_data(file)

    # print(rows)

    data = []

    i = 0
    curr = []
    for i in range(0, rows.shape[0]):
        row = rows[i]
        if i > 0 and i % 7 == 0:
            # print(f"Curr: {curr}")
            data.append(curr)
            curr = []
            
        curr.extend(list(row))
        
        i += 1

    # print(f"Curr: {curr}")
    data.append(curr)

    # print(np.array(data))

    return np.array(data)

def calculate_abs_error(model, X, y):
    err = 0
    for i in range(0, X.shape[0]):
        #print(f'expected: {y[i]}, output: {perceptron.evaluate(X[i, :])}')
        err += abs(model.evaluate(X[i, :]) - y[i])
    return err

def calculate_mean_error(model, X, y):
    err = 0
    for i in range(0, X.shape[0]):
        #print(f'expected: {y[i]}, output: {perceptron.evaluate(X[i, :])}')
        err += abs(model.evaluate(X[i, :]) - y[i])
    err /= X.shape[0]
    return err

def calculate_standard_deviation(model, X, y):
    stdev = 0
    for i in range(0, X.shape[0]):
        #print(f'expected: {y[i]}, output: {perceptron.evaluate(X[i, :])}')
        stdev += pow(model.evaluate(X[i, :]) - y[i], 2)
    stdev /= X.shape[0]
    stdev = math.sqrt(stdev)
    return stdev

def calculate_classification_metrics(model, X, y):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for i in range(0, X.shape[0]):
        output = model.evaluate(X[i, :])
        output = 1 if output >= 0 else -1

        expected_output = y[i]
    
        # expected is positive
        if expected_output == 1 :
            # is true
            if output == expected_output:
                true_positives += 1
            # is false
            else:
                false_negatives += 1
        else:
            # is true
            if output == expected_output:
                true_negatives += 1
            # is false
            else:
                false_positives += 1

    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    precision = (true_positives) / (true_positives + false_positives) if true_positives + false_positives != 0 else math.inf
    recall = (true_positives) / (true_positives + false_negatives) if true_positives + false_negatives != 0 else math.inf
    f1_score = (2 * precision * recall) / ( precision + recall ) if precision + recall != 0 else math.inf

    res = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

    return res

def print_predictions_with_expected(model, X, y):
    for i in range(0, X.shape[0]):
        print(f'X = {X[i,:]} => y = {model.evaluate(X[i,:])} (should return {y[i]})')