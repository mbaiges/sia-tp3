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
        else:
            data.append(clean_row)   
    return np.array(data)

def import_and_parse_numbers(file):
    rows = import_and_parse_data(file)

    print(rows)

    data = []

    i = 0
    curr = []
    for i in range(0, rows.shape[0]):
        row = rows[i]
        if i > 0 and i % 7 == 0:
            print(f"Curr: {curr}")
            data.append(curr)
            curr = []
            
        curr.extend(list(row))
        
        i += 1

    print(f"Curr: {curr}")
    data.append(curr)

    print(np.array(data))

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

def print_predictions_with_expected(model, X, y):
    for i in range(0, X.shape[0]):
        print(f'X = {X[i,:]} => y = {model.evaluate(X[i,:])} (should return {y[i]})')