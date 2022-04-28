import matplotlib.pyplot as plt
import json
import re
from numpy.core.fromnumeric import mean, ptp
from src.pre_data import load_raw_data
import numpy as np
import seaborn as sns
import copy
import pandas as pd

filename_simple = 'prime100epoch.log'
filename_wo_simple = 'prime100epoch_wo_simple.log'
filename_wo_sub = 'prime100epoch_w_sub.log'
filename_wo_simplify = 'prime100epoch_w_simplify.log'
filename_wo_simplify_sub = 'prime100epoch_w_simplify_sub.log'
filename_wo_simplify_math = 'prime100epoch_w_simplify_math.log'
filename_wo_simplify_math_augment = "prime100epoch_w_simplify_math_augment.log"
filename_wo_math_final_1e5 = 'prime100epoch_w_simplify_math_final_1e5.log'
filename_wo_math_final_1e12 = 'prime100epoch_w_simplify_math_final_1e12.log'
filename_w_loss_v2 = 'prime100epoch_math23k_loss_v2.log'
filename_w_loss_v1 = 'prime100epoch_math23k_loss_v1.log'
filename_cm17k_prime = 'prime100epoch_cm17.log'
filename_cm17k_new = 'prime100epoch_cm17k_variable.log'
filename_math23k_sota = 'prime100epoch_math23k_sota.log'

buffer_simplify = 'buffer_exp_simplify.json'
buffer_sub = 'buffer_exp_sub.json'


def read_log_file_exfeature(filename, feature):
    acc = []
    f = open(filename)
    line = f.readline()
    while line:
        number_str = ''
        if feature in line:
            for c in line:
                if str.isdigit(c) or c == '.':
                    number_str += c
            acc.append(float(number_str[1:]))
        line = f.readline()
    return acc


buffer_file = 'buffer_exp_simplify.json'


def read_buffer(filename):
    num = 0
    empty = 0
    no_eq = []
    with open(filename, 'r') as load_f:
        load_dict = json.load(load_f)
        for key, value in load_dict.items():
            num += len(value)
            if len(value) == 0:
                empty += 1
                no_eq.append(key)
    return empty, no_eq


def statistic_number_operator(filename):
    '''
    统计label中数字和操作符最多出现了多少次
    统计自顶向下分解做溯因推理的复杂度最多是多少。
    '''
    operation = {'*', '-', '+', '/', '^'}
    count = {'+': [], '-': [], '*': [], '/': [], '^': []}
    dataset = load_raw_data(filename)
    add_count = []
    for data in dataset:
        count_local = {'+': 0, '-': 0, '*': 0, '/': 0, '^': 0}
        eq = data['equation'][2:]
        for e in eq:
            if e in operation:
                count_local[e] += 1
        if count_local['+'] > count['+']:
            add_count.append(data)
        for c in count_local.keys():
            count[c] = max(count[c], count_local[c])
    print(add_count[-5:-3])


def plot_accuracy(list_acc, minLength):
    sns.set(style="darkgrid")
    sns.set_context("notebook")
    sns.color_palette("magma", as_cmap=True)
    sns.set(font_scale=0.8)
    x = range(minLength)

    dict_acc = {}
    for key, value in list_acc.items():
        dict_acc[key] = value[:minLength]
    df = pd.DataFrame(dict_acc, index=x)

    ax = sns.relplot(data=df, kind='line', aspect=1, linewidth=2, palette='hls')
    ax.set_xlabels('epoch')
    ax.set_ylabels('accuracy')
    plt.yticks(np.concatenate([np.arange(0, 0.5, 0.1), np.arange(0.5, 0.7, 0.03)]), rotation=45)
    plt.savefig('acc_pic/1e.png')


if __name__ == '__main__':
    # statistic_number_operator('data/Math_23K.json')
    #===============================================================
    # num_empty, no_eq = read_buffer(buffer_file)
    # print(num_empty)

    # =================================================================
    accuracy = 'test_answer_acc1'
    acc_prime = read_log_file_exfeature(filename_simple, accuracy)
    primes = []
    for acc in acc_prime:
        primes.append(copy.deepcopy(acc))
        primes.append(copy.deepcopy(acc))
        primes.append(copy.deepcopy(acc))
        primes.append(copy.deepcopy(acc))
        primes.append(copy.deepcopy(acc))

    acc_sub = read_log_file_exfeature(filename_wo_sub, accuracy)
    acc_simplify = read_log_file_exfeature(filename_wo_simplify, accuracy)
    acc_simplify_sub = read_log_file_exfeature(filename_wo_simplify_sub, accuracy)
    acc_simplify_math = read_log_file_exfeature(filename_wo_simplify_math, accuracy)
    acc_simplify_math_augment = read_log_file_exfeature(filename_wo_simplify_math_augment, accuracy)
    acc_final1e5 = read_log_file_exfeature(filename_wo_math_final_1e5, accuracy)
    acc_final1e12 = read_log_file_exfeature(filename_wo_math_final_1e12, accuracy)
    acc_cm17k = read_log_file_exfeature(filename_cm17k_prime, accuracy)
    acc_cm17k_new = read_log_file_exfeature(filename_cm17k_new, accuracy)
    acc_math23k_loss_v2 = read_log_file_exfeature(filename_w_loss_v2, accuracy)
    acc_math23k_loss_v1 = read_log_file_exfeature(filename_w_loss_v1, accuracy)
    acc_math23k_sota = read_log_file_exfeature(filename_math23k_sota, accuracy)
    # =================================================================
    accuracy_set = {"acc_math23k_sota": acc_math23k_sota}
    minLength = min(len(acc) for key, acc in accuracy_set.items())
    # =================================================================
    plot_accuracy(accuracy_set, minLength)
