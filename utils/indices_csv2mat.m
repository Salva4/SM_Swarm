clear all; close all; clc

for partition = 1:10
    T_train = readtable(['../data/partitions/csv/iTrainB', ...
        num2str(partition), '.csv']);
    T_val = readtable(['../data/partitions/csv/iValB', ...
        num2str(partition), '.csv']);
    T_test = readtable(['../data/partitions/csv/iTestB', ...
        num2str(partition), '.csv']);
    T_train = T_train(2:end, 1);
    train_indices = table2array(T_train);
    T_val = T_val(2:end, 1);
    val_indices = table2array(T_val);
    T_test = T_test(2:end, 1);
    test_indices = table2array(T_test);
    save(['../data/partitions/mat/indicesB', num2str(partition), '.mat'], ...
        'train_indices', 'val_indices', 'test_indices');
    clear all;
end