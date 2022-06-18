clear all; close all; clc

name = 'original';
T = readtable(['../data/datasets/csv/', name, '.csv']);
if name ~= "original" && name ~= "balanced_original"
    T = T(2:end, :);
end
X = table2array(T);
y = X(:, end);
X = X(:, 1:end-1);
X_norm = normalize(X);
save(['../data/datasets/mat/', name, '.mat'], 'X', 'y', 'X_norm')