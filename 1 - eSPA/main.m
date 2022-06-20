%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Application of eSPA on all the reduced datasets
%
% Code taken from "Analysis of the Worms, Illia Horenko 2022"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear all
close all

addpath('ProgramFiles/')

%% INPUT
DS_NAME = input(['- Write the name of the dataset:\n', ...
	'original / lda / pca / autoenc / pca_corr1 / pca_corr2 / pca_corr3\n'], 's');

if DS_NAME(1:3) == "pca"
    SMALL = input(['- Small dataset? (only 10000 first samples)\n', ...
	'no / yes\n'], 's');
else
    SMALL = 'no';
end

if SMALL == "no" && (DS_NAME(1:end-1) ~= "pca_corr")
    BALANCED = input('Balanced dataset? no / yes\n', 's');
else
    BALANCED = "no";
end

W_espa = input('eSPA hyperparameter eps_W = ');
CL_espa = input('eSPA hyperparameter eps_CL = ');

% End of input


%% Execution
results_AUC = zeros(10, 2);
results_time = zeros(10, 2);
for partition = 1:1
    if DS_NAME ~= "lda" && DS_NAME ~= "balanced_lda"
        ds_mat = [DS_NAME, '.mat'];
    elseif DS_NAME == "lda" && BALANCED == "no"
        ds_mat = ['lda/lda_', num2str(partition), '.mat'];
    elseif DS_NAME == "lda"
        ds_mat = ['lda/balanced_lda_', num2str(partition), '.mat'];
    end

    path_indices = '/Users/marcsalvado/Desktop/SM_Proj_CODE/data/partitions/mat/indices';

    if SMALL == "yes"
        path_indices = [path_indices, 'S'];
    elseif BALANCED == "yes"
        path_indices = [path_indices, 'B'];
        ds_mat = ['balanced_', ds_mat];
    end

    ds_path = '/Users/marcsalvado/Desktop/SM_Proj_CODE/data/datasets/mat/';
    ds_path = [ds_path, ds_mat];
    load(ds_path)

    if SMALL == "yes"
        X = X(1:10000, :); y = y(1:10000);
    end

    path_indices = [path_indices, num2str(partition), '.mat'];
    load(path_indices)

    % pi is the one-hot version of y
    pi = zeros(2, size(y, 1));
    for i = 1:size(y,1)
        pi(y(i)+1, i)=1;
    end



    %% Rescale features to interval [0,1]
    X = X_norm;     % normalization has already been done as preprocessing
    X = X';     % turn TxN into NxT, N dimensionality, T sample size

    warning('off')
    rand('seed',1);
    randn('seed',1);

    %% Set this flag to 1 if you have the licence for a "Parallel Computing" toolbox of MATLAB
    flag_parallel=1;

    %% The eSPA parameter exploring region has not been modified from the original code 
    % Number of eSPA patterns/boxes/clusters
    K=4;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % %%% Esborrar 1/2
    % flag_parallel=0;
    % %%%

    T = size(X,2);        % sample size

    %% Select flag_AUC=1 to use Area Under Curve as a performance metrics
    %% selecting flag_AUC=0 implies an Accuracy as a performance metrics
    flag_AUC=1;

    %% eSPA parameters region
    reg_param(:,1) = [W_espa; CL_espa];

    paroptions = statset('UseParallel',true);

    % %%% Esborrar 2/2
    % display('Parallel false')
    % paroptions = statset('UseParallel',false);
    % %%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Extract training set from the dataset (valid+test will be done in
    % ...SPACL_Replica_ts.m)
    X_train = X(:, train_indices + 1);
    pi_train = pi(:, train_indices + 1);
    X_valid = X(:, val_indices + 1);
    pi_valid = pi(:, val_indices + 1);
    X_valid_ts = X(:, test_indices + 1);
    pi_valid_ts = pi(:, test_indices + 1);

    seed=rng;
    out_eSPA = SPACL_kmeans_dim_entropy_analytic_v3_ts(X_train,pi_train,...
        K,40,[],reg_param,X_valid,pi_valid,X_valid_ts,pi_valid_ts,1,flag_AUC,flag_parallel);

    format long
    eSPA_AUC = -out_eSPA.L_pred_valid_Markov
    eSPA_time = out_eSPA.time_Markov
    results(partition, 1) = eSPA_AUC;
    results(partition, 2) = eSPA_time;
end

results