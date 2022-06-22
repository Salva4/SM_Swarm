# MLP

- The datasets and the indices for the train-validation-test partitions are stored in .csv and .mat files in the directory `data`.

- The folder `utils` contains useful scripts that have been used to create the different datasets from the original and the random partitions, split the partitions into several files (for later ease), convert both datasets and partitions to .mat files (for eSPA) and box-plot the results of all the experiments (AUC and running time).

- Finally, the folder `src` contains the scripts for the experiments, hyperparameter tuning, and a history of the run experiments. +more specific README's.

There is some inconsistency between the names of the datasets used in `utils` and in some of the scripts with others, due to a posterior renaming. However, the dataset names should be self-explainatory: \\
    - Swarm_Behaviour = original
    - swarm_lda = lda
    - pca3 (10 components) = pca
    - auto_enc = autoenc
    - balanced_data = balanced_original