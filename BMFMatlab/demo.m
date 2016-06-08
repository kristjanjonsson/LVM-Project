train_vec = csvread('../data/ml_100k_train.csv');
probe_vec = csvread('../data/ml_100k_test.csv');

restart=1;
fprintf(1,'Running Probabilistic Matrix Factorization (PMF) \n');
pmf

restart=1;
fprintf(1,'\nRunning Bayesian PMF\n');
bayespmf

