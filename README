----------------------------------------
---------------|0 FILES|----------------
----------------------------------------

===========================
0.0 [Training Scripts are:]
===========================

- DNN.py
- GBM.py
- KRR.py
- LASSO.py
- SVR.py

all 5 scripts have their hyperparameter & mode selection
in the indicated part at the start after the libraries
imports, inside the code.

====================================
0.1 [Training scripts' outputs are:]
====================================
-A CSV file that is appended, every run with
information about the completed run. This file
is generated at the root folder.

-A file for the boxplot data, which is stored under
'boxplot_data' folder which contains essential data
for boxplot generation later.

-A file that stores the trained model's information
of weights and architecture after the training is
complete, which is stored under the 'pretrained_models'
folder.

-Two prediction files, the big one is for the evaluation
set of the training set provided/used, the small one is 
for the Hidden Trajectories.


(Every single run is labeled with a different random ID,
between 0 and 10.000 for mapping convenience between models,
boxplot data, statistics and so on and so forth)

=====================
0.2 [makeBoxPlots.py]
=====================

Inside this script, you can change the names of the models
that want you to produce boxplots for and the output is 
boxplot pdf-images that provide visual elaboration on the
statistics of the chosen models' performance (Testing MSE Loss)
along with an extra txt file which contains the boxplot statistical
information with numbers.

===================
0.3 [data (folder)]
===================

Folder 'data' contains the configured datasets that our models use
as their input.

================================
0.4 [train & test id .txt files]
================================

Are used in training scripts in order to discriminate trajectories
between train and test.

----------------------------------------
- |1 Requirements & Technical Details| -
----------------------------------------


======================
1.0 [Module | Version]
======================

"We ensure compatibility with the following versions of modules"

Python | 3.8.10 
torch | 1.11.0
pandas | 1.3.3
matplotlib | 3.4.3
sklearn | 0.0

================
1.1 [How to run]
================

To run any of [LASSO, DNN, SVR, GBM, KRR].py learning scripts:

	1. Configure the hyperparameters as you like as per instructed at section [0.0]
	2. Open a cmd to HV_CODE directory and run "python3 [chosen_method].py"
	
To run makeBoxPlots.py script which makes the boxplots:
	1. You need to have atleast 1 pretrained_model of each method (because it produces 		comparative results)
	2. Configure the models you choose as per instructed at section [0.2]
	
1.2 [Technical Requirements]
	1. Scripts do not require gpu to run.
	2. All algorithms have low requirements except some extreme tunings at KRR and SVR
	where we found that a minimum of 16gb ram is needed.


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
For troubleshooting connect with th.tranos@uoi.gr
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

