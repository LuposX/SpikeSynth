# SpikeSynth: Energy-Efficient Adaptive Analog Printed Spiking Neural Networks


This is the GitHub repository for the ICCAD 2025 paper with the same name.


The data can be found here: https://1drv.ms/f/c/a31285484594c370/ErPw8IcCU5tCl2CpgQnXkj8BY41yb5YgZAaSnQjNQNRNEw?e=On30Sp


How to use the code:

 1. Store your SPIKE data and use the 1_read.ipynb file to create the dataset
 2. Train the surrogate model with 2_modeling.py
 3. Use the exp_pSNN_lP.py to train the P-LSNN


 Example usage of 2_modeling.py with seed 0 and transformer model gpt-nano:

~~~
$ python 2_modeling.py 0 0 
~~~

Example usage of PrintedSpkingNN_lP.py:

~~~
$ python exp_pSNN_lP.py --DATASET 00 --SEED 0 --projectname pLSNN
~~~

This creates a folder pLSNN with the log files and the model.

If you want to include variation switch exp_pSNN_lP.py to exp_pSNN_var_lP.py

If you want to reproduce the baseline approach [1] switch exp_pSNN_lP.py to exp_pSNN.py. Here you need to train a seperate surrogate model within the surrogate_baseline directory.


[1] Analog Printed Spiking Neuromorphic Circuit
Pal, P.; Zhao, H.; Shatta, M,; Hefenbrock, M.; Mamaghani, S. B.; Nassif, S.; Beigl, M.; Tahoori, M. B.
2024 Design, Automation & Test in Europe Conference & Exhibition (DATE), IEEE, 2024
