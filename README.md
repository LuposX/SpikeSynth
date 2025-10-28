# 🧠 Flex-Spike: Analysis of flexible analog spiking neural network architecture for neuromorphic learning

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Built with Nix](https://img.shields.io/badge/built%20with-Nix-5277C3.svg)](https://nixos.org/)

> [!NOTE]
> This project is currently in development as part of a bachelor’s thesis.

## 🚀 Installation

Clone the repository:

```sh
git clone https://github.com/LuposX/SpikeSynth.git
cd SpikeSynth
````

### Using Nix

You can use Nix to install the required Python packages and enter a development shell:

```sh
nix develop
```

### Using Pip

Alternatively, you can install the dependencies globally with Pip:

```sh
pip install -r requirements.txt
```


## 🧩 Usage

### Surrogate

To create the surrogate model, navigate to the `surrogate` folder.
Place your **SPIKE data** for the circuit you want to simulate into the `data` directory.

1. Use the notebook `1_create_surrogate_dataset.ipynb` to generate the dataset for training the surrogate model.
   The resulting dataset will be saved as `data/dataset.ds`.
2. Use `2_tain_gpt_surrogate.py` to train the baseline GPT surrogate model, or `2_train_rsnn_surrogate.py` to train the RSNN surrogate model.
   Logging requires a **Weights & Biases (wandb)** account.
3. To perform hyperparameter optimization, run:

   ```sh
   python 3_hyperparameter_search_rsnn.py
   ```

#### Running Hyperparameter Search on a Cluster

If you want to run the hyperparameter search on a cluster system that uses **SLURM**, you can use the file `slurm_sweep.sh` in the `surrogate` folder.

1. Edit the script to adjust the task parameters to your requirements.

2. Submit the job:

   ```sh
   sbatch slurm_sweep.sh
   ```

3. To see an estimate of when your job will start:

   ```sh
   squeue --start -j <job-id>
   ```

4. To view your running jobs:

   ```sh
   squeue -u $USER
   ```

All SLURM logs will be stored in the `logs_slurm` directory.


### pLSNN

To create the full pLSNN model, use the notebook:

```sh
train_pRSNN.ipynb
```

* To include **variation**, switch `exp_pSNN_lP.py` to `exp_pSNN_var_lP.py`.
* To reproduce the **baseline approach** from [1], switch `exp_pSNN_lP.py` to `exp_pSNN.py`.
  In this case, you’ll need to train a separate surrogate model within the `surrogate_baseline` directory.

## 📂 Data

Experimental data for this project can be found [here](https://1drv.ms/f/c/a31285484594c370/ErPw8IcCU5tCl2CpgQnXkj8BY41yb5YgZAaSnQjNQNRNEw?e=On30Sp).


## 🤝 Credits

This repository is based on the the ICCAD 2025 paper and its github Repo **SpikeSynth: Energy-Efficient Adaptive Analog Printed Spiking Neural Networks**.

**Reference**
[1] Pal, P.; Zhao, H.; Shatta, M.; Hefenbrock, M.; Mamaghani, S. B.; Nassif, S.; Beigl, M.; Tahoori, M. B.
“Analog Printed Spiking Neuromorphic Circuit,”
*2024 Design, Automation & Test in Europe Conference & Exhibition (DATE)*, IEEE, 2024.

## 🪪 License

This project is licensed under the [MIT License](LICENSE).