# SpikeSynth: Energy-Efficient Adaptive Analog Printed Spiking Neural Networks

> [!NOTE]
> This project is currently in development as part of a bachelor’s thesis.


## Installation

Clone the repository:

```sh
git clone https://github.com/LuposX/SpikeSynth.git
```

### Using Nix

You can use Nix to install the required Python packages and enter a development shell:

```sh
nix develop
```

### Using pip

Alternatively, you can install the dependencies globally with pip:

```sh
pip install -r requirements.txt
```


## Usage

### Surrogate

To create the surrogate model, navigate to the `surrogate` folder.
Place your **SPIKE data** for the circuit you want to simulate into the `data` directory.

* Use the notebook `1_create_surrogate_dataset.ipynb` to generate the dataset for training the surrogate model.
  The resulting dataset will be saved as `data/dataset.ds`.
* Use `2_create_gpt_surrogate.ipynb` to train the baseline GPT surrogate model, or `2_create_rsnn_surrogate.ipynb` to train the RSNN surrogate model.
  Logging requires a **Weights & Biases (wandb)** account.
* To perform hyperparameter optimization, run `3_hyperparameter_search_rsnn.py`.



### pLSNN

To create the full pLSNN model, use the notebook:

```sh
train_pRSNN.ipynb
```

* To include **variation**, switch `exp_pSNN_lP.py` to `exp_pSNN_var_lP.py`.
* To reproduce the **baseline approach** from [1], switch `exp_pSNN_lP.py` to `exp_pSNN.py`.
  In this case, you’ll need to train a separate surrogate model within the `surrogate_baseline` directory.


## Credits

This repository accompanies the ICCAD 2025 paper
**SpikeSynth: Energy-Efficient Adaptive Analog Printed Spiking Neural Networks**.

The data can be found [here](https://1drv.ms/f/c/a31285484594c370/ErPw8IcCU5tCl2CpgQnXkj8BY41yb5YgZAaSnQjNQNRNEw?e=On30Sp).

**Reference**
[1] Pal, P.; Zhao, H.; Shatta, M.; Hefenbrock, M.; Mamaghani, S. B.; Nassif, S.; Beigl, M.; Tahoori, M. B.
“Analog Printed Spiking Neuromorphic Circuit,”
*2024 Design, Automation & Test in Europe Conference & Exhibition (DATE)*, IEEE, 2024.