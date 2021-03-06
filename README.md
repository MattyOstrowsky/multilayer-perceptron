# multilayer-perceptron

<img align="right" width="25%" src="https://github.com/gunater/multilayer-perceptron/blob/main/assets/icons8-artificial-intelligence-100.png">

## Table of contents

* [General info](#general-info)
* [Installation](#installation)
* [How to run it](#how-to-run-it)
* [License](#license)

## General info
A multilayer perceptron (MLP) is a class of feedforward artificial neural network (ANN). 
A Multilayer Perceptron has input and output layers, and one or more hidden layers with many neurons stacked together.

## Installation

1. Git clone repository:
```bash
$ git clone https://github.com/gunater/multilayer-perceptron.git
```
2. Install the necessary python dependencies you can use `pipenv`:
```bash
$ pipenv install
$ piipenv shell
```
or you can install from requirements.txt with `pip`:
```bash
$ pip install -r requirements.txt
```
## How to run it
To run the script, go to the main directory:
```bash
$ cd project/
```
and then run example script with:
```bash
$ python3 example.py
```
The program will run and you will get output:
```bash
Epochs: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [00:02<00:00, 1725.40it/s]
====================================================================================================
Training complete!
Avg mse error: 0.20 after 5000 epochs
====================================================================================================
Network believes that is equal to [0.94 0.06 0.   0.04]
Network believes that is equal to [0.04 0.95 0.07 0.  ]
Network believes that is equal to [0.   0.03 0.93 0.04]
Network believes that is equal to [0.04 0.   0.06 0.95]
```
And you will also receive a file example_plot.png:
<p align="center">
  <img align="center" width="auto" src="https://github.com/gunater/multilayer-perceptron/blob/main/example_plot.png?raw=true">
</p>

## License
All code is licensed under an MIT license. This allows you to re-use the code freely, remixed in both commercial and non-commercial projects. The only requirement is to include the same license when distributing.


<a target="_blank" href="https://icons8.com/icon/61864/artificial-intelligence">Artificial Intelligence</a> icon by <a target="_blank" href="https://icons8.com">Icons8</a>
