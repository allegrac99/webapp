# Topology_Aware_Quantum_Circuit_Synthesis_with_RL
Generalizing Reinforcement Learning-based Quantum Circuit Synthesis across Multiple Topologies

## Description
This project introduces a topology-aware reinforcement learning-based quantum circuit synthesis which addresses different topologies simultaneously. Performance is evaluated in terms of the CX depth and depth values of the synthesized circuits and is compared to the state-of-the-art greedy techniques and a SAT solver. Experimental tests demonstrate that the proposed synthesizer outperforms state-of-the-art greedy techniques while being significantly faster than SAT solvers. The trained models have been tested both on routed and non-routed datasets.

## Installation
Follow these steps to set up the environment and install the necessary packages to run the provided code:
1. Clone the repository
2. Create and activate a virtual environment with Python installed inside (an anaconda environment with Python=3.11.5 was used for this project)
3. Install the required packages using requirements.txt

## Contents
The directories and files related to the test experiments for the 5 qubit-case are provided.
The decription of them is provided below:
- `qiskit_sat_synthesis_main`: directory containing a collection of SAT-based synthesis methods for various Qiskit objects
- `datasets`: directory containing the datasets used for testing the RL trained model. It contains both the routed and non-routed datasets
- `results`: directory containing the files where the evaluated performance obtained from the test is saved. In particular, there are two objects related to the depths and CX depths of the circuits synthesized by the RL trained model, the state-of-the-art greedy techniques and the SAT solver, and a text file related to the time averages of the algorithms
- `plots`: directory containing the plots related to the comparison of the performance og the RL trained model with that of the state-of-the-art
- `3q_model.zip`: RL trained model used for the test in the 3-qubit case 
- `5q_model.zip`: RL trained model used for the test  in the 5-qubit case
- `generate_tst_dataset.py`: file containing the code to generate a new test dataset and save it in the `datasets` directory
- `test.py`: file containing the code to test the trained model: it saves the results in the `results` directory and plots the performance in the `plots` directory
- `synthesis_env.py`: file containing the environment class
- `synthesize.py`: file containing the synthesis method presented
- `main.py`: file containing the code to use the synthesis method presented

## Usage
If you want to synthesize a Clifford operator and obtain the near-optimal synthesized quantum circuit you can run the `main.py` file with the following command:
```
python main.py
```
In the `results` and `plots` directories the values related to the tests that has already been conducted are present.

To test the synthesis method on the same data used during the test session, you can use the datasets present in the `datasets` directory.
