
# Physical Phenomena Simulation and Analysis

This repository provides the implementation of Mechanistic Neural Network for simulating and analyzing various physical phenomena, including free fall, pendulum motion, and double pendulum dynamics. Paper can be found here: https://arxiv.org/pdf/2402.13077v1

## How to Run the Scripts

To run the scripts, follow these steps:

1. Navigate to the base directory of the repository.
2. Execute one of the following Python scripts based on the physical phenomenon you're interested in:

   ```bash
   python fit/free_fall_bounce.py
   ```
   or
   ```bash
   python fit/free_fall.py
   ```
   or
   ```bash
   python fit/holonomic_pendulum.py
   ```
   or
   ```bash
   python fit/non_holonomic_pendulum.py
   ```
   or
   ```bash
   python fit/double_pendulum.py
   ```

Running those scripts will replicate the experiments from the beginning and it will save the results under the folder `trajectory_plots`

## Output Files and Visualizations

- **Trajectories**:
  - You can find pre-generated trajectory plots in the `Trajectories` folder located in the base directory.
  - The folder contains:
    - Separate trajectory plots generated every 100 epochs.
    - A merged plot showing all trajectories in one graph for better visualization.

- **Videos**:
  - The `Videos` needs to be included and are not part of the repository

