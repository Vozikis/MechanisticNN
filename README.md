
# Physical Phenomena Simulation and Analysis

This repository provides scripts for simulating and analyzing various physical phenomena, including free fall, pendulum motion, and double pendulum dynamics. The repository also includes pre-generated trajectories and experiment videos.

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

## Output Files and Visualizations

- **Trajectories**:
  - You can find pre-generated trajectory plots in the `Trajectories` folder located in the base directory.
  - The folder contains:
    - Separate trajectory plots generated every 100 epochs.
    - A merged plot showing all trajectories in one graph for better visualization.

- **Videos**:
  - The `Videos` folder in the base directory contains experiment videos corresponding to the different physical phenomena.

## Requirements

Make sure you have Python installed along with the required dependencies. You can install the necessary packages using:

```bash
pip install -r requirements.txt
```

## Contributing

Feel free to open issues or submit pull requests if you find any bugs or have suggestions for improvements.

## License

This project is licensed under the [MIT License](LICENSE).

---

Happy experimenting! 🎉
