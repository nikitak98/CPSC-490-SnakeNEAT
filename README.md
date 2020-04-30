# SnakeNEAT
Training Snake AI Agent with NEAT Genetic Algorithm for Neural Networks

### Requirements

This implementation uses numpy, pygame and neat-python packages. To visualize statistics matplotlib is also required. To install requirements run:

```bash
pip install -r requirements.txt
```

### Using game.py

This is the main file used to run the GA. It will create a folder with the current timestamp where a genome and its seed will be saved every time a new max fitness is set. After training is finished it will output statistics and a visualization of the winning network.

```bash
python3 game.py [--load-checkpoint file_name]
```

### Using replay.py

Replay takes as arguments a genome file as outputted by game.py and an optional seed file. If a seed file is provided, the snake will play exactly as when it was recorded. Otherwise, it will be randomly initialized.

```bash
python3 replay.py genome_file [seed_file]
```
