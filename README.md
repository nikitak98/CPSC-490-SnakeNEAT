# SnakeNEAT
Training Snake AI Agent with NEAT Genetic Algorithm for Neural Networks

### Using game.py

This is the main file used to run the GA. It will create a folder with the current timestamp where a genome and its seed will be saved every time a new max fitness is set.

python3 game.py [--load-checkpoint file_name]

### Using replay.py

Replay takes as arguments a genome file as outputted by game.py and an optional seed file. If a seed file is provided, the snake will play exactly as when it was recorded. Otherwise, it will be randomly initialised.

python3 replay.py genome_file [seed_file]
