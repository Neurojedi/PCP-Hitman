After AlphaZero's impressive success in the arguably hardest combinatorial two player games, like Go and Chess, we want to find out whether it can help with yet another great challenge...connect four! :) 


This repository contains the code to train an alphazero like model in connect four, as well as the option to play against different agents (alphazero, random, minimax, mcts) or to let them play against each other. 

### Installation

To install this repository you can first get the necessary packages running 

    pip install -r requirements.txt 

  
or if you're using conda

    conda env create -f environment.yml


### Contents

The repository is structured as follows:

    .
    ├── agents/                        # Different agents
    │   ├── AlphaZeroAgent.py
    │   ├── MCTS.py
    │   ├── minimax.py        
    │   └── random.py
    ├── neuralnet/                     # Neural network and training utilities
    │   ├── ResNet.py                  # Network architecture
    │   ├── connect4_model_graph.png   # Architecture diagram
    │   └── utils.py                   # Helper functions   
    ├── tests/
    │   ├── test_game_utils.py         # Tests for board logic and mechanics
    │   ├── test_mcts.py               # Tests for agent scoring and decision-making
    │   └── test_utils.py              # Tests for network functions
    ├── weights/                       # Checkpoints of trained AlphaZero agents
    │   └── ... 
    ├── Model.ipynb                    # Summary of model architecture
    ├── Results.ipynb                  # Plots and performance results
    ├── game_utils.py                  # Core game logic 
    ├── main.py                  # Play against the AlphaZero agent yourself
    ├── play_AlphaMinimax.py           # AlphaZero vs. Minimax
    ├── play_AlphaRandom.py            # AlphaZero vs. random agent
    ├── training.py                    # Train a new AlphaZero agent 
    └── README.md                      # Project documentation (you are here)


                         
