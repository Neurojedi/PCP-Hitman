
After AlphaZero's impressive success in the arguably hardest combinatorial two player games,like Go and Chess, we want to find out whether it can help with yet another great challenge...connect four! :) 

This repository contains the code to train a alphazero like model in connect four, as well as the option to play against different agents (alphazero, random, minimax, mcts) or to let them play against each other. 

To install this repository you can first get the necessary packages running 

  pip install -r requirements.txt 
or if you're using conda
  conda env create -f environment.yml

The repository is structured as follows:
.
├── agents/                        # contains the different agents
│   └── AlphaZeroAgent.py
│   └── MCTS.py
│   └── minimax.py        
│   └── random.py
├── neuralnet/                      # the neural network and network functions for training the AlphaZero agent
│   └── ResNet.py                   # Network Architecture
│   └── connect4_model_graph.png    # scheme of architecture
│   └── utils.py                    # network helper functions   
├── tests/
│   └── test_game_utils.py          # Tests for board logic, game mechanics
│   └── test_mcts.py                # Tests for agent scoring and decision-making
│   └── test_utils.py               # Tests network functions
├── weights/                        # Checkpoints of trained AlphaZero agent
│   └── ... 
├── Model.ipynb                     # summary of model architecture
├── Results.ipynb                   # notebook used to plot performance results of the trained AlphaZero agent
├── game_utils.py                   # Core game logic 
├──play_Alpha.py                    # run to play against the AlphaZero agent yourself
├──play_AlphaMinimax.py             # run to let the AlphaZero agent play against the Minimax agent
├──play_AlphaRandom.py              # run to let the AlphaZero agent play against the random agent
├── training.py                     # run to train a new AlphaZero agent 
└── README.md                       # Project documentation (you are here)

                         
