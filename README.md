# ActorMimic
Code for paper http://arxiv.org/abs/1511.06342.
Allows a single deep network to play several Atari 2600 games at once. This network can then be used as a weight initialization in a new target game, which can speed up initial learning in that game. 

# Installation

1. Install torch (http://torch.ch/docs/getting-started.html#_).
2. Run ./install_dependencies.sh to install xitari and alewrap.
3. Find Atari 2600 ROMs and put them in roms/ directory.

# Results

Below is a set of instructions to re-create similar results as in the transfer section of the paper (section 5.2). 

NOTE: training the full pipeline from multitask to transfer results can easily take over 3 weeks even with a fast GPU.

## Multitask using Policy Regression Objective
This learns a multitask network on 13 source tasks using the policy regression objective. These 13 source tasks are the same as in the paper. The goal of training here is to later use this network for transfer, therefore training is stopped early at 4 million frames per game since that seemed best for later transfer results. 

    $ cd scripts
    $ ./run_amn_polreg_paper [1-based gpuid]

## Transfer using Policy Regression Objective
This learns a DQN on a new target task using the network trained above with the policy regression objective as a weight initialization. Below I have included scripts for training on the games where transfer had the largest effect.

Breakout:

    $ cd scripts
    $ ./run_dqn_polreg_4mil_breakout [1-based gpuid]
  
Star Gunner:

    $ cd scripts
    $ ./run_dqn_polreg_4mil_star_gunner [1-based gpuid]
  
Video Pinball:

    $ cd scripts
    $ ./run_dqn_polreg_4mil_video_pinball [1-based gpuid]
  
## Multitask using Policy+Feature Regression Objective
This learns a multitask network on 13 source tasks using the combined policy and feature regression objective. These 13 source tasks are the same as in the paper. The goal of training here is to later use this network for transfer, therefore training is stopped early at 4 million frames per game since that seemed best for later transfer results. 

    $ cd scripts
    $ ./run_amn_featreg_paper [1-based gpuid]

## Transfer using Policy+Feature Regression Objective
This learns a DQN on a new target task using the network trained above with the combined policy and feature regression objective as a weight initialization. Below I have included scripts for training on the games where transfer had the largest effect.

Breakout:

    $ cd scripts
    $ ./run_dqn_featreg_4mil_breakout [1-based gpuid]

Star Gunner:

    $ cd scripts
    $ ./run_dqn_featreg_4mil_star_gunner [1-based gpuid]
  
Video Pinball:

    $ cd scripts
    $ ./run_dqn_featreg_4mil_video_pinball [1-based gpuid]

# Acknowledgments

Code adapted from Deepmind's 2015 paper:

http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html

Github version here:

https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner
