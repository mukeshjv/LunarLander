# Expected Sarsa-LunarLander-v2-Python

Python Implementation of [Expected Sarsa](http://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/vanseijenadprl09.pdf) for [LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/)

## Dependencies

- python 3.7
- numpy 1.18.1
- gym 0.17.2

## Configs

- step_size: 1e-3
- beta_m: 0.9
- beta_v: 0.999
- epsilon: 1e-8
- gamma: 0.99
- tau: 0.00

## Train your own model

    python model.py

## Results

Initial     |   After 200 episodes  |   After 500 episodes  
:-------------------------:|:-------------------------:|:-------------------------:
![start](https://github.com/mukeshjv/LunarLander/blob/master/blob/start_ll.gif)  | ![mid](https://github.com/mukeshjv/LunarLander/blob/master/blob/mid_ll.gif)  | ![end](https://github.com/mukeshjv/LunarLander/blob/master/blob/end_ll.gif)  

## Plot

![plot](blob/result.png)
