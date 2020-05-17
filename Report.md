# Report

My Agent succeded in 120 episodes. You can see the plot below of the training

![Training](./result.png)

scipt to plot the performance:
``` bash
python view.py
```

In this project, I used **PPO (Proximal Policy Optimization)**
algorithm, and for PPO, I developed **GAE (Generalized Advantage Estimation)**.

For policy, I picked **Actor/Critic framework**, where actor outputs the
probability of actions or draw actions from the disribution based on the current state,
and critic outputs the state value function. To reduce complexity,
actor and critic share hidden layers.

After fixed steps update of multi agent environment, actor and critic are
mini-batch updated by PPO/GAE.

Layers for actor
```
Input: State(33)
Dense(512) LeakyReLU  
Dense(256) LeakyReLU
Dense(4) tanh
Output:
```

Layers for critic
```
Input: State(33)
Dense(512) LeakyReLU   
Dense(256) LeakyReLU  
Dense(1)  
Output:  
```


The output of actor model is used by mean parameter of normal distribution.
When agent plays in simulator, actor draws actions from distribution at random.
When agent updates model based on actions already drew from the model,
actor returns likelihood of actions of distribution.
(Variance parameters of normal distribution are also model parameters)

Because output of agents fluctuates based on normal distribution, agent can reach
optimal parameters in the balance between exploration and exploitation.

PPO is the policy-based method derived from **TRPO (Trust Region Policy Optimization)**.
PPO calculates gradient based on policy likelihood ratio between old one and updated new one.
PPO uses clipping for likelihood ratios to prevent gradient explosion.

GAE is the method to calculate "generalized advantage", which balance
between TD-error and discounted sum of rewards.
In this implementations, advantage is calculated based on GAE.

Hyper-parameters

- Discount factor gamma: 0.99
- GAE lambda: 0.96.
- Before model update, advantage is normalized to standard normal distribution.
- PPO clipping parameter (eps): 0.1
- Batch 128
- Actor/Critic are simultaneously updated using sum of loss of each model.
- Model is updated by mini-batch.
  - Optimizer: Adam
  - Learning rate: 1e-4
  - epoch: 10
  - mini batch size: 128
