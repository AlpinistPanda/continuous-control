[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"


# Continuous Control
This project is part of Udacity Deep Reinforcement Learning Nanodegree. It is
the second project to fullfil the requirements.


In this project I developed policy-based agent which controls robot arm
to touch the swinging balls as much time as it can in the simulator which is created in Unity Game Engine.
This project uses target simulator: [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

In this project I developed 20 agents, each with its own environment. (2nd option)
Udacity requires the students to satisfy next criteria:
- Agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
  - After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores.
  - This yields an **average score** for each episode (where the average is over all 20 agents).
- The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.

See [Report.md](./Report.md)

# Getting Started
In this repository, there is best model already trained for enough steps,
but you can train in your environment following next procedures if you want.


 Download the environment from one of the links below.  You need only select the environment that matches your operating system:

  - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
  - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
  - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
  - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

 Place the file in the DRLND GitHub repository, in the `./bin/` folder, and unzip (or decompress) the file.

 Run the python scripts.
``` bash
# Train the agent
# Trrained model is saved to bestmodel.pth when achieved criteria
python train.py
# View the best model (bestmodel.pth) in the simulator
python view.py
# Plot the chart of best model scores (bestmodel_score.png)
python eval.py
```
