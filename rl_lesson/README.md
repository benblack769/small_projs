
I am not that talented myself at MLops (or ML theory, or anything else in ML). But I know one person who has great MLops: Costa Huang, developer of [CleanRL](https://docs.cleanrl.dev/), the only RL library in existence that is both friendly to new users and academic reaserch. 




### Installation

Install CleanRL:

```
pip install poetry
pip install wandb
git clone https://github.com/vwxyzjn/cleanrl.git
cd cleanrl && poetry install && cd ..
```

Next, if you are interested in experiment tracking (which you should be), create an account on Wandb https://wandb.ai/

And then log into wandb:

```
wandb login
```


### Running


Now, we can get started with the RL:

```
python cleanrl/cleanrl/ppo.py \
    --seed 1 \
    --env-id CartPole-v0 \
    --total-timesteps 50000 
```

To see your logs with tensorboard, you can run

```
tensorboard --logdir runs
```


To set up video capture and log everything with wandb, do:

```
xvfb-run -s "-screen 0 1024x768x24" python cleanrl/cleanrl/ppo.py \
    --seed 1 \
    --env-id CartPole-v0 \
    --total-timesteps 50000 \
    --wandb-project-name rl_lesson \
    --track \
     --capture-video
```

### Something harder

Lunar lander:

Install environment 

```
pip install gym[box2d]
```

```
xvfb-run -s "-screen 0 1024x768x24" python cleanrl/cleanrl/ppo.py \
    --seed 1 \
    --env-id LunarLander-v2 \
    --num-envs 16 \
    --track \
    --total-timesteps 1000000  \
    --capture-video
```

Note that it doesn't perform very well, even after 1,000,000 steps.

