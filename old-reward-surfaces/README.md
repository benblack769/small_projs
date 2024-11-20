# Original RL surfaces

Project obsolete: modern version of project [https://github.com/RyanNavillus/reward-surfaces](https://github.com/RyanNavillus/reward-surfaces)

Idea: plot reward surfaces around trained models in [RL baselines zoo](https://github.com/araffin/rl-baselines-zoo)

Repository contains copies of these models, so it is quite large.

### To recreate experiments

The experiment data in the `vis` folder can be recreated with:

```
python generate_data.py
```

Note that this may take a long, long time, i.e. weeks, even with a powerful CPU.
