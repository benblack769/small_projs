### Writeup about results

[https://weepingwillowben.github.io/intelligent-code/random_walks/](https://weepingwillowben.github.io/intelligent-code/random_walks/)

### Usage

compile with qmake, perhaps via qt creator

command args are

     <executable> <maze_filename> <liniar_walk[true,false]> <random_dir_weight> <pull_to_end_weight> <avoid_self_weight> <lin_walk_len>

if liniar_walk is not "true" then the other arguments are ignored, it behaves as a regular random walk

example

    ./path maze1.png true 10 0.3 -3 4
