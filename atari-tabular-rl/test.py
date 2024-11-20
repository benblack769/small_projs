import multi_agent_ale_py
import example
import os
import time
import multiprocessing

pathstart = os.path.dirname(multi_agent_ale_py.__file__)
game = "montezuma_revenge"
final_path = os.path.join(pathstart, "ROM", game, game + ".bin")
assert os.path.exists(final_path)
n_cpu = multiprocessing.cpu_count()
#n_cpu = 1
print(example.run_main(final_path, 30000000, n_cpu, 1000000000))
