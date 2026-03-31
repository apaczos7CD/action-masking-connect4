from envs.connect4_wrapper import OneAgentVsRandomGym
env = OneAgentVsRandomGym(seed=0)
obs, _ = env.reset()
assert env.action_masks().shape == (7,)

# wypełnij kolumnę 0 i sprawdź, że maska = False
while env.action_masks()[0]:
    obs, r, d, t, _ = env.step(0)
    
# wymuś nielegalny ruch => terminal −1
obs, r, d, t, info = env.step(0)
assert d and r == -1.0 and info.get("illegal_move", False)
print("Smoke OK")