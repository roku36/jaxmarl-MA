"""
Introduction running and visualising the MPE environments using random actions.
"""

!pip list

import pkg_resources
import sys
print(f"Python: {sys.executable}\n")
installed_packages = [(d.project_name, d.version) for d in pkg_resources.working_set]
for name, version in sorted(installed_packages):
    print(f"{name}=={version}")

import matplotlib
import jax
from jaxmarl import make
from jaxmarl.environments.mpe import MPEVisualizer

# Parameters + random keys
max_steps = 25
key = jax.random.PRNGKey(0)
key, key_r, key_a = jax.random.split(key, 3)

# Instantiate environment
env = make('MPE_simple_reference_v3')
obs, state = env.reset(key_r)
print('list of agents in environment', env.agents)

# Sample random actions
key_a = jax.random.split(key_a, env.num_agents)
actions = {}
for i, agent in enumerate(env.agents):
    if isinstance(agent, str):
        action = env.action_space(agent).sample(key_a[i])
    elif isinstance(agent, int):
        action = env.action_space(agent_id=agent).sample(key_a[i])
    else:
        raise TypeError(f"Unexpected agent type: {type(agent)}")
    
    actions[agent] = action
# actions = {agent: env.action_space(str(agent)).sample(key_a[i]) for i, agent in enumerate(env.agents)}
print('example action dict', actions)

# Collect trajectory
state_seq = []
for _ in range(max_steps):
    state_seq.append(state)
    # Iterate random keys and sample actions
    key, key_s, key_a = jax.random.split(key, 3)
    key_a = jax.random.split(key_a, env.num_agents)
    actions = {agent: env.action_space(agent).sample(key_a[i]) for i, agent in enumerate(env.agents)}

    # Step environment
    obs, state, rewards, dones, infos = env.step(key_s, state, actions)

# Visualise
viz = MPEVisualizer(env, state_seq)
viz.animate(view=False, save_fname="gifs/mpe.gif")















