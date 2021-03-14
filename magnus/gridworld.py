import gym
import numpy as np
import ray
from really import SampleManager
from gridworlds import GridWorld
#%%time
import random
import os
#from really import SampleManager  # important !!
from really.utils import (
    dict_to_dict_of_datasets,
)

#import pickel for saving

"""
Your task is to solve the provided Gridword with tabular Q learning!
In the world there is one place where the agent cannot go, the block.
There is one terminal state where the agent receives a reward.
For each other state the agent gets a reward of 0.
The environment behaves like a gym environment.
Have fun!!!!

"""


class TabularQ(object):

    def __init__(self, h, w, action_space):

        self.action_space = action_space
        #print(h,w)
        self.q_table = np.zeros((h, w, 4))


        #self.q_table.fill(0)


        #print(self.q_table)
        #print(self.q_table[2, 2])
        #q_table[0][3]

        ## # TODO:
        pass

    def __call__(self, state):
        ## # TODO:

        output = {}
        #print(state)
        #print(state[0][0], state[0][1])
        #print(self.q_table[2, 2])
        a = int(state[0][0])
        b = int(state[0][1])
        #print(a, b)
        #print(np.asmatrix(self.q_table[a, b]))

        output["q_values"] = np.asmatrix(self.q_table[a, b]) # achtung vllt []


        #output["q_values"] = np.random.normal(size=(1, self.action_space))
        #print(output["q_values"])
        #print(np.random.normal(size=(1, self.action_space)))

        return output

    # # TODO:
    def get_weights(self):

        #output =

        return self.q_table


    def set_weights(self, q_vals):

        #print(q_vals)
        self.q_table = q_vals
        pass





    # what else do you need?

"""
def maxQA(q, s):
    max = -9999
    sa = 0
    for k in q[s]:
        if (q[s][k] > max):
            max = q[s][k]
            sa = k
    return sa, max
 """


"""
    # update Q values depending on whether the mode  is doubleQLearning or not
    def updateQValues(doubleQLearning, s, a, r, nxt_s, alpha):
        GAMMA = 1
        if doubleQLearning:
            p = np.random.random()
            if (p < .5):
                nxt_a, maxq = maxQA(Q1, nxt_s)
                Q1[s][a] = Q1[s][a] + alpha * (r + GAMMA * Q2[nxt_s][nxt_a] - Q1[s][a])
            else:
                nxt_a, maxq = maxQA(Q2, nxt_s)
                Q2[s][a] = Q2[s][a] + alpha * (r + GAMMA * Q1[nxt_s][nxt_a] - Q2[s][a])
        else:
            nxt_a, maxq = maxQA(Q1, nxt_s)
            Q1[s][a] = Q1[s][a] + alpha * (r + GAMMA * maxq - Q1[s][a])
        return nxt_a
"""
#   maybe save file

# returns the action that makes the max Q value, as welle as the max Q value



if __name__ == "__main__":
    action_dict = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

    env_kwargs = {
        "height": 3,
        "width": 4,
        "action_dict": action_dict,
        "start_position": (2, 0),
        "reward_position": (0, 3),
    }

    # you can also create your environment like this after installation: env = gym.make('gridworld-v0')
    env = GridWorld(**env_kwargs)

    model_kwargs = {"h": env.height, "w": env.width, "action_space": 4}

    kwargs = {
        "model": TabularQ,
        "environment": GridWorld,
        "num_parallel": 2,
        "total_steps": 5,
        "model_kwargs": model_kwargs,
        "epsilon": 0.5

        # and more
    }

    # initilize
    ray.init(log_to_driver=False)

    manager = SampleManager(**kwargs)
    #print(manager.sample(10))

    manager.get_data(do_print=True)
    #manager.set_agent(4)


    """manager.test(
       max_steps=100,
        test_episodes=10,
        render=True,
        do_print=True,
        evaluation_measure="time_and_reward",
    )
    """

    agent = manager.get_agent()

    # keys for replay buffer -> what you will need for optimization
    optim_keys = ["state", "action", "reward", "state_new", "not_done"]

    # initialize buffer


    #def train(q_table)
    alpha = 0.2
    gamma = 0.85

    epsilon = 0.01
    buffer_size = 5000
    test_steps = 1000
    epochs = 5
    sample_size = 1000
    optim_batch_size = 8
    saving_after = 5


    all_epochs = []
    all_penalties = []
    manager.initilize_buffer(buffer_size, optim_keys)

    for e in range(epochs):
        print("collecting experience..")
        data = manager.get_data()
        manager.store_in_buffer(data)

        # sample data to optimize on from buffer
        sample_dict = manager.sample(sample_size)
        print(f"collected data for: {sample_dict.keys()}")
        # create and batch tf datasets
        data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)
        state, next_state, action, reward = data_dict['state'], data_dict['state_new'], data_dict['action'], data_dict[
            'reward']
        q_table = agent.get_weights().copy()
        for s, s_next, a, r in zip(state, next_state, action, reward):
            old_value = q_table[s[0][0], s[0][1], a[0]]
            next_max = np.max(q_table[s_next[0][0], s_next[0][1]])
            new_value = old_value + alpha * ((r.numpy()[0]) + gamma * next_max - old_value)

            q_table[s[0][0], s[0][1], a[0]] = new_value
            if r.numpy()[0] != 0:
                pass
                # breakpoint()
        # set new weights
        # breakpoint()
        manager.set_agent(q_table)
        # get new weights
        agent = manager.get_agent()
        print(q_table)
        print("das war Epoche", e)
    """
    for i in range(1, 1001):
        state = env.reset()
        epochs, penalties, reward, = 0, 0, 0
        done = False
        #print(state)

        #while not done:
        for e in range(epochs):

            q_table = agent.get_weights()

            #print(q_table)
            
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            


            data = manager.get_data()

            #print(data)

            manager.store_in_buffer(data)

            # sample data to optimize on from buffer
            sample_dict = manager.sample(sample_size)

            print(f"collected data for: {sample_dict.keys()}")

            # create and batch tf datasets
            data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)
            #print(data_dict)
            states = data_dict["state"]
            actions = data_dict["action"]
            rewards = data_dict["reward"]
            new_states = data_dict["state_new"]
            for s, a, r, ns in zip(states, actions, rewards, new_states):
                

            next_state, reward, done, info = env.step(action)

            old_value = q_table[int(state[0]), int(state[1]), action]

            next_max = np.max(q_table[int(next_state[0]), int(next_state[1])])



            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)


            #print(new_value)
            q_copy = q_table.copy()
            q_copy[int(state[0]), int(state[1]), action] = new_value

            #new_weights = agent.model.get_weights()

            #print(new_weights)
            #new_weights = agent.model.get_weights()

            # set new weights
            manager.set_agent(q_copy)


            # get new weights
            agent = manager.get_agent()

            if reward == -1:
                penalties += 1
                state = next_state
                epochs += 1
                print(penalties, epochs)

        if i % 10 == 0:
            print(f"Ep.: {i}")
    """
    print("Training ended.\n")

    manager.test(
       max_steps=100,
        test_episodes=30,
        render=True,
        do_print=True,
        evaluation_measure="time_and_reward",
    )

    """
    buffer_size = 5000
    test_steps = 1000
    epochs = 5
    sample_size = 1000
    optim_batch_size = 8
    saving_after = 5


    agent = manager.get_agent()


    for e in range(epochs):
        # experience replay
        print("collecting experience..")
        data = manager.get_data()




#von mir
    # where to save your results to: create this directory in advance!
    saving_path = os.getcwd() + "/progress_test2"

    buffer_size = 5000
    test_steps = 1000
    epochs = 5
    sample_size = 1000
    optim_batch_size = 8
    saving_after = 5

    # keys for replay buffer -> what you will need for optimization
    optim_keys = ["state", "action", "reward", "state_new", "not_done"]

    # initialize buffer
    manager.initilize_buffer(buffer_size, optim_keys)

    # initilize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=5, aggregator_keys=["loss", "time_steps"]
    )

    # initial testing:
    print("test before training: ")
    #manager.test(test_steps, do_print=True)
    print("test before training: ")

    # get initial agent
    agent = manager.get_agent()

    for e in range(epochs):

        # training core

        # experience replay
        print("collecting experience..")
        data = manager.get_data()
        manager.store_in_buffer(data)

        # sample data to optimize on from buffer
        sample_dict = manager.sample(sample_size)
        print(f"collected data for: {sample_dict.keys()}")
        # create and batch tf datasets
        #data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)

        print("optimizing...")

        # TODO: iterate through your datasets

        # TODO: optimize agent

        dummy_losses = [
            np.mean(np.random.normal(size=(64, 100)), axis=0) for _ in range(1000)
        ]

        new_weights = agent.model.get_weights()

        # set new weights
        manager.set_agent(new_weights)

        # get new weights
        agent = manager.get_agent()
        # update aggregator
        time_steps = manager.test(test_steps)
        manager.update_aggregator(loss=dummy_losses, time_steps=time_steps)
        # print progress
        print(
            f"epoch ::: {e}  loss ::: {np.mean([np.mean(l) for l in dummy_losses])}   avg env steps ::: {np.mean(time_steps)}"
        )

        # yeu can also alter your managers parameters
        manager.set_epsilon(epsilon=0.99)

        if e % saving_after == 0:
            # you can save models
            manager.save_model(saving_path, e)

        # and load mmodels
    manager.load_model(saving_path)
    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=10, render=True)
"""



