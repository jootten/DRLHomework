import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import gym
import ray
from really import SampleManager  # important !!
from really.utils import (
    dict_to_dict_of_datasets,
)  # convenient function for you to create tensorflow datasets


class MyModel(tf.keras.Model):
    def __init__(self, state_dim = 4, action_size = 2):
        super(MyModel, self).__init__()


        self.hidden_layer_1 = tf.keras.layers.Dense(units=16,
                                               activation=tf.keras.activations.relu
                                               )
        self.hidden_layer_2 = tf.keras.layers.Dense(units=16,
                                               activation=tf.keras.activations.relu #tf.nn.leaky_relu
                                               )
        self.output_layer = tf.keras.layers.Dense(units=action_size,
                                               activation=tf.keras.activations.linear, use_bias=False
                                               )

        """
        self.layer = tf.keras.layers.Dense(action_size)
        self.layer2 = tf.keras.layers.Dense(16, activation=tf.nn.tanh)
        self.layer3 = tf.keras.layers.Dense(state_dim)


        self.state_in = tf.placeholder(tf.float32, shape=[None, *state_dim])
        self.action_in = tf.placeholder(tf.int32, shape=[None])
        self.q_target_in = tf.placeholder(tf.float32, shape=[None])
        action_one_hot = tf.one_hot(self.action_in, depth=action_size)

        self.hidden1 = tf.layers.dense(self.state_in, 100, activation=tf.nn.relu)
        self.q_state = tf.layers.dense(self.hidden1, action_size, activation=None)
        self.q_state_action = tf.reduce_sum(tf.multiply(self.q_state, action_one_hot), axis=1)

        self.loss = tf.reduce_mean(tf.square(self.q_state_action - self.q_target_in))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        """

    def call(self, x_in):

        output = {}
        x = self.hidden_layer_1(x_in)
        v = self.hidden_layer_2(x)
        w = self.output_layer(v)
        output["q_values"] = w
        #output["value_estimate"] = v
        return output


"""

class ModelContunous(tf.keras.Model):
    def __init__(self, output_units=2):

        super(ModelContunous, self).__init__()

        self.layer_mu = tf.keras.layers.Dense(output_units)
        self.layer_sigma = tf.keras.layers.Dense(output_units, activation=None)
        self.layer_v = tf.keras.layers.Dense(1)

    def call(self, x_in):

        output = {}
        mus = self.layer_mu(x_in)
        sigmas = tf.exp(self.layer_sigma(x_in))
        v = self.layer_v(x_in)
        output["mu"] = mus
        output["sigma"] = sigmas
        output["value_estimate"] = v

        return output


"""


class QNetwork():
    def __init__(self, state_dim = 4, action_size = 2):
        self.state_in = tf.placeholder(tf.float32, shape=[None, *state_dim])
        self.action_in = tf.placeholder(tf.int32, shape=[None])
        self.q_target_in = tf.placeholder(tf.float32, shape=[None])
        action_one_hot = tf.one_hot(self.action_in, depth=action_size)

        self.hidden1 = tf.layers.dense(self.state_in, 100, activation=tf.nn.relu)
        self.q_state = tf.layers.dense(self.hidden1, action_size, activation=None)
        self.q_state_action = tf.reduce_sum(tf.multiply(self.q_state, action_one_hot), axis=1)

        self.loss = tf.reduce_mean(tf.square(self.q_state_action - self.q_target_in))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def update_model(self, session, state, action, q_target):
        feed = {self.state_in: state, self.action_in: action, self.q_target_in: q_target}
        session.run(self.optimizer, feed_dict=feed)

    def get_q_state(self, session, state):
        q_state = session.run(self.q_state, feed_dict={self.state_in: state})
        return q_state



if __name__ == "__main__":

    kwargs = {
        "model": MyModel,
        "environment": "CartPole-v0",
        "num_parallel": 2,
        "total_steps": 100,
        "action_sampling_type": "epsilon_greedy", #"thompson",#
        "num_episodes": 20,
        "epsilon": 1,
    }

    ray.init(log_to_driver=False)

    manager = SampleManager(**kwargs)
    # where to save your results to: create this directory in advance!
    saving_path = os.getcwd() + "/Save for later"

    buffer_size = 5000
    test_steps = 200
    epochs = 90
    sample_size = 1000
    optim_batch_size = 1 #1
    saving_after = 10
    alpha = 0.2
    gamma = 0.9
    epsilon = 1

    epsilon_step = 0.92 #0.92

    # keys for replay buffer -> what you will need for optimization
    optim_keys = ["state", "action", "reward", "state_new", "not_done"]
    optimizer = tf.keras.optimizers.Adam()
    # initialize buffer
    manager.initilize_buffer(buffer_size, optim_keys)

    # initilize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=5, aggregator_keys=["loss", "time_steps"]
    )

    # initial testing:

    #manager.test(test_steps, do_print=True, render=True)

    #manager.test(test_steps, do_print=True)
    # get initial agent
    load = False
    if(load==True):
        agent = manager.load_model(path=saving_path)
    else: agent = manager.get_agent()

    #manager.test(test_steps, test_episodes=100, render=True)
    print("test before training: ")
    #print()
    #exit()
    for e in range(epochs):
        # training core
        if(e == 0 ):
            data = manager.get_data(total_steps=buffer_size)
        # experience replay
        #print("collecting experience..")
        data = manager.get_data()
        #print(data)
        manager.store_in_buffer(data)

        # sample data to optimize on from buffer
        sample_dict = manager.sample(sample_size)
        #print(sample_dict)
        #print(f"collected data for: {sample_dict.keys()}")
        # create and batch tf datasets
        data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)
        #print(data_dict)
        #print("optimizing...")

        #print(agent.get_weights())
        #print(agent.q_val(2, 0))
        #quit()

        state, next_state, action, reward, not_done = data_dict['state'], data_dict['state_new'], data_dict['action'], data_dict[
            'reward'], data_dict['not_done']



        losses=[]
        #q_table = agent.get_weights().copy()
        for s, s_next, a, r, d in zip(state, next_state, action, reward, not_done):
            #print("s", s)
            #print("s_next", s_next)
            #print("a", a)
            #print("r", r)
            #print("d", d)


            q_target = np.expand_dims(r, -1) + gamma * np.expand_dims(agent.max_q(s_next), -1) * np.expand_dims(d, -1)
            #print("q_target", q_target)

            #breakpoint()
            with tf.GradientTape() as tape:
                loss = tf.keras.losses.MSE(q_target, agent.q_val(s, a))

            gradient = tape.gradient(loss, agent.model.trainable_variables)
            optimizer.apply_gradients(zip(gradient, agent.model.trainable_variables))
            losses.append(loss)

            #old_value = q_table[s[0][0], s[0][1], a[0]]
            #next_max = np.max(q_table[s_next[0][0], s_next[0][1]])
            #new_value = old_value + alpha * ((r.numpy()[0]) + gamma * next_max - old_value)


            #q_table[s[0][0], s[0][1], a[0]] = new_value
            #if r.numpy()[0] != 0:
             #   pass


        # TODO: iterate through your datasets


        # TODO: optimize agent

        #dummy_losses = [
        #    np.mean(np.random.normal(size=(64, 100)), axis=0) for _ in range(1000)
        #]

        new_weights = agent.model.get_weights()

        #print(new_weights)
        # set new weights
        manager.set_agent(new_weights)

        # get new weights
        agent = manager.get_agent()
        # update aggregator
        time_steps = manager.test(test_steps)
        manager.update_aggregator(loss = losses, time_steps=time_steps)
        # print progress
        print(
            f"epoch ::: {e}  loss ::: {np.mean([np.mean(l) for l in losses])}   avg env steps ::: {np.mean(time_steps)}"
        )

        # yeu can also alter your managers parameters
        if(epsilon > 0) :
            epsilon *= epsilon_step
            manager.set_epsilon(epsilon=epsilon) #variabl


        if e % saving_after == 0 and e != 0:
            # you can save models
            manager.save_model(saving_path, e)

    # and load mmodels
    manager.load_model(saving_path)
    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=100, render=True)
