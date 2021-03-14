# TODO
"""
Returned action is maybe just mu and sigma!!
- vor critic loss
"""

from scipy.stats import norm, lognorm

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
import time

import tensorflow.keras.layers as kl
from tensorflow.keras.layers import Layer
import tensorflow_probability as tfp

import math


class MyModel(tf.keras.Model):
    def __init__(self, state_dim=4,
                 normalize_mean=None,
                 normalize_sd=None):
        super(MyModel, self).__init__()
        """

        self.hidden_layer_1 = tf.keras.layers.Dense(units=16, #input_shape=(32, 4),
                                               activation=tf.nn.leaky_relu#tf.nn.relu#, input_dim=state_dim
                                               )
        self.hidden_layer_2 = tf.keras.layers.Dense(units=16,
                                               activation=tf.nn.leaky_relu#tf.nn.relu
                                               )
        self.output_layer = tf.keras.layers.Dense(units=action_size,
                                               activation=tf.keras.activations.linear, use_bias=False
                                               )
        """
        # if normalize_mean is not None:
        #   assert normalize_sd is not None
        #  assert normalize_sd.shape == input_shape
        # assert normalize_mean.shape == input_shape
        # self.normalize_mean = normalize_mean
        # self.normalize_sd = normalize_sd

        self.action_space_size = 2

        self.fc1 = tf.keras.layers.Dense(units=128, activation='relu')  # , kernel_regularizer="l2"
        self.fc2 = tf.keras.layers.Dense(units=64, activation='relu')
        self.fc3 = tf.keras.layers.Dense(units=32, activation='relu')

        self.mu_out = kl.Dense(units=self.action_space_size, activation='tanh')
        self.sigma_out = kl.Dense(units=self.action_space_size, activation='softplus')

    def call(self, x):
        # if self.normalize_mean is not None:
        #   x = (x - self.normalize_mean) / self.normalize_sd
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        mu = self.mu_out(x)

        # sigma = tf.exp(self.sigma_out(x))
        sigma = self.sigma_out(x)
        # sigma = tf.math.exp(self.sigma_out(x))
        output = {}

        # return tf.reshape(mu, [-1, self.action_space_size]), tf.reshape(sigma, [-1, self.action_space_size])

        # output["q_values"] = x #

        output["mu"] = mu
        output["sigma"] = sigma

        # output["q_values"] = tfp.distributions.Normal(mu, sigma)
        return output
        # return tf.reshape(mu, [-1, self.action_space_size]), tf.reshape(sigma, [-1, self.action_space_size])


"""
class Critic(tf.keras.Model):

    def __init__(self, input_shape = (8,),out = 1):
        super(Critic, self).__init__()
        #self.fc1 = kl.Dense(units=128, input_shape=[8, ], activation='relu', kernel_regularizer="l2")
        #self.fc2 = kl.Dense(units=64, activation='relu')
        #self.out = kl.Dense(units=1, activation=None)



        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(128,
                                  #input_shape=input_shape,
                                  activation='relu',
                                  kernel_regularizer='l2'),
            tf.keras.layers.Dense(64,
                                  activation='relu',
                                  kernel_regularizer='l2'),
            tf.keras.layers.Dense(1,
                                  activation=None,
                                  use_bias=False)
        ])

    def call(self, x):
        #x = self.fc1(x)
        #x = self.fc2(x)
        #x = self.out(x)
        x = self.mlp(x)
        x.shape
        return x
"""


class Critic(tf.keras.Model):

    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = kl.Dense(units=32, input_shape=[8, ], activation='relu', kernel_regularizer="l2", trainable=True,
                            dtype='float64')
        self.fc2 = kl.Dense(units=32, activation='relu', trainable=True)
        self.out = kl.Dense(units=1, activation=None, trainable=True)

    def call(self, x):  #

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x


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


if __name__ == "__main__":
    """
    env_id = "LunarLanderContinuous-v2"
    _env = gym.make(env_id)

    model_kwargs = {
        #"input_shape": _env.observation_space.shape,
        #"action_space": _env.action_space.n,
        "normalize_mean": 0, #np.zeros(_env.observation_space.shape),
        "normalize_sd": 1 #np.ones(_env.observation_space.shape)
    }
    """

    kwargs = {
        "model": MyModel,
        "environment": "LunarLanderContinuous-v2",
        "num_parallel": 5,
        "total_steps": 200,
        "num_episodes": 20,
        'returns': ['monte_carlo', 'log_prob'],
        'action_sampling_type': "continous_normal_diagonal",

    }
    # "model_kwargs": model_kwargs
    # "gamma": 0.9,
    moving_mean = 0
    moving_sd = 1

    ray.init(log_to_driver=False)

    manager = SampleManager(**kwargs)
    # where to save your results to: create this directory in advance!
    saving_path = os.getcwd() + "/PPO"
    # buffer_size = 400
    test_steps = 300
    epochs = 1250
    sample_size = 3000
    optim_batch_size = 100  # 1
    saving_after = 10
    # target_kl = 0.01
    # gamma = 0.9
    # alpha = 0.2
    # GAMMA = 0.9
    # epsilon = 0.9
    # epsilon_step = 0.92  # 0.92 #0.8 #0.94

    ENTROPY_COEF = 0.001
    # keys for replay buffer -> what you will need for optimization
    # optim_keys = ["state", "action", "reward", "state_new", "not_done"]

    optimizer_critic = tf.keras.optimizers.Adam(0.001)
    optimizer_actor = tf.keras.optimizers.Adam(lr=0.0003)  # 0.0003 0.
    # initialize buffer
    # manager.initilize_buffer(buffer_size, optim_keys)

    # initilize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=5, aggregator_keys=["loss", "reward"]
    )

    # initial testing:

    # manager.test(test_steps, do_print=True, render=True, test_episodes=5)
    print("Hier")
    # manager.test(test_steps, do_print=True)
    # get initial agent
    load = False
    if (load == True):
        agent = manager.load_model(path=saving_path)
        # manager.test(test_steps, test_episodes=100, render=True)
        print("test before training: ")
    else:
        agent = manager.get_agent()

    start_time = time.time()

    # manager.test(test_steps, do_print=True, render=True)
    # exit()
    critic_net = Critic()  # **manager.kwargs["model_kwargs"]
    # actor_net = MyModel()#
    start_time = time.time()
    print("H", (time.time() - start_time))

    # print("start", time.strftime("%a, %d %b %Y %H:%M:%S +0000", (time.time() - start_time)))
    for e in range(epochs):

        # manager.kwargs["model_kwargs"]["normalize_mean"] = moving_mean
        # manager.kwargs["model_kwargs"]["normalize_sd"] = moving_sd

        step = 0
        start_time_episode = time.time()
        # training core

        # if (e == 0):
        #   data = manager.get_data(total_steps=buffer_size)
        # else:
        #   data = manager.get_data()

        # experience replay
        # print("collecting experience..")
        # print(data)

        # manager.store_in_buffer(data)

        # sample data to optimize on from buffer
        sample_dict = manager.sample(sample_size, from_buffer=False)  #
        # print(sample_dict)
        # print(f"collected data for: {sample_dict.keys()}")
        # create and batch tf datasets
        data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)

        # print(data_dict)
        # print("optimizing...")

        # print(agent.get_weights())
        # print(agent.q_val(2, 0))
        # quit()

        # weights = agent.get_weights()
        # actor_net.set_weights(weights)

        # weights = agent.get_weights()
        # print(weights,"\n\n")
        # print(weights.shape)

        # print(actor_net.get_weights())
        # print(actor_net.get_weights().shape)

        # print("Hierrrr")

        state, next_state, action, reward, not_done, monte_carlo_values, log_prob_old = data_dict['state'], data_dict[
            'state_new'], data_dict['action'], data_dict['reward'], data_dict['not_done'], data_dict['monte_carlo'], \
                                                                                        data_dict['log_prob']

        # actor_net = MyModel(**manager.kwargs["model_kwargs"])
        # actor_net.set_weights(agent.get_weights())

        # breakpoint()
        # use moving averages to normalize state vector
        # moving_mean = np.mean(data_dict['state'], axis=0)
        # moving_sd = np.std(data_dict['state'], axis=0)

        if e == 0: print(len(sample_dict['not_done']))
        """
        state, action, monte_carlo_values = data_dict['state'], data_dict['action'], data_dict['monte_carlo']
        """
        losses_actor = []
        losses_critic = []
        # print("Time in sec", (time.time() - start_time))
        # print("start", time.strftime("%a, %d %b %Y %H:%M:%S +0000", (time.time() - start_time)))
        save_for_debug = {}
        # try:
        if (1 == 3):
            for s, a, m in zip(state, action, monte_carlo_values):
                # print("start", time.strftime("%a, %d %b %Y %H:%M:%S +0000", (time.time() - start_time)))
                # print("Da", (time.time() - start_time))

                with tf.GradientTape() as tape:
                    # TODO Auchtung critic loss explodiert look at shapes
                    critic_est = critic_net(s)  # np.expand_dims(s, 0)
                    # advantages = m - critic_est  #Wirklich monte Carlo hier??
                    # estimated_return = r + GAMMA * critic_est # from memory 57
                    loss_critic = tf.keras.losses.MSE(tf.squeeze(critic_est),
                                                      tf.cast(m, dtype="float32"))  # tf.cast(m, dtype= "float32")

                gradients_critic = tape.gradient(loss_critic, critic_net.trainable_variables)

                optimizer_critic.apply_gradients(zip(gradients_critic, critic_net.trainable_variables))

                advantages = tf.cast(np.expand_dims(m, -1),
                                     dtype="float32") - critic_est  # passt  # Wirklich monte Carlo hier??
                # advantages = 1
                # if(step < 100 and e == 0):
                #   advantages /= 10
                # breakpoint()
                with tf.GradientTape() as tape:
                    # TODO
                    # Hier right dist, normalize, cast
                    output = agent.model(s)
                    mu, sigma = output["mu"], output["sigma"]
                    act_dist = tfp.distributions.Normal(mu, sigma)
                    # act_dist.entropy(m) #Hier monte CARLO
                    # breakpoint()
                    logprob = act_dist.log_prob(tf.cast(a, dtype="float32"))
                    # breakpoint()
                    actor_loss = -logprob * advantages - ENTROPY_COEF * act_dist.entropy()

                    # maybe cast

                gradients_actor = tape.gradient(actor_loss, agent.model.trainable_variables)
                optimizer_actor.apply_gradients(zip(gradients_actor, agent.model.trainable_variables))

                losses_actor.append(np.mean(actor_loss))
                losses_critic.append(np.mean(loss_critic))
                # look in sample manager use monte carlo und logprob
                breakpoint()
                step += 1

        """
        state_sum=0
        states_num = 0
        for s in state:
            state_sum+=s
            states_num +=1

        state_mean = state_sum / states_num
        state_std = np.std(state_mean,0)+1e-20

        nstate_sum=0
        #states_num = 0
        for ns in next_state:
            nstate_sum+=ns
        nstate_mean = np.mean(nstate_sum,0)
        nstate_std = np.std(nstate_mean,0)+1e-20
       """

        if (1 == 1):
            for i, (s, ns, a, r, nd, m, lp_old) in enumerate(
                    zip(state, next_state, action, reward, not_done, monte_carlo_values, log_prob_old)):
                # if(i >= 1): break

                # print("start", time.strftime("%a, %d %b %Y %H:%M:%S +0000", (time.time() - start_time)))
                # moving_mean += 1 / (step + 125) * (np.mean(state, axis=0) - moving_mean)
                # moving_sd += 1 / (step + 125) * (np.std(state, axis=0) - moving_sd)
                # s = (s - state_mean) / state_std
                # normalized = (s - np.mean(s, 0)) / (np.std(s, 0) + 1e-20)
                # nnormalized = (ns - np.mean(ns, 0)) / (np.std(ns, 0) + 1e-20)
                with tf.GradientTape() as tape:

                    critic_est = critic_net(s)
                    # critic_est_next = critic_net(ns)
                    advantages = tf.cast(m, dtype="float32") - tf.squeeze(critic_est)
                    # advantages = np.expand_dims(r, -1) + gamma * critic_est_next - critic_est * np.expand_dims(nd, -1)
                    # advantages = tf.cast(r, dtype="float32") + tf.squeeze(critic_est_next - critic_est) * tf.cast(nd,
                    #                                                                                             dtype="float32")

                    #
                    # breakpoint()

                    # anderer versuch für advantages:
                    # advantages = tf.cast(m, dtype = "float32" )  - tf.squeeze(critic_est) #Wirklich monte Carlo hier??

                    # loss_critic = tf.keras.losses.MSE(tf.squeeze(critic_est), tf.cast(m, dtype = "float32" )) #tf.cast(m, dtype= "float32")
                    loss_critic = pow(advantages, 2)
                    loss_critic = tf.reduce_mean(loss_critic)
                    # breakpoint()

                    # advantage = tf.cast(r, dtype = "float32" ) + tf.cast(nd, dtype = "float32" )*tf.squeeze(gamma * (critic_net(ns) - critic_net(s)))
                    # breakpoint()
                    # loss_critic = pow(advantage,2)

                # breakpoint()
                gradients_critic = tape.gradient(loss_critic, critic_net.trainable_variables)
                # breakpoint()
                optimizer_critic.apply_gradients(zip(gradients_critic, critic_net.trainable_variables))

                # advantages = tf.cast(np.expand_dims(m, -1), dtype="float32") - critic_est#passt  # Wirklich monte Carlo hier??
                # advantages = 1
                # if(step < 100 and e == 0):
                #   advantages /= 10
                # breakpoint()
                with tf.GradientTape() as tape:
                    output = agent.model(s)
                    # mu, sigma = output["mu"].numpy(), output["sigma"].numpy()
                    mu, sigma = output["mu"], output["sigma"]
                    # Achtung Sigma explodiert!!!!!!!!!!
                    tf.clip_by_value(sigma, -1000, 1000)
                    # breakpoint()
                    # print("mus", mus)
                    # print("sigmas", sigmas)

                    # new_action = norm.rvs(mu, sigma)

                    # output["action"] = action
                    # logging.warning('action')
                    # logging.warning(action)

                    # if return_log_prob:
                    act_dist = tfp.distributions.Normal(mu, sigma)
                    # logprob_new = np.sum(norm.logpdf(a, mu, sigma), axis=-1)
                    logprob_new = tf.reduce_sum(act_dist.log_prob(tf.cast(a, dtype="float32")), axis=-1)
                    # tf.reduce_sum(act_dist.log_prob(tf.cast(a, dtype="float32")), axis=-1)
                    # next try
                    # normalized = (s-np.mean(s,0))/(np.std(s,0)+1e-20) #to avoid division by zero
                    """
                    output = agent.model(nnormalized) #agent.model(np.expand_dims(s, 0))
                    mu, sigma = output["mu"], output["sigma"]

                    act_dist = tfp.distributions.Normal(mu, sigma)
                    logprob_new = tf.clip_by_value(act_dist = tfp.distributions.Normal(mu, sigma),-1e10, 1e10) #achtung log prob vllt inf !!!!!
                    #breakpoint()
                    """
                    # KL_div = tf.math.exp(logprob_new - tf.cast(lp_old, dtype="float32"))
                    # breakpoint()
                    KL_div = tf.math.exp(logprob_new - tf.cast(lp_old, dtype="float32"))
                    # breakpoint()

                    if np.isnan(KL_div).any():
                        print("output is nan, KL clipped to 1.2")
                        KL_div = 1.2
                        L_clip = KL_div * advantages
                        breakpoint()
                    else:
                        L_clip = tf.minimum(KL_div * advantages,
                                            (tf.clip_by_value(KL_div, 1 - 0.2, 1 + 0.2) * advantages))
                    # actor_loss =  -logprob_ * np.expand_dims(advantages, -1) - ENTROPY_COEF * act_dist.entropy() # brauch ich das?
                    # breakpoint()
                    # actor_loss = -np.expand_dims(L_clip, -1)# - ENTROPY_COEF * act_dist.entropy()  # ich brauch das noch, aber shpae passt nicht, #TODO mit - ???

                    actor_loss = -tf.reduce_mean(L_clip)

                # breakpoint()
                gradients_actor = tape.gradient(actor_loss, agent.model.trainable_variables)

                optimizer_actor.apply_gradients(zip(gradients_actor, agent.model.trainable_variables))

                # breakpoint()
                losses_actor.append(np.mean(actor_loss))
                losses_critic.append(np.mean(loss_critic))
                # look in sample manager use monte carlo und logprob
                # breakpoint()
                step += 1
                # print(KL_div)
                # breakpoint()
                # print(step)
                save_for_debug["L_clip"], save_for_debug["logprob_new"], save_for_debug["lp_old"], save_for_debug[
                    "advantages"], save_for_debug["agent.model.trainable_variables"], save_for_debug["output"], \
                save_for_debug[
                    "a"] = L_clip, logprob_new, lp_old, advantages, agent.model.trainable_variables, output, a
                if np.isnan(actor_loss):
                    breakpoint()

                # early stopping
                if (tf.reduce_mean(KL_div).numpy() > 1.5):
                    print(tf.reduce_mean(KL_div).numpy(), f"---------- early stopping after {i} steps")
                    break

        if (1 == 2):
            for s, ns, a, r, nd, m in zip(state, next_state, action, reward, not_done, monte_carlo_values):
                # for s, a, m in zip(state, action, monte_carlo_values):
                # print("start", time.strftime("%a, %d %b %Y %H:%M:%S +0000", (time.time() - start_time)))
                # print("Da", (time.time() - start_time))
                # advantage = 0
                with tf.GradientTape() as tape:
                    # TODO Auchtung critic loss explodiert look at shapes
                    critic_est = critic_net(np.expand_dims(s, 0))

                    # breakpoint()
                    advantages = tf.cast(m, dtype="float32") - tf.squeeze(critic_est)  # Wirklich monte Carlo hier??
                    # estimated_return = r + GAMMA * critic_est # from memory 57
                    loss_critic = tf.keras.losses.MSE(tf.squeeze(critic_est),
                                                      tf.cast(m, dtype="float32"))  # tf.cast(m, dtype= "float32")
                    # breakpoint()

                    # advantage = tf.cast(r, dtype = "float32" ) + tf.cast(nd, dtype = "float32" )*tf.squeeze(gamma * (critic_net(ns) - critic_net(s)))
                    # breakpoint()
                    # loss_critic = pow(advantage,2)

                gradients_critic = tape.gradient(loss_critic, critic_net.trainable_variables)
                # breakpoint()
                optimizer_critic.apply_gradients(zip(gradients_critic, critic_net.trainable_variables))

                # advantages = tf.cast(np.expand_dims(m, -1), dtype="float32") - critic_est#passt  # Wirklich monte Carlo hier??
                # advantages = 1
                # if(step < 100 and e == 0):
                #   advantages /= 10
                # breakpoint()
                with tf.GradientTape() as tape:
                    # TODO
                    # Hier right dist, normalize, cast
                    output = agent.model(s)  # np.expand_dims(0)
                    mu, sigma = output["mu"], output["sigma"]
                    act_dist = tfp.distributions.Normal(mu, sigma)
                    # act_dist.entropy(m) #Hier monte CARLO
                    # breakpoint()
                    logprob = act_dist.log_prob(tf.cast(a, dtype="float32"))
                    # breakpoint()

                    # breakpoint()
                    actor_loss = -logprob * advantages - ENTROPY_COEF * act_dist.entropy()  # brauch ich das?

                    # maybe cast

                gradients_actor = tape.gradient(actor_loss, agent.model.trainable_variables)
                optimizer_actor.apply_gradients(zip(gradients_actor, agent.model.trainable_variables))
                # breakpoint()
                losses_actor.append(np.mean(actor_loss))
                losses_critic.append(np.mean(loss_critic))
                # look in sample manager use monte carlo und logprob
                # breakpoint()
                step += 1
                print(step)

        # new try without for
        """
        #for s, a, m in zip(state, action, monte_carlo_values):
        #print("start", time.strftime("%a, %d %b %Y %H:%M:%S +0000", (time.time() - start_time)))
        #print("Da", (time.time() - start_time))
        breakpoint()
        with tf.GradientTape() as tape:
            #TODO Auchtung critic loss explodiert look at shapes
            critic_est = critic_net(state)#np.expand_dims(s, 0)
            # advantages = m - critic_est  #Wirklich monte Carlo hier??
            # estimated_return = r + GAMMA * critic_est # from memory 57
            loss_critic = tf.keras.losses.MSE(tf.squeeze(critic_est), tf.cast(monte_carlo_values, dtype = "float32" )) #tf.cast(m, dtype= "float32")



        gradients_critic = tape.gradient(loss_critic, critic_net.trainable_variables)
        #breakpoint()
        optimizer_critic.apply_gradients(zip(gradients_critic, critic_net.trainable_variables))

        advantages = tf.cast(np.expand_dims(monte_carlo_values, -1), dtype="float32") - critic_est#passt  # Wirklich monte Carlo hier??
        #advantages = 1
        #if(step < 100 and e == 0):
         #   advantages /= 10
        #breakpoint()
        with tf.GradientTape() as tape:
            #TODO
            #Hier right dist, normalize, cast
            output = agent.model(state)
            mu, sigma = output["mu"], output["sigma"]
            act_dist = tfp.distributions.Normal(mu, sigma)
            #act_dist.entropy(m) #Hier monte CARLO
            #breakpoint()
            logprob = act_dist.log_prob(tf.cast(action, dtype= "float32"))
            #breakpoint()
            actor_loss =  -logprob * advantages - ENTROPY_COEF * act_dist.entropy()

            #maybe cast

        gradients_actor = tape.gradient(actor_loss, agent.model.trainable_variables)
        optimizer_actor.apply_gradients(zip(gradients_actor, agent.model.trainable_variables))

        losses_actor.append(np.mean(actor_loss))
        losses_critic.append(np.mean(loss_critic))
        # look in sample manager use monte carlo und logprob
        #
        #step +=1

        """
        print(
            f"epoch ::: {e} losses_actor ::: {np.mean(losses_actor)}  loss_critic ::: {np.mean(loss_critic)} Time in sec::: {(time.time() - start_time)}"
            #
        )
        new_weights = agent.model.get_weights()

        # print(new_weights)
        # set new weights
        manager.set_agent(new_weights)

        # get new weights
        agent = manager.get_agent()
        if (e % 10 == 0 and e != 0):

            # manager.test(test_steps, test_episodes=2, do_print=True, render=True)

            # print progress
            reward_test = manager.test(test_steps, evaluation_measure="reward", render=True, test_episodes=5)
            manager.update_aggregator(loss=losses_actor, reward=reward_test)  # time_steps=time_steps,

            print(
                f"epoch ::: {e} losses_actor ::: {round(np.mean(losses_actor), 3)}  loss_critic ::: {round(np.mean(loss_critic), 3)} avg reward ::: {round(np.mean(reward_test), 3)} Time in sec::: {round((time.time() - start_time), 3)}"
                #
            )

            if ((e % saving_after == 0) & (e != 0)):
                # you can save models
                manager.save_model(saving_path, e)

    manager.test(test_steps, do_print=True, render=True)
