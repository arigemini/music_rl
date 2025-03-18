import collections
import os
import random
import urllib

import matplotlib.pyplot as plt
from note_seq import melodies_lib as mlib
from note_seq import midi_io
import numpy as np
import scipy.special
import tensorflow.compat.v1 as tf

from magenta.models.rl_tuner import rl_tuner_eval_metrics
from magenta.models.rl_tuner import rl_tuner_ops

from music_theory_mixin import MusicTheoryMixin
import note_rnn_loader


TRAIN_SEQUENCE_LENGTH = 192

class MusicRl(MusicTheoryMixin):
    # References https://github.com/magenta/magenta/tree/main/magenta/models/rl_tuner

    def __init__(self, output_dir, dqn_hparams, reward_mode, reward_scaler, output_every_nth, num_notes_in_melody):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.input_size = rl_tuner_ops.NUM_CLASSES
            self.num_actions = rl_tuner_ops.NUM_CLASSES
            self.output_every_nth = output_every_nth
            self.output_dir = output_dir
            self.save_path = os.path.join(output_dir, 'music_rl.ckpt')
            self.reward_scaler = reward_scaler
            self.reward_mode = reward_mode
            self.num_notes_in_melody = num_notes_in_melody

            print('Retrieving checkpoint of Note RNN from Magenta download server.')
            urllib.request.urlretrieve('http://download.magenta.tensorflow.org/models/rl_tuner_note_rnn.ckpt', 'note_rnn.ckpt')
            self.note_rnn_checkpoint_dir = os.getcwd()
            self.note_rnn_checkpoint_file = os.path.join(self.note_rnn_checkpoint_dir, 'note_rnn.ckpt')
            self.note_rnn_hparams = rl_tuner_ops.default_hparams()

            self.dqn_hparams = dqn_hparams
            self.discount_rate = tf.constant(self.dqn_hparams.discount_rate)
            self.target_network_update_rate = tf.constant(self.dqn_hparams.target_network_update_rate)

            self.optimizer = tf.train.AdamOptimizer()

            self.actions_executed_so_far = 0
            self.experience = collections.deque(maxlen=self.dqn_hparams.max_experience)
            self.iteration = 0
            self.num_times_store_called = 0
            self.num_times_train_called = 0

        self.reward_last_n = 0
        self.rewards_batched = []
        self.music_theory_reward_last_n = 0
        self.music_theory_rewards_batched = []
        self.note_rnn_reward_last_n = 0
        self.note_rnn_rewards_batched = []
        self.eval_avg_reward = []
        self.eval_avg_music_theory_reward = []
        self.eval_avg_note_rnn_reward = []
        self.target_val_list = []

        self.beat = 0
        self.composition = []
        self.composition_direction = 0
        self.leapt_from = None
        self.steps_since_last_leap = 0

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.initialize_internal_models_graph_session()

    def initialize_internal_models_graph_session(self):
        with self.graph.as_default():
            # Add internal networks to the graph.
            tf.logging.info('Initializing q network')
            self.q_network = note_rnn_loader.NoteRNNLoader(
                    self.graph, 'q_network',
                    self.note_rnn_checkpoint_dir,
                    checkpoint_file=self.note_rnn_checkpoint_file,
                    hparams=self.note_rnn_hparams)

            tf.logging.info('Initializing target q network')
            self.target_q_network = note_rnn_loader.NoteRNNLoader(
                    self.graph,
                    'target_q_network',
                    self.note_rnn_checkpoint_dir,
                    checkpoint_file=self.note_rnn_checkpoint_file,
                    hparams=self.note_rnn_hparams)

            tf.logging.info('Initializing reward network')
            self.reward_rnn = note_rnn_loader.NoteRNNLoader(
                    self.graph, 'reward_rnn',
                    self.note_rnn_checkpoint_dir,
                    checkpoint_file=self.note_rnn_checkpoint_file,
                    hparams=self.note_rnn_hparams)

            tf.logging.info('Q network cell: %s', self.q_network.cell)

            # Add rest of variables to graph.
            tf.logging.info('Adding RL graph variables')
            self.build_graph()

            # Prepare saver and session.
            self.saver = tf.train.Saver()
            self.session = tf.Session(graph=self.graph)
            self.session.run(tf.global_variables_initializer())

            self.q_network.initialize_and_restore(self.session)
            self.target_q_network.initialize_and_restore(self.session)
            self.reward_rnn.initialize_and_restore(self.session)

            # Double check that the model was initialized from checkpoint properly.
            reward_vars = self.reward_rnn.variables()
            q_vars = self.q_network.variables()

            reward1 = self.session.run(reward_vars[0])
            q1 = self.session.run(q_vars[0])

            if np.sum((q1 - reward1)**2) == 0.0:
                tf.logging.info('\nSuccessfully initialized internal nets from checkpoint!')
            else:
                tf.logging.fatal('Error! The model was not initialized from checkpoint properly')

    def prime_internal_model(self, model):
        return self.get_random_note()

    def get_random_note(self):
        note_idx = np.random.randint(0, self.num_actions - 1)
        return np.array(rl_tuner_ops.make_onehot([note_idx], self.num_actions)).flatten()

    def reset_composition(self):
        self.beat = 0
        self.composition = []
        self.composition_direction = 0
        self.leapt_from = None
        self.steps_since_last_leap = 0

    def build_graph(self):
        tf.logging.info('Adding reward computation portion of the graph')
        with tf.name_scope('reward_computation'):
            self.reward_scores = tf.identity(self.reward_rnn(), name='reward_scores')

        tf.logging.info('Adding taking action portion of graph')
        with tf.name_scope('taking_action'):
            self.action_scores = tf.identity(self.q_network(), name='action_scores')
            tf.summary.histogram('action_scores', self.action_scores)
            self.action_softmax = tf.nn.softmax(self.action_scores, name='action_softmax')
            self.predicted_actions = tf.one_hot(tf.argmax(self.action_scores, dimension=1, name='predicted_actions'), self.num_actions)

        tf.logging.info('Add estimating future rewards portion of graph')
        with tf.name_scope('estimating_future_rewards'):
            self.next_action_scores = tf.stop_gradient(self.target_q_network())
            tf.summary.histogram('target_action_scores', self.next_action_scores)
            self.rewards = tf.placeholder(tf.float32, (None,), name='rewards')
            self.target_vals = tf.reduce_max(self.next_action_scores, reduction_indices=[1,])

            self.future_rewards = self.rewards + self.discount_rate * self.target_vals

        tf.logging.info('Adding q value prediction portion of graph')
        with tf.name_scope('q_value_prediction'):
            self.action_mask = tf.placeholder(tf.float32, (None, self.num_actions), name='action_mask')
            self.masked_action_scores = tf.reduce_sum(self.action_scores * self.action_mask, reduction_indices=[1,])

            temp_diff = self.masked_action_scores - self.future_rewards

            self.prediction_error = tf.reduce_mean(tf.square(temp_diff))

            # Compute gradients.
            self.params = tf.trainable_variables()
            self.gradients = self.optimizer.compute_gradients(self.prediction_error)

            # Clip gradients.
            for i, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    self.gradients[i] = (tf.clip_by_norm(grad, 5), var)

            for grad, var in self.gradients:
                tf.summary.histogram(var.name, var)
                if grad is not None:
                    tf.summary.histogram(var.name + '/gradients', grad)

            # Backprop.
            self.train_op = self.optimizer.apply_gradients(self.gradients)

        tf.logging.info('Adding target network update portion of graph')
        with tf.name_scope('target_network_update'):
            self.target_network_update = []
            for v_source, v_target in zip(self.q_network.variables(), self.target_q_network.variables()):
                update_op = v_target.assign_sub(self.target_network_update_rate * (v_target - v_source))
                self.target_network_update.append(update_op)
            self.target_network_update = tf.group(*self.target_network_update)

        tf.summary.scalar('prediction_error', self.prediction_error)

        self.summarize = tf.summary.merge_all()
        self.no_op1 = tf.no_op()

    def train(self, num_steps, exploration_period):
        tf.logging.info('Evaluating initial model...')
        self.evaluate_model()

        self.actions_executed_so_far = 0

        self.reset_composition()
        last_observation = self.prime_internal_models()

        for i in range(num_steps):
            state = np.array(self.q_network.state_value).flatten()

            action, new_observation, reward_scores = self.action(
                last_observation, exploration_period, enable_random=True, sample_next_obs=False
            )

            new_state = np.array(self.q_network.state_value).flatten()
            new_reward_state = np.array(self.reward_rnn.state_value).flatten()

            reward = self.collect_reward(last_observation, new_observation, reward_scores)

            self.store(last_observation, state, action, reward, new_observation, new_state, new_reward_state)

            # Used to keep track of how the reward is changing over time.
            self.reward_last_n += reward

            self.composition.append(np.argmax(new_observation))
            self.beat += 1

            if i > 0 and i % self.output_every_nth == 0:
                tf.logging.info('Evaluating model...')
                self.evaluate_model()
                self.save_model('q')

                self.rewards_batched.append(self.reward_last_n)
                self.music_theory_rewards_batched.append(self.music_theory_reward_last_n)
                self.note_rnn_rewards_batched.append(self.note_rnn_reward_last_n)

                # Save a checkpoint.
                save_step = len(self.rewards_batched)*self.output_every_nth
                self.saver.save(self.session, self.save_path, global_step=save_step)

                r = self.reward_last_n
                tf.logging.info('Training iteration %s', i)
                tf.logging.info('\tReward for last %s steps: %s', self.output_every_nth, r)
                tf.logging.info('\t\tMusic theory reward: %s', self.music_theory_reward_last_n)
                tf.logging.info('\t\tNote RNN reward: %s', self.note_rnn_reward_last_n)

                exploration_p = rl_tuner_ops.linear_annealing(
                    self.actions_executed_so_far, exploration_period, 1.0, self.dqn_hparams.random_action_probability
                )
                tf.logging.info('\tExploration probability is %s', exploration_p)

                self.reward_last_n = 0
                self.music_theory_reward_last_n = 0
                self.note_rnn_reward_last_n = 0

            # Backprop.
            self.training_step()

            # Update current state as last state.
            last_observation = new_observation

            # Reset the state after each composition is complete.
            if self.beat % self.num_notes_in_melody == 0:
                tf.logging.debug('\nResetting composition!\n')
                self.reset_composition()
                last_observation = self.prime_internal_models()

    def action(self, observation, exploration_period=0, enable_random=True, sample_next_obs=False):
        assert len(observation.shape) == 1, 'Single observation only'

        self.actions_executed_so_far += 1

        exploration_p = rl_tuner_ops.linear_annealing(
            self.actions_executed_so_far, exploration_period, 1.0, self.dqn_hparams.random_action_probability
        )

        # Run the observation through the q_network.
        input_batch = np.reshape(observation, (self.q_network.batch_size, 1, self.input_size))
        lengths = np.full(self.q_network.batch_size, 1, dtype=int)

        (action, action_softmax, self.q_network.state_value, reward_scores, self.reward_rnn.state_value) = self.session.run(
            [self.predicted_actions, self.action_softmax, self.q_network.state_tensor, self.reward_scores, self.reward_rnn.state_tensor],
            {
                self.q_network.melody_sequence: input_batch,
                self.q_network.initial_state: self.q_network.state_value,
                self.q_network.lengths: lengths,
                self.reward_rnn.melody_sequence: input_batch,
                self.reward_rnn.initial_state: self.reward_rnn.state_value,
                self.reward_rnn.lengths: lengths,
            }
        )

        reward_scores = np.reshape(reward_scores, (self.num_actions))
        action_softmax = np.reshape(action_softmax, (self.num_actions))
        action = np.reshape(action, (self.num_actions))

        if enable_random and random.random() < exploration_p:
            note = self.get_random_note()
            return note, note, reward_scores
        else:
            if not sample_next_obs:
                return action, action, reward_scores
            else:
                obs_note = rl_tuner_ops.sample_softmax(action_softmax)
                next_obs = np.array(
                        rl_tuner_ops.make_onehot([obs_note], self.num_actions)).flatten()
                return action, next_obs, reward_scores

    def store(self, observation, state, action, reward, newobservation, newstate, new_reward_state):
        if self.num_times_store_called % self.dqn_hparams.store_every_nth == 0:
            self.experience.append((observation, state, action, reward, newobservation, newstate, new_reward_state))
        self.num_times_store_called += 1

    def training_step(self):
        if self.num_times_train_called % self.dqn_hparams.train_every_nth == 0:
            if len(self.experience) < self.dqn_hparams.minibatch_size:
                return

            # Sample experience.
            samples = random.sample(range(len(self.experience)),
                                                            self.dqn_hparams.minibatch_size)
            samples = [self.experience[i] for i in samples]

            # Batch states.
            states = np.empty((len(samples), self.q_network.cell.state_size))
            new_states = np.empty((len(samples),
                                                         self.target_q_network.cell.state_size))
            reward_new_states = np.empty((len(samples),
                                                                        self.reward_rnn.cell.state_size))
            observations = np.empty((len(samples), self.input_size))
            new_observations = np.empty((len(samples), self.input_size))
            action_mask = np.zeros((len(samples), self.num_actions))
            rewards = np.empty((len(samples),))
            lengths = np.full(len(samples), 1, dtype=int)

            for i, (o, s, a, r, new_o, new_s, reward_s) in enumerate(samples):
                observations[i, :] = o
                new_observations[i, :] = new_o
                states[i, :] = s
                new_states[i, :] = new_s
                action_mask[i, :] = a
                rewards[i] = r
                reward_new_states[i, :] = reward_s

            observations = np.reshape(observations, (len(samples), 1, self.input_size))
            new_observations = np.reshape(new_observations, (len(samples), 1, self.input_size))

            _, _, target_vals, _ = self.session.run(
                [
                    self.prediction_error,
                    self.train_op,
                    self.target_vals,
                    self.no_op1,
                ],
                {
                    self.q_network.melody_sequence: observations,
                    self.q_network.initial_state: states,
                    self.q_network.lengths: lengths,
                    self.target_q_network.melody_sequence: new_observations,
                    self.target_q_network.initial_state: new_states,
                    self.target_q_network.lengths: lengths,
                    self.action_mask: action_mask,
                    self.rewards: rewards,
                }
            )

            total_logs = (self.iteration * self.dqn_hparams.train_every_nth)
            if total_logs % self.output_every_nth == 0:
                self.target_val_list.append(np.mean(target_vals))

            self.session.run(self.target_network_update)

            self.iteration += 1

        self.num_times_train_called += 1

    def evaluate_model(self, num_trials=100, sample_next_obs=True):
        note_rnn_rewards = [0] * num_trials
        music_theory_rewards = [0] * num_trials
        total_rewards = [0] * num_trials

        for t in range(num_trials):

            last_observation = self.prime_internal_models()
            self.reset_composition()

            for _ in range(self.num_notes_in_melody):
                _, new_observation, reward_scores = self.action(
                        last_observation,
                        0,
                        enable_random=False,
                        sample_next_obs=sample_next_obs)

                note_rnn_reward = self.reward_from_reward_rnn_scores(new_observation, reward_scores)
                music_theory_reward = self.reward_music_theory(new_observation)
                adjusted_mt_reward = self.reward_scaler * music_theory_reward
                total_reward = note_rnn_reward + adjusted_mt_reward

                note_rnn_rewards[t] = note_rnn_reward
                music_theory_rewards[t] = music_theory_reward * self.reward_scaler
                total_rewards[t] = total_reward

                self.composition.append(np.argmax(new_observation))
                self.beat += 1
                last_observation = new_observation

        self.eval_avg_reward.append(np.mean(total_rewards))
        self.eval_avg_note_rnn_reward.append(np.mean(note_rnn_rewards))
        self.eval_avg_music_theory_reward.append(np.mean(music_theory_rewards))

    def collect_reward(self, obs, action, reward_scores):
        # Gets and saves log p(a|s) as output by reward_rnn.
        note_rnn_reward = self.reward_from_reward_rnn_scores(action, reward_scores)
        self.note_rnn_reward_last_n += note_rnn_reward

        if self.reward_mode == 'scale':
            # Makes the model play a scale (defaults to c major).
            reward = self.reward_scale(obs, action)

        elif self.reward_mode == 'key':
            # Makes the model play within a key.
            reward = self.reward_key_distribute_prob(action)

        elif self.reward_mode == 'music_theory_all':
            tf.logging.debug('Note RNN reward: %s', note_rnn_reward)
            reward = self.reward_music_theory(action)

            tf.logging.debug('Total music theory reward: %s', self.reward_scaler * reward)
            tf.logging.debug('Total note rnn reward: %s', note_rnn_reward)

            self.music_theory_reward_last_n += reward * self.reward_scaler
            return reward * self.reward_scaler + note_rnn_reward

        elif self.reward_mode == 'music_theory_only':
            reward = self.reward_music_theory(action)

        else:
            tf.logging.fatal('ERROR! Not a valid reward mode. Cannot compute reward')

        self.music_theory_reward_last_n += reward * self.reward_scaler
        return reward * self.reward_scaler

    def reward_from_reward_rnn_scores(self, action, reward_scores):
        action_note = np.argmax(action)
        normalization_constant = scipy.special.logsumexp(reward_scores)
        return reward_scores[action_note] - normalization_constant

    def get_reward_rnn_scores(self, observation, state):
        state = np.atleast_2d(state)

        input_batch = np.reshape(observation, (self.reward_rnn.batch_size, 1,
                                                                                     self.num_actions))
        lengths = np.full(self.reward_rnn.batch_size, 1, dtype=int)

        rewards, = self.session.run(
                self.reward_scores,
                {self.reward_rnn.melody_sequence: input_batch,
                 self.reward_rnn.initial_state: state,
                 self.reward_rnn.lengths: lengths})
        return rewards

    def generate_music_sequence(self, title='rltuner_sample', visualize_probs=False, prob_image_name=None, length=None, most_probable=False):
        if length is None:
            length = self.num_notes_in_melody

        self.reset_composition()
        next_obs = self.prime_internal_models()
        tf.logging.info('Priming with note %s', np.argmax(next_obs))

        lengths = np.full(self.q_network.batch_size, 1, dtype=int)

        if visualize_probs:
            prob_image = np.zeros((self.input_size, length))

        generated_seq = [0] * length
        for i in range(length):
            input_batch = np.reshape(next_obs, (self.q_network.batch_size, 1, self.num_actions))

            softmax, self.q_network.state_value = self.session.run(
                [self.action_softmax, self.q_network.state_tensor],
                {
                    self.q_network.melody_sequence: input_batch,
                    self.q_network.initial_state: self.q_network.state_value,
                    self.q_network.lengths: lengths,
                }
            )
            softmax = np.reshape(softmax, (self.num_actions))

            if visualize_probs:
                prob_image[:, i] = softmax  # np.log(1.0 + softmax)

            if most_probable:
                sample = np.argmax(softmax)
            else:
                sample = rl_tuner_ops.sample_softmax(softmax)
            generated_seq[i] = sample
            next_obs = np.array(rl_tuner_ops.make_onehot([sample], self.num_actions)).flatten()

        tf.logging.info('Generated sequence: %s', generated_seq)
        melody = mlib.Melody(rl_tuner_ops.decoder(generated_seq, self.q_network.transpose_amount))

        sequence = melody.to_sequence(qpm=rl_tuner_ops.DEFAULT_QPM)
        filename = rl_tuner_ops.get_next_file_name(self.output_dir, title, 'mid')
        midi_io.sequence_proto_to_midi_file(sequence, filename)

        tf.logging.info('Wrote a melody to %s', self.output_dir)

        if visualize_probs:
            tf.logging.info('Visualizing note selection probabilities:')
            plt.figure()
            plt.imshow(prob_image, interpolation='none', cmap='Reds')
            plt.ylabel('Note probability')
            plt.xlabel('Time (beat)')
            plt.gca().invert_yaxis()
            if prob_image_name is not None:
                plt.savefig(self.output_dir + '/' + prob_image_name)
            else:
                plt.show()

    def evaluate_music_theory_metrics(self, num_compositions=10000, key=None, tonic_note=rl_tuner_ops.C_MAJOR_TONIC):
        return rl_tuner_eval_metrics.compute_composition_stats(
            self,
            num_compositions=num_compositions,
            composition_length=self.num_notes_in_melody,
            key=key,
            tonic_note=tonic_note
        )

    def save_model(self, name='q', directory=None):
        if directory is None:
            directory = self.output_dir

        save_loc = os.path.join(directory, name)
        self.saver.save(self.session, save_loc,
                                        global_step=len(self.rewards_batched)*self.output_every_nth)

        self.save_stored_rewards(name)

    def save_stored_rewards(self, file_name):
        training_epochs = len(self.rewards_batched) * self.output_every_nth
        filename = os.path.join(self.output_dir,
                                                        file_name + '-' + str(training_epochs))
        np.savez(filename,
                         train_rewards=self.rewards_batched,
                         train_music_theory_rewards=self.music_theory_rewards_batched,
                         train_note_rnn_rewards=self.note_rnn_rewards_batched,
                         eval_rewards=self.eval_avg_reward,
                         eval_music_theory_rewards=self.eval_avg_music_theory_reward,
                         eval_note_rnn_rewards=self.eval_avg_note_rnn_reward,
                         target_val_list=self.target_val_list)

    def save_model_and_figs(self, name, directory=None):
        self.save_model(name, directory=directory)
        self.plot_rewards(image_name='TrainRewards-' + name + '.eps',
                                            directory=directory)
        self.plot_evaluation(image_name='EvaluationRewards-' + name + '.eps',
                                                 directory=directory)
        self.plot_target_vals(image_name='TargetVals-' + name + '.eps',
                                                    directory=directory)

    def plot_rewards(self, image_name=None, directory=None):
        if directory is None:
            directory = self.output_dir

        reward_batch = self.output_every_nth
        x = [reward_batch * i for i in np.arange(len(self.rewards_batched))]
        plt.figure()
        plt.plot(x, self.rewards_batched)
        plt.plot(x, self.music_theory_rewards_batched)
        plt.plot(x, self.note_rnn_rewards_batched)
        plt.xlabel('Training epoch')
        plt.ylabel('Cumulative reward for last ' + str(reward_batch) + ' steps')
        plt.legend(['Total', 'Music theory', 'Note RNN'], loc='best')
        if image_name is not None:
            plt.savefig(directory + '/' + image_name)
        else:
            plt.show()

    def plot_evaluation(self, image_name=None, directory=None, start_at_epoch=0):
        if directory is None:
            directory = self.output_dir

        reward_batch = self.output_every_nth
        x = [reward_batch * i for i in np.arange(len(self.eval_avg_reward))]
        start_index = start_at_epoch // self.output_every_nth
        plt.figure()
        plt.plot(x[start_index:], self.eval_avg_reward[start_index:])
        plt.plot(x[start_index:], self.eval_avg_music_theory_reward[start_index:])
        plt.plot(x[start_index:], self.eval_avg_note_rnn_reward[start_index:])
        plt.xlabel('Training epoch')
        plt.ylabel('Average reward')
        plt.legend(['Total', 'Music theory', 'Note RNN'], loc='best')
        if image_name is not None:
            plt.savefig(directory + '/' + image_name)
        else:
            plt.show()

    def plot_target_vals(self, image_name=None, directory=None):
        if directory is None:
            directory = self.output_dir

        reward_batch = self.output_every_nth
        x = [reward_batch * i for i in np.arange(len(self.target_val_list))]

        plt.figure()
        plt.plot(x, self.target_val_list)
        plt.xlabel('Training epoch')
        plt.ylabel('Target value')
        if image_name is not None:
            plt.savefig(directory + '/' + image_name)
        else:
            plt.show()

    def prime_internal_models(self):
        self.prime_internal_model(self.target_q_network)
        self.prime_internal_model(self.reward_rnn)
        next_obs = self.prime_internal_model(self.q_network)
        return next_obs

    def restore_from_directory(self, directory=None, checkpoint_name=None, reward_file_name=None):
        if directory is None:
            directory = self.output_dir

        if checkpoint_name is not None:
            checkpoint_file = os.path.join(directory, checkpoint_name)
        else:
            tf.logging.info('Directory %s.', directory)
            checkpoint_file = tf.train.latest_checkpoint(directory)

        if checkpoint_file is None:
            tf.logging.fatal('Error! Cannot locate checkpoint in the directory')
            return
        tf.logging.info('Attempting to restore from checkpoint %s', checkpoint_file)

        self.saver.restore(self.session, checkpoint_file)

        if reward_file_name is not None:
            npz_file_name = os.path.join(directory, reward_file_name)
            tf.logging.info('Attempting to load saved reward values from file %s', npz_file_name)
            npz_file = np.load(npz_file_name)

            self.rewards_batched = npz_file['train_rewards']
            self.music_theory_rewards_batched = npz_file['train_music_theory_rewards']
            self.note_rnn_rewards_batched = npz_file['train_note_rnn_rewards']
            self.eval_avg_reward = npz_file['eval_rewards']
            self.eval_avg_music_theory_reward = npz_file['eval_music_theory_rewards']
            self.eval_avg_note_rnn_reward = npz_file['eval_note_rnn_rewards']
            self.target_val_list = npz_file['target_val_list']
