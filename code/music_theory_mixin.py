import numpy as np
import tensorflow.compat.v1 as tf
from magenta.models.rl_tuner import rl_tuner_ops


NOTE_OFF = 0
NO_EVENT = 1


class MusicTheoryMixin:
    # References https://github.com/magenta/magenta/tree/main/magenta/models/rl_tuner

    def reward_music_theory(self, action):
        reward = self.reward_key(action)
        tf.logging.debug('Key: %s', reward)
        prev_reward = reward

        reward += self.reward_tonic(action)
        if reward != prev_reward:
            tf.logging.debug('Tonic: %s', reward)
        prev_reward = reward

        reward += self.reward_penalize_repeating(action)
        if reward != prev_reward:
            tf.logging.debug('Penalize repeating: %s', reward)
        prev_reward = reward

        reward += self.reward_penalize_autocorrelation(action)
        if reward != prev_reward:
            tf.logging.debug('Penalize autocorr: %s', reward)
        prev_reward = reward

        reward += self.reward_motif(action)
        if reward != prev_reward:
            tf.logging.debug('Reward motif: %s', reward)
        prev_reward = reward

        reward += self.reward_repeated_motif(action)
        if reward != prev_reward:
            tf.logging.debug('Reward repeated motif: %s', reward)
        prev_reward = reward

        # New rewards based on Gauldin's book, "A Practical Approach to Eighteenth
        # Century Counterpoint"
        reward += self.reward_preferred_intervals(action)
        if reward != prev_reward:
            tf.logging.debug('Reward preferred_intervals: %s', reward)
        prev_reward = reward

        reward += self.reward_leap_up_back(action)
        if reward != prev_reward:
            tf.logging.debug('Reward leap up back: %s', reward)
        prev_reward = reward

        reward += self.reward_high_low_unique(action)
        if reward != prev_reward:
            tf.logging.debug('Reward high low unique: %s', reward)

        return reward

    def random_reward_shift_to_mean(self, reward):
        s = np.random.randint(0, 2) * .1
        if reward > .5:
            reward -= s
        else:
            reward += s
        return reward

    def reward_scale(self, obs, action, scale=None):
        if scale is None:
            scale = rl_tuner_ops.C_MAJOR_SCALE

        obs = np.argmax(obs)
        action = np.argmax(action)
        reward = 0
        if action == 1:
            reward += .1
        if obs < action < obs + 3:
            reward += .05

        if action in scale:
            reward += .01
            if obs in scale:
                action_pos = scale.index(action)
                obs_pos = scale.index(obs)
                if obs_pos == len(scale) - 1 and action_pos == 0:
                    reward += .8
                elif action_pos == obs_pos + 1:
                    reward += .8

        return reward

    def reward_key_distribute_prob(self, action, key=None):
        if key is None:
            key = rl_tuner_ops.C_MAJOR_KEY

        reward = 0

        action_note = np.argmax(action)
        if action_note in key:
            num_notes_in_key = len(key)
            extra_prob = 1.0 / num_notes_in_key

            reward = extra_prob

        return reward

    def reward_key(self, action, penalty_amount=-1.0, key=None):
        if key is None:
            key = rl_tuner_ops.C_MAJOR_KEY

        reward = 0

        action_note = np.argmax(action)
        if action_note not in key:
            reward = penalty_amount

        return reward

    def reward_tonic(self, action, tonic_note=rl_tuner_ops.C_MAJOR_TONIC,
                                     reward_amount=3.0):
        action_note = np.argmax(action)
        first_note_of_final_bar = self.num_notes_in_melody - 4

        if self.beat == 0 or self.beat == first_note_of_final_bar:
            if action_note == tonic_note:
                return reward_amount
        elif self.beat == first_note_of_final_bar + 1:
            if action_note == NO_EVENT:
                return reward_amount
        elif self.beat > first_note_of_final_bar + 1:
            if action_note in (NO_EVENT, NOTE_OFF):
                return reward_amount
        return 0.0

    def reward_non_repeating(self, action):
        penalty = self.reward_penalize_repeating(action)
        if penalty >= 0:
            return .1

    def detect_repeating_notes(self, action_note):
        num_repeated = 0
        contains_held_notes = False
        contains_breaks = False

        # Note that the current action yas not yet been added to the composition
        for i in range(len(self.composition)-1, -1, -1):
            if self.composition[i] == action_note:
                num_repeated += 1
            elif self.composition[i] == NOTE_OFF:
                contains_breaks = True
            elif self.composition[i] == NO_EVENT:
                contains_held_notes = True
            else:
                break

        if action_note == NOTE_OFF and num_repeated > 1:
            return True
        elif not contains_held_notes and not contains_breaks:
            if num_repeated > 4:
                return True
        elif contains_held_notes or contains_breaks:
            if num_repeated > 6:
                return True
        else:
            if num_repeated > 8:
                return True

        return False

    def reward_penalize_repeating(self, action, penalty_amount=-100.0):
        action_note = np.argmax(action)
        is_repeating = self.detect_repeating_notes(action_note)
        if is_repeating:
            return penalty_amount
        else:
            return 0.0

    @classmethod
    def autocorrelate(cls, signal, lag=1):
        n = len(signal)
        x = np.asarray(signal) - np.mean(signal)
        c0 = np.var(signal)

        if c0 == 0:
            return 1
        return (x[lag:] * x[:n - lag]).sum() / float(n) / c0

    def reward_penalize_autocorrelation(self, action, penalty_weight=3.0):
        composition = self.composition + [np.argmax(action)]
        lags = [1, 2, 3]
        sum_penalty = 0
        for lag in lags:
            coeff = self.autocorrelate(composition, lag=lag)
            if not np.isnan(coeff):
                if np.abs(coeff) > 0.15:
                    sum_penalty += np.abs(coeff) * penalty_weight
        return -sum_penalty

    def detect_last_motif(self, composition=None, bar_length=8):
        if composition is None:
            composition = self.composition

        if len(composition) < bar_length:
            return None, 0

        last_bar = composition[-bar_length:]

        actual_notes = [a for a in last_bar if a not in (NO_EVENT, NOTE_OFF)]
        num_unique_notes = len(set(actual_notes))
        if num_unique_notes >= 3:
            return last_bar, num_unique_notes
        else:
            return None, num_unique_notes

    def reward_motif(self, action, reward_amount=3.0):
        composition = self.composition + [np.argmax(action)]
        motif, num_notes_in_motif = self.detect_last_motif(composition=composition)
        if motif is not None:
            motif_complexity_bonus = max((num_notes_in_motif - 3)*.3, 0)
            return reward_amount + motif_complexity_bonus
        else:
            return 0.0

    def detect_repeated_motif(self, action, bar_length=8):
        composition = self.composition + [np.argmax(action)]
        if len(composition) < bar_length:
            return False, None

        motif, _ = self.detect_last_motif(
                composition=composition, bar_length=bar_length)
        if motif is None:
            return False, None

        prev_composition = self.composition[:-(bar_length-1)]

        # Check if the motif is in the previous composition.
        for i in range(len(prev_composition) - len(motif) + 1):
            for j in range(len(motif)):
                if prev_composition[i + j] != motif[j]:
                    break
            else:
                return True, motif
        return False, None

    def reward_repeated_motif(self, action, bar_length=8, reward_amount=4.0):
        is_repeated, motif = self.detect_repeated_motif(action, bar_length)
        if is_repeated:
            actual_notes = [a for a in motif if a not in (NO_EVENT, NOTE_OFF)]
            num_notes_in_motif = len(set(actual_notes))
            motif_complexity_bonus = max(num_notes_in_motif - 3, 0)
            return reward_amount + motif_complexity_bonus
        else:
            return 0.0

    def detect_sequential_interval(self, action, key=None):
        if not self.composition:
            return 0, None, None

        prev_note = self.composition[-1]
        action_note = np.argmax(action)

        c_major = False
        if key is None:
            key = rl_tuner_ops.C_MAJOR_KEY
            c_notes = [2, 14, 26]
            g_notes = [9, 21, 33]
            e_notes = [6, 18, 30]
            c_major = True
            tonic_notes = [2, 14, 26]
            fifth_notes = [9, 21, 33]

        # get rid of non-notes in prev_note
        prev_note_index = len(self.composition) - 1
        while prev_note in (NO_EVENT, NOTE_OFF) and prev_note_index >= 0:
            prev_note = self.composition[prev_note_index]
            prev_note_index -= 1
        if prev_note in (NOTE_OFF, NO_EVENT):
            tf.logging.debug('Action_note: %s, prev_note: %s', action_note, prev_note)
            return 0, action_note, prev_note

        tf.logging.debug('Action_note: %s, prev_note: %s', action_note, prev_note)

        # get rid of non-notes in action_note
        if action_note == NO_EVENT:
            if prev_note in tonic_notes or prev_note in fifth_notes:
                return (rl_tuner_ops.HOLD_INTERVAL_AFTER_THIRD_OR_FIFTH,
                                action_note, prev_note)
            else:
                return rl_tuner_ops.HOLD_INTERVAL, action_note, prev_note
        elif action_note == NOTE_OFF:
            if prev_note in tonic_notes or prev_note in fifth_notes:
                return (rl_tuner_ops.REST_INTERVAL_AFTER_THIRD_OR_FIFTH,
                                action_note, prev_note)
            else:
                return rl_tuner_ops.REST_INTERVAL, action_note, prev_note

        interval = abs(action_note - prev_note)

        if c_major and interval == rl_tuner_ops.FIFTH and (
                prev_note in c_notes or prev_note in g_notes):
            return rl_tuner_ops.IN_KEY_FIFTH, action_note, prev_note
        if c_major and interval == rl_tuner_ops.THIRD and (
                prev_note in c_notes or prev_note in e_notes):
            return rl_tuner_ops.IN_KEY_THIRD, action_note, prev_note

        return interval, action_note, prev_note

    def reward_preferred_intervals(self, action, scaler=5.0, key=None):
        interval, _, _ = self.detect_sequential_interval(action, key)
        tf.logging.debug('Interval:', interval)

        if interval == 0:  # either no interval or involving uninteresting rests
            tf.logging.debug('No interval or uninteresting.')
            return 0.0

        reward = 0.0

        # rests can be good
        if interval == rl_tuner_ops.REST_INTERVAL:
            reward = 0.05
            tf.logging.debug('Rest interval.')
        if interval == rl_tuner_ops.HOLD_INTERVAL:
            reward = 0.075
        if interval == rl_tuner_ops.REST_INTERVAL_AFTER_THIRD_OR_FIFTH:
            reward = 0.15
            tf.logging.debug('Rest interval after 1st or 5th.')
        if interval == rl_tuner_ops.HOLD_INTERVAL_AFTER_THIRD_OR_FIFTH:
            reward = 0.3

        # large leaps and awkward intervals bad
        if interval == rl_tuner_ops.SEVENTH:
            reward = -0.3
            tf.logging.debug('7th')
        if interval > rl_tuner_ops.OCTAVE:
            reward = -1.0
            tf.logging.debug('More than octave.')

        # common major intervals are good
        if interval == rl_tuner_ops.IN_KEY_FIFTH:
            reward = 0.1
            tf.logging.debug('In key 5th')
        if interval == rl_tuner_ops.IN_KEY_THIRD:
            reward = 0.15
            tf.logging.debug('In key 3rd')

        # smaller steps are generally preferred
        if interval == rl_tuner_ops.THIRD:
            reward = 0.09
            tf.logging.debug('3rd')
        if interval == rl_tuner_ops.SECOND:
            reward = 0.08
            tf.logging.debug('2nd')
        if interval == rl_tuner_ops.FOURTH:
            reward = 0.07
            tf.logging.debug('4th')

        # larger leaps not as good, especially if not in key
        if interval == rl_tuner_ops.SIXTH:
            reward = 0.05
            tf.logging.debug('6th')
        if interval == rl_tuner_ops.FIFTH:
            reward = 0.02
            tf.logging.debug('5th')

        tf.logging.debug('Interval reward', reward * scaler)
        return reward * scaler

    def detect_high_unique(self, composition):
        max_note = max(composition)
        return list(composition).count(max_note) == 1

    def detect_low_unique(self, composition):
        no_special_events = [x for x in composition
                                                 if x not in (NO_EVENT, NOTE_OFF)]
        if no_special_events:
            min_note = min(no_special_events)
            if list(composition).count(min_note) == 1:
                return True
        return False

    def reward_high_low_unique(self, action, reward_amount=3.0):
        if len(self.composition) + 1 != self.num_notes_in_melody:
            return 0.0

        composition = np.array(self.composition)
        composition = np.append(composition, np.argmax(action))

        reward = 0.0

        if self.detect_high_unique(composition):
            reward += reward_amount

        if self.detect_low_unique(composition):
            reward += reward_amount

        return reward

    def detect_leap_up_back(self, action, steps_between_leaps=6):
        if not self.composition:
            return 0

        outcome = 0

        interval, action_note, prev_note = self.detect_sequential_interval(action)

        if action_note in (NOTE_OFF, NO_EVENT):
            self.steps_since_last_leap += 1
            tf.logging.debug('Rest, adding to steps since last leap. It is'
                                             'now: %s', self.steps_since_last_leap)
            return 0

        # detect if leap
        if interval >= rl_tuner_ops.FIFTH or interval == rl_tuner_ops.IN_KEY_FIFTH:
            if action_note > prev_note:
                leap_direction = rl_tuner_ops.ASCENDING
                tf.logging.debug('Detected an ascending leap')
            else:
                leap_direction = rl_tuner_ops.DESCENDING
                tf.logging.debug('Detected a descending leap')

            # there was already an unresolved leap
            if self.composition_direction != 0:
                if self.composition_direction != leap_direction:
                    tf.logging.debug('Detected a resolved leap')
                    tf.logging.debug('Num steps since last leap: %s',
                                                     self.steps_since_last_leap)
                    if self.steps_since_last_leap > steps_between_leaps:
                        outcome = rl_tuner_ops.LEAP_RESOLVED
                        tf.logging.debug('Sufficient steps before leap resolved, '
                                                         'awarding bonus')
                    self.composition_direction = 0
                    self.leapt_from = None
                else:
                    tf.logging.debug('Detected a double leap')
                    outcome = rl_tuner_ops.LEAP_DOUBLED

            # the composition had no previous leaps
            else:
                tf.logging.debug('There was no previous leap direction')
                self.composition_direction = leap_direction
                self.leapt_from = prev_note

            self.steps_since_last_leap = 0

        # there is no leap
        else:
            self.steps_since_last_leap += 1
            tf.logging.debug('No leap, adding to steps since last leap. '
                                             'It is now: %s', self.steps_since_last_leap)

            # If there was a leap before, check if composition has gradually returned
            # This could be changed by requiring you to only go a 5th back in the
            # opposite direction of the leap.
            if (self.composition_direction == rl_tuner_ops.ASCENDING and
                    action_note <= self.leapt_from) or (
                            self.composition_direction == rl_tuner_ops.DESCENDING and
                            action_note >= self.leapt_from):
                tf.logging.debug('detected a gradually resolved leap')
                outcome = rl_tuner_ops.LEAP_RESOLVED
                self.composition_direction = 0
                self.leapt_from = None

        return outcome

    def reward_leap_up_back(self, action, resolving_leap_bonus=5.0,
                                                    leaping_twice_punishment=-5.0):
        leap_outcome = self.detect_leap_up_back(action)
        if leap_outcome == rl_tuner_ops.LEAP_RESOLVED:
            tf.logging.debug('Leap resolved, awarding %s', resolving_leap_bonus)
            return resolving_leap_bonus
        elif leap_outcome == rl_tuner_ops.LEAP_DOUBLED:
            tf.logging.debug('Leap doubled, awarding %s', leaping_twice_punishment)
            return leaping_twice_punishment
        else:
            return 0.0
