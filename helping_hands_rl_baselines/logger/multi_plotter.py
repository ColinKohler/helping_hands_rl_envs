'''
.. moduleauthor: Colin Kohler <github.com/ColinKohler>
'''

import os
import pickle
import numpy as np
import more_itertools
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

class MultiPlotter(object):
  '''
  Plotting utility used to average multiple runs.
  '''
  def __init__(self, log_filepaths, log_names):
    self.logs = self.loadLogs(log_filepaths, log_names)

  def loadLogs(self, filepaths, names):
    '''

    '''
    logs = dict()

    for n, f in zip(names, filepaths):
      runs = list()
      for fp in f:
        if os.path.exists(fp):
          with open(fp, 'rb') as f:
            runs.append(pickle.load(f))
        else:
          print('No log found at {}'.format(fp))
      logs[n] = runs

    return logs

  def plotLearningCurves(self, title, filepath, max_eps, window=100):
    '''
    Plot mulitple learning curves on a single plot.

    Args:
    '''

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Avg. Reward')

    for log_name, log in self.logs.items():
      sr = list()
      for run in log:
        eps_rewards = run['training_eps_rewards'][:max_eps]
        avg_reward = np.mean(list(more_itertools.windowed(eps_rewards, window)), axis=1)

        if avg_reward.shape[0] < (max_eps - window + 1):
          n = avg_reward.shape[0]
          avg_reward = np.pad(avg_reward, (0, (max_eps - window + 1) - n), 'edge')
        sr.append(avg_reward)

      sr = np.array(sr)
      x = np.arange(sr.shape[1])

      sr_mean = np.mean(sr, axis=0)
      sr_std = np.std(sr, axis=0) / np.sqrt(len(log))
      ax.plot(x, sr_mean.squeeze(), label=log_name)
      ax.fill_between(x, sr_mean.squeeze() - sr_std, sr_mean.squeeze() + sr_std, alpha=0.5)

    ax.legend()
    plt.savefig(filepath)
    plt.close()

  def plotEvalRewards(self, title, filepath, num_eval_intervals, window=1):
    '''
    Plot mulitple evaluation curves on a single plot.

    Args:
    '''

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Avg. Reward')

    for log_name, log in self.logs.items():
      sr = list()
      for run in log:
        eval_rewards = run['eval_eps_rewards'][:num_eval_intervals]
        eval_rewards = [np.mean(eps) for eps in eval_rewards]
        avg_eval_rewards = np.mean(list(more_itertools.windowed(eval_rewards, window)), axis=1)

        if avg_eval_rewards.shape[0] < (num_eval_intervals - window + 1):
          n = avg_eval_rewards.shape[0]
          avg_eval_rewards = np.pad(avg_eval_rewards, (0, (num_eval_intervals - window + 1) - n), 'edge')
        sr.append(avg_eval_rewards)

      sr = np.array(sr)
      x = np.arange(sr.shape[1]) * 500

      sr_mean = np.mean(sr, axis=0)
      sr_std = np.std(sr, axis=0) / np.sqrt(len(log))
      ax.plot(x, sr_mean.squeeze(), label=log_name)
      ax.fill_between(x, sr_mean.squeeze() - sr_std, sr_mean.squeeze() + sr_std, alpha=0.5)

    ax.legend()
    plt.savefig(filepath)
    plt.close()
