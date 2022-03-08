'''
.. moduleauthor: Colin Kohler <github.com/ColinKohler>
'''

import numpy as np
import more_itertools
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

class Plotter(object):
  '''
  Plotting utility.
  '''
  def __init__(self, log_filepaths, log_names):
    self.logs = self.loadLogs(log_filepaths, log_names)

  def loadLogs(self, filepaths, names):
    '''

    '''
    logs = dict()

    for n, fp in zip(names, filepaths):
      if os.path.exists(fp):
        with open(fp, 'rb') as f:
          logs[n] = append(pickle.load(f))
      else:
        print('No log found at {}'.format(f))

    return logs

  def plotLearningCurves(self, title, filepath, window=100):
    '''
    Plot mulitple learning curves on a single plot.

    Args:
    '''

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Avg. Reward')
    ax.legend()

    for log_name, log in self.logs.values():
      eps_rewards = log['training_eps_rewards']
      avg_reward = np.mean(list(more_itertools.windowed(eps_rewards, window)), axis=1)
      xs = np.arange(window, len(avg_reward) + window)
      ax.plot(xs, avg_reward, label=log_name)

    plt.savefig(filepath)
    plt.close()

  def plotLearningCurve(self, name, title, filepath, window=100):
    '''
    Plot the learning curve for the given episode rewards.

    Args:
    '''
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Avg. Reward')

    eps_reward = self.logs[name]['training_eps_rewards']
    avg_reward = np.mean(list(more_itertools.windowed(eps_rewards, window)), axis=1)
    xs = np.arange(window, len(avg_reward) + window)
    ax.plot(xs, avg_reward)

    plt.savefig(filepath)
    plt.close()
