import argparse

from helping_hands_rl_baselines.logger.multi_plotter import MultiPlotter
from helping_hands_rl_baselines.scripts.log_filepaths import log_filepaths, log_plot_names

if __name__ == '__main__':
  parser =  argparse.ArgumentParser()
  parser.add_argument('log', type=str,
    help='Log to plot')
  parser.add_argument('--lc_eps', type=int, default=5000,
    help='Number of episodes to plot on the learning curve')
  parser.add_argument('--lc_window', type=int, default=100,
    help='Window size for learning curve averaging.')
  parser.add_argument('--eval_intervals', type=int, default=40,
    help='Number of intervals to plot on the evaluation curve')
  parser.add_argument('--eval_window', type=int, default=5,
    help='Window size for evaluation curve averaging.')
  args = parser.parse_args()

  log_filepath = log_filepaths[args.log]
  plot_name = log_plot_names[args.log]
  log_names = ['vison+force', 'vision']

  base_dir = 'helping_hands_rl_baselines/scripts/outputs/'
  plotter = MultiPlotter(log_filepath, log_names)
  plotter.plotLearningCurves(plot_name, base_dir + 'train.pdf', args.lc_eps, window=args.lc_window)
  plotter.plotEvalRewards(plot_name, base_dir + 'eval_rewards.pdf', args.eval_intervals, window=args.eval_window)
  plotter.plotEvalValues(plot_name, base_dir +'eval_values.pdf', args.eval_intervals, window=args.eval_window)
