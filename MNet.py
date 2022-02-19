import sys, argparse

import MNet
import api as Server

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--train', help='Train model', action='store_true')
  parser.add_argument('-lr', '--learning-rate', help='Train learning rate')
  parser.add_argument('-e', '--epochs', help='Number of epochs to train')
  parser.add_argument('-b', '--batch', help='Batch size')
  parser.add_argument('-m', '--model', help='Model number')
  args = parser.parse_args()

  if args.learning_rate:
    MNet.learning_rate = float(args.learning_rate)

  if args.epochs:
    MNet.num_epochs = int(args.epochs)

  if args.batch:
    MNet.batch_size = int(args.batch)

  if args.train:
    MNet.run()
  elif args.model:
    Server.run(args.model)
  else:
    Server.run()