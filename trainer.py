import argparse
from minigpt import mini_gpt_training_config
import train_utils


def main():
    training_config = mini_gpt_training_config()
    train_utils.basic_train_loop(training_config)


if __name__ == '__main__':
    main()
