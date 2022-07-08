from eval.eval import run_evaluation
import argparse
import glob
from sys import exit
import os

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='~/Data', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    def comma_list(s): return s.split(',')
    parser.add_argument('--set_names', type=comma_list)
    args = parser.parse_args()

    # run evaluation module

    ### run iqa/fiqa eval
    sets = glob.glob('iqa_test_sets/*/*.list')
    sets = list(sets)
    sets.sort(key=lambda x: os.path.basename(os.path.dirname(x)), reverse=False)
    assert len(sets) == 500
    run_evaluation(args, sets[:1])
