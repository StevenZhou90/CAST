from eval.eval import run_evaluation
import argparse

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='~/Data', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    def comma_list(s): return s.split(',')
    parser.add_argument('--set_names', type=comma_list)
    args = parser.parse_args()

    # run evaluation module
    run_evaluation(args)
