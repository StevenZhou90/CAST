from inference.eval_ import run_evaluation
import argparse
import glob
from sys import exit
import os

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='~/Data', type=str)
    parser.add_argument('--batch_size', default=1024, type=int)
    def comma_list(s): return s.split(',')
    parser.add_argument('--set_names', type=comma_list)
    args = parser.parse_args()

    # run evaluation module

    ### run iqa/fiqa eval
    sets = []
    # for d in ['facial_hair', 'no_facial_hair']:
    #     sets += list(glob.glob(f'all_sets/{d}/*_?.list'))
    for d in ['mustache', 'no_mustache']:
        sets += list(glob.glob(f'all_sets/{d}/*.list'))
    # sets = glob.glob('all_sets/race_gender_low_qual/*/*.list')
    # sets = glob.glob('all_sets/race_gender_no_softmax/*/*.list')
    # sets = glob.glob('all_sets/glasses/*/0.list')
    sets = glob.glob('all_sets/mf-sdd/*.list')
    # sets = glob.glob('all_sets/iqa_test_sets/*/1.list')
    # sets = glob.glob('all_sets/iqa_sets/magface*/1.list')
    # sets = list(sets) + list(glob.glob('all_sets/iqa_sets/magface*/2.list'))

    sets = glob.glob('CC11/*/*.list')
    # sets = sets[:20]



    sets = list(sets)
    assert len(sets) > 0, f'sets list empty'
    sets.sort(key=lambda x: os.path.basename(os.path.dirname(x)), reverse=False)
    print('number of sets', len(sets))
    # assert len(sets) == 500
    run_evaluation(args, sets, save_pth='facial_hair', data_size=2000, num_sets=10)
