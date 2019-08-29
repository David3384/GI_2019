import numpy as np
import pandas as pd
from scipy import stats
import sys

# given a numpy array that encodes the result
# print the classification statistics
def print_results_from_np(result_np):
    print(np2df(result_np))

# print the results from a directory
def print_results_from_dir(dir_name, include_fold=False):
    print('--------------------------')
    print('directory name %s' % dir_name)

    # load results
    fold_result = np.loadtxt('../experiments/' + dir_name + '/result_by_fold.np')
    fold_result = np.reshape(fold_result, (-1, 4, 3))

    # if include fold, then print out the result for each fold
    if include_fold:
        for idx in range(len(fold_result)):
            print('results for fold %d.' % (idx + 1))
            print_results_from_np(fold_result[idx])

    # mean
    print('Mean for each entry')
    print_results_from_np(np.mean(fold_result, axis=0))

    # standard deviation
    print('Standard deviation for each entry')
    print_results_from_np(np.std(fold_result, axis=0))

    print('--------------------------')

# mapping a 4 * 3 numpy array to dataframe
# with attributes being precision, recall, f1, support
def np2df(result_np):
    result_np = result_np.T
    d = [{'precision': r[0], 'recall': r[1], 'f-score': r[2], 'support': r[3]} for r in result_np]
    df = pd.DataFrame(d)
    return df

# get the macro f1 of different folds
def get_fold_macro_f(dir_name):
    fold_result = np.loadtxt(dir_name + '/result_by_fold.np')
    fold_result = np.reshape(fold_result, (-1, 4, 3)) #shape (5, 4, 3), 5 folds, 4 metrics, 3 categories
    macrof1 = np.mean(fold_result[:, 2, :], axis=-1) #shape (5,)
    return macrof1


# compare the results from two directories
# conduct statistical significance test check whether improvement is significant
def compare_dirs(dir1, dir2):
    # print the results from the first and second directory
    print_results_from_dir(dir1)
    print_results_from_dir(dir2)

    # retrive the f-scores from the directories
    mf1, mf2 = get_fold_macro_f(dir1), get_fold_macro_f(dir2)

    # statistical significance test, paired t-test
    alpha = stats.ttest_rel(mf1, mf2)

    # print the results
    print('macro-f1 score achieved by each fold for %s' % dir1)
    print('macro-f1 score achieved by each fold for %s' % dir2)
    print('%s averaged macro-f1 score = %.3f' % (dir1, np.mean(mf1)))
    print('%s averaged macro-f1 score = %.3f' % (dir2, np.mean(mf2)))
    print(alpha)

    # return the statistical significance
    return alpha

# print the result from a list of directories
def print_dirs(dirs):
    print('----------')
    for d in dirs:
        print('directory name %s, results across 5 fold as follows: ' % d)
        print(get_fold_macro_f(d))
        print(np.mean(get_fold_macro_f(d))) #average f-score of 5 folds

# you can directly run this file through terminal by
# "python3 read_results.py <list of directories that you want to print the results>"
if __name__ == '__main__':
    print_dirs(sys.argv[1:])
    #compare_dirs(sys.argv[1], sys.argv[2])

