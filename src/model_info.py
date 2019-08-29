import numpy as np

def get_model_info(num_runs, num_folds, num_models):
    discrete_input = set([1, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    pad_elmo = set([3, 4, 5])
    return {
        'model_ids': model_ids(),
        'discrete_input': discrete_input,
        'pad_elmo': pad_elmo,
        'thresholds': read_all_model_threshold(num_runs, num_folds, num_models)
    }

# model_id to experiment directory mapping is defined here
def model_ids():
    model_id2experiment_dir = {}
    model_id2experiment_dir[1] = "../experiments/CNN+DSWEMB+CF_run_"
    model_id2experiment_dir[2] = "../experiments/CNN+TwitterWEMB+CF_run_"
    model_id2experiment_dir[3] = "../experiments/CNN+DSWEMB+CF+DSELMo_run_"
    model_id2experiment_dir[4] = "../experiments/CNN+DSWEMB+DSELMo_run_"
    model_id2experiment_dir[5] = "../experiments/CNN+DSWEMB+CF+NonDSELMo_run_"
    model_id2experiment_dir[6] = "../experiments/LSTMAttn+DSWEMB+DSELMo+CF_run_"
    model_id2experiment_dir[7] = "../experiments/LSTMAttn+DSWEMB+DSELMo+CF+RTL_run_"
    model_id2experiment_dir[8] = "../experiments/LSTMAttn+DSWEMB+CF_run_"
    model_id2experiment_dir[9] = "../experiments/LSTMAttn+DSWEMB+CF+RTL_run_"
    model_id2experiment_dir[10] = "../experiments/CNN+DSWEMB+CF+KR_run_"
    model_id2experiment_dir[11] = "../experiments/CNN+DSWEMB+CF+CY_run_"
    model_id2experiment_dir[12] = "../experiments/CNN+DSWEMB+CF+DROP_run_"
    model_id2experiment_dir[13] = "../experiments/LSTMAttn+DSWEMB+CF+KR_run_"
    model_id2experiment_dir[14] = "../experiments/LSTMAttn+DSWEMB+CF+CY_run_"
    model_id2experiment_dir[15] = "../experiments/LSTMAttn+DSWEMB+CF+DROP_run_"
    model_id2experiment_dir[16] = "../experiments/LSTMAttn+DSWEMB_run_"
    model_id2experiment_dir[17] = "../experiments/LSTMAttn+DSWEMB+RTL_run_"
    model_id2experiment_dir[18] = "../experiments/LSTMAttn+DSWEMB+CFLAST_run_"
    model_id2experiment_dir[19] = "../experiments/LSTMAttn+DSWEMB+CFLAST+RTL_run_"
    return model_id2experiment_dir


# reading the experiment threshold from
def read_experiment_threshold(experiment_dir, num_folds=5):
    # read the readme file where the threshold is written
    with open(experiment_dir + '/README', 'r') as f:
        model_thresholds = [float(l.strip().split(' ')[-1]) for l in f if 'threshold' in l][:2 * num_folds]

    # casting as numpy, reshape, and return
    model_thresholds = np.array(model_thresholds).reshape(num_folds, 2)
    agg_thresholds, loss_thresholds = model_thresholds[:, 0], model_thresholds[:, 1]
    return agg_thresholds, loss_thresholds


# return all of the model thresholds
def read_all_model_threshold(num_runs, num_folds, num_models):
    model_id2experiment_dir = model_ids()

    thresholds = {}
    for model_id in range(1, num_models + 1):
        for run_idx in range(num_runs):
            thresholds[(model_id, run_idx)] = read_experiment_threshold(model_id2experiment_dir[model_id] + str(run_idx),
                                                                        num_folds)
    return thresholds