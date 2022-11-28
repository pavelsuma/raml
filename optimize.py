import os
import pickle
from pathlib import Path
import optuna

from logger import get_logger
from modelhelpers import load_model
from train import run_train, get_train_splits

PARAMS = {
    "lr"        : [1e-4, 2e-4],
    "lam_1"     : [0.6, 0.8],
    "lam_2"     : [0.6, 0.8],
}
TRIALS = 6

def objective(args, datasets, t_model, trial):
    args.lr = trial.suggest_float("lr", *PARAMS["lr"], log=True)
    args.lam_1 = trial.suggest_float("lam_1", *PARAMS["lam_1"])
    args.lam_2 = trial.suggest_float("lam_2", *PARAMS["lam_2"])

    logger = get_logger(args.logger, Path(args.directory) / 'tensorboard' / str(trial.number), str(trial.number))
    logger.log_text({'params': trial.params})
    logger.log_scalars(trial.params)

    s_model = load_model(args.data_root, args, args.student, args.student_path).to('cuda:0')
    mAP, _ = run_train(args, datasets, s_model, t_model, logger, trial)
    return mAP

class TestTrialCallback:
    def __init__(self, logger):
        self.logger = logger

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        self.logger.set_epoch(self.logger.epoch + 1)
        self.logger.log_scalars({'opt/mAP_val_asym': trial.value})

def optimize(args, cfgs, t_model, feats, logger):
    test_callback = TestTrialCallback(logger)
    # TODO: refactor to avoid loading model just for mean and std of input images
    if t_model is None:
        net_meta = load_model(args.data_root, args, args.student, args.student_path).to('cuda:0').meta
    else:
        net_meta = t_model.meta
    train_split = get_train_splits(args, cfgs, net_meta, feats)
    study = optuna.create_study(direction="maximize", study_name=logger.exp_name,
                                pruner=optuna.pruners.NopPruner())
    study.enqueue_trial({"lr": args.lr, "lam_1": args.lam_1, "lam_2": args.lam_2})
    study.optimize(lambda trial: objective(args, train_split, t_model, trial), n_trials=TRIALS, callbacks=[test_callback])

    print_study_stats(study, logger)
    with open(os.path.join(args.directory, 'study.pkl'), 'wb') as handle:
        pickle.dump(study, handle)

    args.lr = study.best_trial.params['lr']
    args.lam_1 = study.best_trial.params['lam_1']
    args.lam_2 = study.best_trial.params['lam_2']

def print_study_stats(study, logger):
    pruned_trials = study.get_trials(states=(optuna.trial.TrialState.PRUNED,))
    complete_trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))

    print(f">> Study statistics: \n \
              Number of finished trials: {len(study.trials)} \n \
              Number of pruned trials: {len(pruned_trials)} \n \
              Number of complete trials: {len(complete_trials)}")

    trial = study.best_trial
    print(f"Best trial: \
                Value: {trial.value} \
                Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    logger.log_text({'Best parameters': trial.params})
    logger.log_text({'Best trial': trial.number})
