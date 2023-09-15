import importlib.util
from typing import TYPE_CHECKING

from .utils import BestRun


if TYPE_CHECKING:
    from .trainer import SetFitTrainer


def is_optuna_available():
    return importlib.util.find_spec("optuna") is not None

def is_sigopt_available():
    return importlib.util.find_spec("sigopt") is not None

def default_hp_search_backend():
    if is_optuna_available():
        return "optuna"


def run_hp_search_optuna(trainer: "SetFitTrainer", n_trials: int, direction: str, **kwargs) -> BestRun:
    import optuna

    # Heavily inspired by transformers.integrations.run_hp_search_optuna
    # https://github.com/huggingface/transformers/blob/cbb8a37929c3860210f95c9ec99b8b84b8cf57a1/src/transformers/integrations.py#L160
    def _objective(trial):
        trainer.objective = None
        trainer.train(trial=trial)
        # If there hasn't been any evaluation during the training loop.
        if getattr(trainer, "objective", None) is None:
            metrics = trainer.evaluate()
            trainer.objective = trainer.compute_objective(metrics)
        return trainer.objective

    timeout = kwargs.pop("timeout", None)
    n_jobs = kwargs.pop("n_jobs", 1)
    study = optuna.create_study(direction=direction, **kwargs)
    study.optimize(_objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)
    best_trial = study.best_trial
    return BestRun(str(best_trial.number), best_trial.value, best_trial.params, study)



def run_hp_search_sigopt(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    import sigopt

    if importlib.metadata.version("sigopt") >= "8.0.0":
        sigopt.set_project("huggingface")

        experiment = sigopt.create_experiment(
            name="huggingface-tune",
            type="offline",
            parameters=trainer.hp_space(None),
            metrics=[{"name": "objective", "objective": direction, "strategy": "optimize"}],
            parallel_bandwidth=1,
            budget=n_trials,
        )

        #logger.info(f"created experiment: https://app.sigopt.com/experiment/{experiment.id}")

        for run in experiment.loop():
            with run:
                trainer.objective = None
                trainer.train(resume_from_checkpoint=None, trial=run.run)

                # If there hasn't been any evaluation during the training loop.
                if getattr(trainer, "objective", None) is None:
                    metrics = trainer.evaluate()
                    trainer.objective = trainer.compute_objective(metrics)
                run.log_metric("objective", trainer.objective)

        best = list(experiment.get_best_runs())[0]
        best_run = BestRun(best.id, best.values["objective"].value, best.assignments)
    else:
        from sigopt import Connection

        conn = Connection()
        proxies = kwargs.pop("proxies", None)
        if proxies is not None:
            conn.set_proxies(proxies)

        experiment = conn.experiments().create(
            name="huggingface-tune",
            parameters=trainer.hp_space(None),
            metrics=[{"name": "objective", "objective": direction, "strategy": "optimize"}],
            parallel_bandwidth=1,
            observation_budget=n_trials,
            project="huggingface",
        )

        while experiment.progress.observation_count < experiment.observation_budget:
            suggestion = conn.experiments(experiment.id).suggestions().create()
            trainer.objective = None
            trainer.train(resume_from_checkpoint=None, trial=suggestion)
            # If there hasn't been any evaluation during the training loop.
            if getattr(trainer, "objective", None) is None:
                metrics = trainer.evaluate()
                trainer.objective = trainer.compute_objective(metrics)

            values = [{"name": "objective", "value": trainer.objective}]
            obs = conn.experiments(experiment.id).observations().create(suggestion=suggestion.id, values=values)
            #logger.info(f"[suggestion_id, observation_id]: [{suggestion.id}, {obs.id}]")
            experiment = conn.experiments(experiment.id).fetch()

        best = list(conn.experiments(experiment.id).best_assignments().fetch().iterate_pages())[0]
        best_run = BestRun(best.id, best.value, best.assignments)
    return best_run
