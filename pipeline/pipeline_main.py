from pipeline.optimization_workflow.optimization_manager import OptManager
from epde_struct_evaluator.epde_struct_evaluator import TrackEvaluator
from epde_eq_parse.eq_evaluator import EqReranker
import time
import traceback
import re

max_llm_run = 1
max_iter = 3
start_iter = 0
refine_point = 100

data_args = {"resample_shape": (20, 20),
             "use_cached": True,
             "noise_level": 0,
             "dir_name": "burg_sindy"}

debug = True # True False
print_exc = True
exit_code = False


if __name__ == '__main__':
    best_eq_info_ls = []

    for llm_iter_num in range(max_llm_run):
        t1 = time.time()
        opt_manager = OptManager(max_iter, start_iter, refine_point, debug, print_exc, exit_code,
                                 data_args, n_candidates=4, llm_iter=llm_iter_num)
        opt_manager.explore_solutions()
        try:
            pruned_track, by_project_track = opt_manager.call_pruner()
        except Exception as e:
            print(f"\nException occurred during pruning stage on llm_iter #{llm_iter_num}:")
            print(traceback.format_exc())

        t2 = time.time()

        te = TrackEvaluator(data_args["dir_name"], opt_manager.eq_buffer.full_records_track, pruned_track, t2-t1, llm_iter_num)
        try:
            best_eq_info = te.evaluate("shd")
            best_eq_info_ls.append(best_eq_info)
        except Exception as e:
            print(f"\nException occurred during evaluation on llm_iter #{llm_iter_num}:")
            print(traceback.format_exc())

    # тут костыль чтобы просто записать все в csv
    eq_r = EqReranker(best_eq_info_ls, data_args["dir_name"])
    eq_r.best_run_inf = best_eq_info_ls
    eq_r.to_csv(package="pipeline", experiment_info=f"noise_level_{data_args['noise_level']}")
    print()



