from pipeline.optimization_workflow.optimization_manager import OptManager
from epde_struct_evaluator.epde_struct_evaluator import TrackEvaluator
from epde_eq_parse.eq_evaluator import EqReranker
import time

max_llm_run = 2
max_iter = 5
dir_name = 'burg_sindy'
start_iter = 0
refine_point = 100

debug = True # True False
print_exc = True
exit_code = False


if __name__ == '__main__':
    pruned_eq_info_ls = []

    for iter_num in range(max_llm_run):
        t1 = time.time()
        opt_manager = OptManager(max_iter, start_iter, refine_point, dir_name, debug, print_exc, exit_code,
                                 resample_shape=(20, 20), n_candidates=4)
        opt_manager.explore_solutions()
        pruned_track, by_project_track = opt_manager.call_pruner()

        t2 = time.time()

        te = TrackEvaluator(dir_name, opt_manager.eq_buffer.full_records_track, pruned_track, t2-t1, iter_num)
        pruned_eq_infos = te.evaluate()
        pruned_eq_info_ls.append(pruned_eq_infos)

    eq_r = EqReranker(pruned_eq_info_ls, dir_name)
    eq_r.best_run_inf = pruned_eq_info_ls
    # eq_r.to_csv()
    print()



