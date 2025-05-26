from pipeline.optimization_workflow.optimization_manager import OptManager
from epde_struct_evaluator.epde_struct_evaluator import TrackEvaluator
import time

max_iter = 3
dir_name = 'burg_sindy'
start_iter = 0
refine_point = 100

debug = True # True False
print_exc = True
exit_code = True


if __name__ == '__main__':
    iter_num = 0

    t1 = time.time()
    opt_manager = OptManager(max_iter, start_iter, refine_point, dir_name, debug, print_exc, exit_code,
                             resample_shape=(20, 20), n_candidates=4)
    opt_manager.explore_solutions()
    pruned_track, by_project_track = opt_manager.call_pruner()

    t2 = time.time()

    te = TrackEvaluator(dir_name, opt_manager.eq_buffer.full_records_track, pruned_track, t2-t1, iter_num)
    te.evaluate()
    print()



