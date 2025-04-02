from pipeline.optimization_workflow.optimization_manager import OptManager
from epde_integration.epde_search import EpdeSearcher
import numpy as np

max_iter = 6
dir_name = 'wave'
start_iter = 0
refine_point = 100

debug = True # True False
print_exc = True
exit_code = True


# проверить что data матрицы совпадают и их не надо .T

if __name__ == '__main__':
    opt_manager = OptManager(max_iter, start_iter, refine_point, dir_name, debug, print_exc, exit_code,
                             resample_shape=(20, 20), n_candidates=4)
    opt_manager.explore_solutions()

    pruned_track, _ = opt_manager.call_pruner()
    full_records_track = opt_manager.eq_buffer.full_records_track
    data = opt_manager.evaluator.data['inputs'] # "inputs": [raw_data['t'], raw_data['x'], raw_data['u']]

    epde_searcher = EpdeSearcher(data, full_records_track, pruned_track, dir_name, use_init_population=True)
    epde_searcher.fit()
    print()
