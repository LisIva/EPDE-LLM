from epde_eq_parse.eq_parser import check_schema
from epde_eq_parse.schemas import schemas
from epde_struct_evaluator.structure_converter import StructConverter
from epde_eq_parse.eq_evaluator import EqEvaluator, EqInfo, FrontReranker


class StructEvaluator(object):
    def __init__(self, dir_name, params, eq_str):
        left_side = schemas[dir_name]['left_side']
        struct_conv = StructConverter(eq_str, params, left_side)
        self.dir_name = dir_name
        self.terms_with_coeffs = struct_conv.convert()

    def evaluate(self, loss, runtime, iter_num):
        eq_eval = EqEvaluator(self.dir_name, self.terms_with_coeffs)
        mae = eq_eval.eval_mae(False)
        mae_norm = eq_eval.eval_mae_norm()
        shd = eq_eval.eval_shd()
        return EqInfo(self.terms_with_coeffs, loss, mae, mae_norm, shd, runtime, iter_num, eq_eval.is_correct_schema)


class TrackEvaluator(object):
    def __init__(self, dir_name, records_track, pruned_track, runtime, iter_num):
        self.dir_name = dir_name
        self.records_track = records_track
        self.pruned_track = pruned_track
        self.runtime = runtime
        self.iter_num = iter_num

    def evaluate(self, metric="shd"):
        run_eq_info = []
        for eq_key in self.pruned_track.keys():
            struct_conv = StructEvaluator(self.dir_name, self.records_track[eq_key].params, eq_key)

            eq_info = struct_conv.evaluate(self.records_track[eq_key].loss, self.runtime, self.iter_num)
            run_eq_info.append(eq_info)

        # return run_eq_info if len(run_eq_info) != 0 else None
        if len(run_eq_info) != 0:
            run_eq_info_filtered = [eq_info for eq_info in run_eq_info if eq_info.mae is not None]
            if len(run_eq_info_filtered) == 0:
                return EqInfo(None, None, None, None, None, None, None)
            front_r = FrontReranker(run_eq_info_filtered)
            return front_r.select_best(metric)
        else:
            print('An empty list of equations was received from the LLM')
            return None

# def filter_eq_info(run_eq_info: list[EqInfo]):
#     run_eq_info_filtered = []
#     for eq_info in run_eq_info:
#         if eq_info.mae is not None:
#             run_eq_info_filtered.append(eq_info)
#     return run_eq_info_filtered
