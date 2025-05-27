from epde_eq_parse.eq_parser import check_schema
from epde_eq_parse.schemas import schemas
from epde_struct_evaluator.structure_converter import StructConverter
from epde_eq_parse.eq_evaluator import EqEvaluator, EqInfo


class StructEvaluator(object):
    def __init__(self, dir_name, params, eq_str):
        left_side = schemas[dir_name]['left_side']
        struct_conv = StructConverter(eq_str, params, left_side)
        self.dir_name = dir_name
        self.terms_with_coeffs = struct_conv.convert()


    def evaluate(self, loss, runtime, iter_num):
        eq_eval = EqEvaluator(self.dir_name, self.terms_with_coeffs)
        mae = eq_eval.eval_mae(True)
        shd = eq_eval.eval_shd(False)
        return EqInfo(self.terms_with_coeffs, loss, mae, shd, runtime, iter_num)

class TrackEvaluator(object):
    def __init__(self, dir_name, records_track, pruned_track, runtime, iter_num):
        self.dir_name = dir_name
        self.records_track = records_track
        self.pruned_track = pruned_track
        self.runtime = runtime
        self.iter_num = iter_num

    def evaluate(self):
        run_eq_info = []
        for eq_key in self.pruned_track.keys():
            struct_conv = StructEvaluator(self.dir_name, self.records_track[eq_key].params, eq_key)
            eq_info = struct_conv.evaluate(self.records_track[eq_key].loss, self.runtime, self.iter_num)
            print()
        print()

