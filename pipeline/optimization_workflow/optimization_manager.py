from pipeline.optimization_workflow.evaluator import Evaluator
from pipeline.buffer_handler.eq_buffer import EqBuffer
from pipeline.get_llm_response import get_response, get_debug_response
from pipeline.rebuild_prompt import rebuild_prompt
from pipeline.buffer_handler.eq_pruner import Pruner
from promptconstructor.array_to_txt import Data
from pipeline.clean_directories import clean_output_dir, reset_prompt_to_init, create_llm_iter_n_folders, delete_out_folders
from tqdm import tqdm
import sys
import traceback
import os


class OptManager:
    def __init__(self, max_iter, start_iter, refine_point=100, debug=True, print_exc=True,
                 exit_code=True, data_args=None, n_candidates=4, llm_iter=0):
        if not debug and llm_iter == 0:
            delete_out_folders()

        self.exit_code = exit_code
        self.debug = debug
        self.print_exc = print_exc
        self.refine_point = refine_point
        self.start_iter = start_iter
        self.max_iter = max_iter
        self.data_args = data_args
        self.llm_iter = llm_iter

        self.evaluator = Evaluator(data_args)
        self.eq_buffer = EqBuffer()
        self._pruner = None
        self._n_parent_cand = n_candidates

    def call_pruner(self):
        self._pruner = Pruner(self.eq_buffer, self.evaluator, self._n_parent_cand, self.data_args["dir_name"])
        pruned_track, by_project_track = self._pruner.cut_by_knee()
        return pruned_track, by_project_track

    def explore_solutions(self):
        self.step_0()
        for num in tqdm(range(self.start_iter, min(self.refine_point, self.max_iter)), desc="LLM's progress"):
            new_prompt, score, str_equation, params = self.step_n("continue-iter.txt", num)

    def refine_solutions(self):
        for num in tqdm(range(min(self.refine_point, self.start_iter), self.max_iter), desc="LLM's progress"):
            new_prompt, score, str_equation, params = self.step_n("continue-iter-refinement.txt", num)

    def step_0(self):
        if self.start_iter == 0:
            while True:
                try:
                    create_llm_iter_n_folders(self.llm_iter)
                    if not self.debug:
                        clean_output_dir(self.llm_iter)
                    reset_prompt_to_init(self.llm_iter)

                    new_prompt, score, str_equation, params = self.perform_step(file_name="zero-iter.txt", num=0)
                    self.start_iter = 1
                    break
                except Exception as e:
                    print('An exception occurred on iter #0:')
                    print(traceback.format_exc())
                    if self.exit_code: sys.exit()

    def step_n(self, path, num=1):
        try:
            new_prompt, score, str_equation, params = self.perform_step(path, num=num)
        except Exception as e:
            print(f"\nException occurred on iter #{num}:")
            if self.print_exc:
                # EOL while scanning string literal
                print(traceback.format_exc())
            if self.exit_code:
                sys.exit()
            return None, None, None, None
        return new_prompt, score, str_equation, params

    def perform_step(self, file_name, num):
        if self.debug:
            response = get_debug_response(llm_iter=self.llm_iter, num=num)
        else:
            response = get_response(file_name=file_name, llm_iter=self.llm_iter, num=num,
                                    dir_name=self.data_args["dir_name"],
                                    noise_level=self.data_args["noise_level"], print_info=False)
        score, eq_string, params = self.evaluator.llm_response_eval(response, self.eq_buffer)
        new_prompt, old_prompt = rebuild_prompt(eq_string, score, response, num=num,
                                                llm_iter=self.llm_iter, file_name=file_name)
        return new_prompt, score, eq_string, params
