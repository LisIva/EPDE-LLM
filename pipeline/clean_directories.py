import os
import sys
from pathlib import Path
import shutil
# from pipeline.rebuild_prompt import extract_exp_buffer
import re


PARENT_PATH = Path(os.path.dirname(__file__)).parent


def clean_output_dir(llm_iter=0, dir_path="debug_llm_outputs"):
    dir_path = os.path.join(PARENT_PATH, "pipeline", dir_path, f"llm_iter_{llm_iter}")
    for file_path in os.listdir(dir_path):
        try:
            full_path = os.path.join(dir_path, file_path)
            os.remove(full_path)
        except FileNotFoundError:
            print(f"The directory {dir_path} doesn't exist")


def delete_out_folders(dir_paths=None):
    if dir_paths is None:
        dir_paths = ["prompts", "debug_llm_outputs"]

    for dir_name in dir_paths:
        parent_dir = os.path.join(PARENT_PATH, "pipeline", dir_name)
        for folder_name in os.listdir(parent_dir):
            folder_path = os.path.join(parent_dir, folder_name)
            if os.path.isdir(folder_path) and re.match(r"^llm_iter_\d+$", folder_name):
                try:
                    shutil.rmtree(folder_path)
                except Exception as e:
                    print(f"Error deleting {folder_path}: {e}")

# def reset_buffer_prompt(path):
#     переписать path используя PARENT_PATH
#     start_pos, end_pos, dict_str, file_content = extract_exp_buffer(path)
#     new_content = file_content[:start_pos] + file_content[end_pos:]
#     with open(path, 'w') as prompt:
#         prompt.write(new_content)


def create_llm_iter_n_folders(llm_iter=0, dir_paths=None):
    if dir_paths is None:
        dir_paths = ["prompts", "debug_llm_outputs"]

    for path in dir_paths:
        path = os.path.join(PARENT_PATH, "pipeline", path, f"llm_iter_{llm_iter}")
        os.makedirs(path, exist_ok=True)


def reset_prompt_to_init(llm_iter=0, dir_path="prompts"):
    source = os.path.join(PARENT_PATH, "pipeline", dir_path, "reset-for-continue.txt")
    target = os.path.join(PARENT_PATH, "pipeline", dir_path, f"llm_iter_{llm_iter}", "continue-iter.txt")
    shutil.copyfile(source, target)

    refine_source = os.path.join(PARENT_PATH, "pipeline", dir_path, "reset-for-continue-refinement.txt")
    refine_target = os.path.join(PARENT_PATH, "pipeline", dir_path, f"llm_iter_{llm_iter}", "continue-iter-refinement.txt")
    shutil.copyfile(refine_source, refine_target)


if __name__ == "__main__":
    # clean_output_dir()
    # reset_prompt_to_init()
    delete_out_folders()
