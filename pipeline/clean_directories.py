import os
import sys
from pathlib import Path
import shutil
from pipeline.rebuild_prompt import extract_exp_buffer


PARENT_PATH = Path(os.path.dirname(__file__)).parent


def clean_output_dir(dir_path="debug_llm_outputs"):
    dir_path = os.path.join(PARENT_PATH, "pipeline", dir_path)
    if len(os.listdir(dir_path)) == 0:
        print(f"The directory {dir_path} is already empty")

    for file_path in os.listdir(dir_path):
        try:
            full_path = os.path.join(dir_path, file_path)
            os.remove(full_path)
        except FileNotFoundError:
            print(f"The directory {dir_path} doesn't exist")


# def reset_buffer_prompt(path):
#     переписать path используя PARENT_PATH
#     start_pos, end_pos, dict_str, file_content = extract_exp_buffer(path)
#     new_content = file_content[:start_pos] + file_content[end_pos:]
#     with open(path, 'w') as prompt:
#         prompt.write(new_content)


def reset_prompt_to_init(dir_path="prompts"):
    source = os.path.join(PARENT_PATH, "pipeline", dir_path, "reset-for-continue.txt")
    target = os.path.join(PARENT_PATH, "pipeline", dir_path, "continue-iter.txt")
    shutil.copyfile(source, target)

    refine_source = os.path.join(PARENT_PATH, "pipeline", dir_path, "reset-for-continue-refinement.txt")
    refine_target = os.path.join(PARENT_PATH, "pipeline", dir_path, "continue-iter-refinement.txt")
    shutil.copyfile(refine_source, refine_target)


if __name__ == "__main__":
    # clean_output_dir()
    reset_prompt_to_init()
