import os
import shutil
import re


def clean_output_dir(out_path):
    os.makedirs(out_path, exist_ok=True)
    shutil.rmtree(out_path)


def merge_llm_iter_folders(source_dirs, out_path):
    clean_output_dir(out_path)
    os.makedirs(out_path, exist_ok=True)

    # Find the highest existing index in out_path
    existing_indices = []
    for item in os.listdir(out_path):
        if item.startswith("llm_iter_") and os.path.isdir(os.path.join(out_path, item)):
            try:
                index = int(item.split("_")[-1])
                existing_indices.append(index)
            except ValueError:
                continue
    next_index = max(existing_indices) + 1 if existing_indices else 0

    # Iterate through source directories and merge llm_iter_# folders
    for source_dir in source_dirs:
        files = [f for f in os.listdir(source_dir) if f.startswith("llm_iter_")]
        sorted_files = sorted(files, key=lambda x: int(re.search(r'\d+', x).group()))
        for item in sorted_files:
            if item.startswith("llm_iter_") and os.path.isdir(os.path.join(source_dir, item)):
                source_path = os.path.join(source_dir, item)
                new_name = f"llm_iter_{next_index}"
                dest_path = os.path.join(out_path, new_name)

                # Copy (or move) the folder with the new index
                shutil.copytree(source_path, dest_path)  # Use shutil.move() to move instead
                # print("Len source:", len(os.listdir(source_path)))
                print(f"Merged: {source_path} → {dest_path}")
                assert len(os.listdir(dest_path)) == len(os.listdir(source_path)), "The copied folder does not contain all original files"
                assert len(os.listdir(dest_path)) == 30, "The copied folder does not contain all llm_iter files"
                next_index += 1


dir_name = "kdv_sindy"
parent_path = f"D:\\Users\\Ivik12S\\Desktop\\llm_output\\{dir_name}"
listdir = os.listdir(parent_path)

source_dirs = []
for folder in os.listdir(parent_path):
    source_dirs.append(os.path.join(parent_path, folder))
out_path = f"llm_output_for_eval/{dir_name}"

merge_llm_iter_folders(source_dirs, out_path)
