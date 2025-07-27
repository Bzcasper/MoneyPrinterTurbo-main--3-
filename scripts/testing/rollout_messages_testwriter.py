import json
import os
import gc
from collections import OrderedDict
import logging
import argparse
from termcolor import cprint
import re
from collections import defaultdict
import ast
import copy
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import libcst as cst
import libcst.matchers as m
from libcst.display import dump
from datasets import load_dataset

from kimidev.agentlessnano.model_api import make_model
from kimidev.agentlessnano.utils import *
from kimidev.agentlessnano.testwritter_utils import remove_test_cases
from kimidev.agentlessnano.post_process import generate_model_patch, generate_model_patch_difflib_testwritter

### llm response
def relevant_file_prompt_response(llm_model, problem_statement, structure):
    """
    Generate the prompt for the relevant file selection.
    """

    obtain_relevant_files_prompt = """
Please look through the following GitHub problem description and Repository structure and provide a list of test files that should be run after applying the patch to fix the issue.

### GitHub Problem Description ###
{problem_statement}

###

### Repository Structure ###
{structure}

###

Please only provide the full path and return at most 5 files.
The returned files should be separated by new lines ordered by most to least important and wrapped with ```
For example:
```
file1.py
file2.py
```
    """
    prompt_content = obtain_relevant_files_prompt.format(
            problem_statement=problem_statement,
            structure=show_project_structure(structure, test_flag=True).strip(),
        ).strip()
    
    file_level_message, raw_answer = llm_chat(llm_model, prompt_content)
    model_found_files = raw_answer.strip().split("\n")

    files, classes, functions = get_full_file_paths_and_classes_and_functions(structure)

    model_found_files = [correct_file_path_in_structure(file, structure) for file in model_found_files]

    # sort based on order of appearance in model_found_files
    found_files = correct_file_paths(model_found_files, files)

    
    return file_level_message, found_files

def repair_prompt_response(llm_model, problem_statement, structure, found_files):
    repair_relevant_file_instruction = """
Below are some code segments, each from a relevant test file. One or more of these files may be added some new tests which can reproduce the issue .
    """
    repair_prompt_combine_topn_cot_diff = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

{repair_relevant_file_instruction}
--- BEGIN FILE ---
```
{content}
```
--- END FILE ---

Please first localize some possible locations in those test files within the repo, and then generate *SEARCH/REPLACE* edit updates to the **test** files in the repo, so that the erroneous scenario described in the problem is reproduced.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask

def test__rules__std_L060_raised() -> None:
    try:
        sql = "SELECT   IFNULL(NULL, 100),
            NVL(NULL,100);"
        result = lint(sql, rules=["L060"])
        assert len(result) == 2
    except:
        print("Other issues")
        return

    try:
        assert result[0]["description"] == "Use 'COALESCE' instead of 'IFNULL'."
        assert result[1]["description"] == "Use 'COALESCE' instead of 'NVL'."
        print("Issue resolved")
    except AssertionError:
        print("Issue reproduced")
        return

    return
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.
    """

    file_contents = get_repo_files(structure, found_files)

    file_contentes_dict = remove_test_cases(structure, {}) # SWE_BENCH_VERIFIED does not need to remove test cases.

    contents = ""
    for file_name, content in file_contents.items():
        if(file_name in file_contentes_dict):
            content = file_contentes_dict[file_name]['new_content']
        else:
            file_contentes_dict[file_name] = {
                'old_content': content,
                'new_content': content
            }
        contents += f"{file_name}\n{content}\n\n"

    file_instruction = repair_relevant_file_instruction
    prompt_content = repair_prompt_combine_topn_cot_diff.format(
        repair_relevant_file_instruction=file_instruction,
        problem_statement=problem_statement,
        content=contents.rstrip(),
    ).strip()

    repair_message, search_replace_text = llm_chat(llm_model, prompt_content)

    #TODO: return search&replace text and model patch   

    return repair_message, search_replace_text, file_contentes_dict

def llm_chat(llm_model, prompt):
    traj = llm_model.codegen(prompt)[0]
    traj["prompt"] = prompt
    raw_output = traj["response"]


    def concatenate_responses(raw_output , reasoning_output):
        response = raw_output
        reasoning_response = reasoning_output

        if reasoning_response:
            wrapped_reasoning = f"\u25c1think\u25b7{reasoning_response}\u25c1/think\u25b7"
        else:
            wrapped_reasoning = ""

        return f"{wrapped_reasoning}{response}"

    def post_process(response: str) -> str:
        content = response
        if "◁/think▷" in content:
            content = content.replace("◁think▷", "")
            parts = content.split("◁/think▷")
            content = parts[-1]
        # Extract content between triple backticks (```)
        matches = re.findall(r"```.*?```", content, re.DOTALL)
        
        if matches:
            return "\n".join(matches)  # Return all matched code blocks joined by new lines
        return content  # If no match, return the full response
    answer = post_process(raw_output)

    # prepare meassage
    reasoning_output = traj.get("reasoning_response", "")

    if(reasoning_output):
        message = [{"role": "user", "content": prompt}]
        message.append({"role": "assistant", "content": concatenate_responses(raw_output, reasoning_output)})
    else:
        message = [{"role": "user", "content": prompt}]
        message.append({"role": "assistant", "content": raw_output})

    return message, answer

def solve_instance_data(instance_data, repostructure_dir, output_dir, processed_instance_ids, bad_list, llm_model, enable_gt=True, retry_times=5):
    instance_id = instance_data['instance_id']
    output_file_path = os.path.join(output_dir, f'{instance_id}.jsonl')
    total_retry_times = retry_times

    # Check if the file already exists
    if os.path.exists(output_file_path) and os.path.getsize(output_file_path) != 0:
        print(f"File {output_file_path} already exists. Skipping...")
        return None
    
    # Check if the problem statement is empty
    problem_statement = instance_data.get("problem_statement", {})
    if problem_statement == {}:
        return None

    with open(output_file_path, 'w') as output_file:
        # Check if the instance_id has already been processed
        if instance_id in processed_instance_ids:
            print(f"Duplicate instance_id found: {instance_id}, skipping...")
            return None
        
        # Add the current instance_id to the processed set
        processed_instance_ids.add(instance_id)

        structure = search_instance_id_and_extract_structure(instance_id, repostructure_dir)
        
        # Check if structure is None
        if structure is None:
            print(f"No structure found for instance_id {instance_id}")
            bad_list.append(instance_id)
            return None
        
        gt_patch = instance_data.get("test_patch", "")
        gt_parsed_patch = parse_patch(gt_patch)
        instance_data["parsed_patch"] = gt_parsed_patch
        found_files = get_relevant_files(instance_data)
        found_related_locs, found_edit_locs = generate_found_edit_locs(gt_parsed_patch, structure)
        
        gt_results = {
            "instance_id": instance_id,
            "found_files": found_files,
            "found_related_locs": found_related_locs,
            "found_edit_locs": found_edit_locs,
            "gt_parsed_patch": gt_parsed_patch,
            "gt_patch": gt_patch,
        } 
    
        #if found_related_locs == [] or found_edit_locs == []:
        #    print(f"No related or edit locs found for instance_id {instance_id}")
        #    return 1
        # Generate prompt-response

        model_patch = ""
        llm_model.temperature = 0.0

        while (model_patch == "" or '@@' not in model_patch ) and retry_times > 0:
            file_level_message, pred_found_files = relevant_file_prompt_response(llm_model, problem_statement, structure)

            # 去除输出的非test文件以减少prompt长度，提升速度
            pred_found_files_new = []
            for file in pred_found_files:
                base_name = os.path.basename(file)
                if(base_name.startswith('test')):
                    pred_found_files_new.append(file)
            pred_found_files = pred_found_files_new

            

            if(len(pred_found_files) > 0):
                pred_found_files = pred_found_files[:1] # 取top1的predfile
                if enable_gt == True:
                    repair_message, search_replace_text, file_contentes_dict = repair_prompt_response(llm_model, problem_statement, structure, gt_results["found_files"])
                else:
                    repair_message, search_replace_text, file_contentes_dict = repair_prompt_response(llm_model, problem_statement, structure, pred_found_files)
            else:
                repair_message = []
                file_contentes_dict = {}
                search_replace_text = ""
            
            messages = file_level_message + repair_message

            # Prepare the data to be saved
            instance_dict = {
                "instance_id": instance_id,
                "messages": messages,
                "pred_found_files": pred_found_files,
                "search_replace_text": search_replace_text,
                "gt_results": gt_results,
            }

            # generate model patch
            model_patch = generate_model_patch_difflib_testwritter(file_contentes_dict=file_contentes_dict, search_replace_text=search_replace_text)
            if model_patch == "" or '@@' not in model_patch:
                llm_model.temperature = 1.0
                retry_times -= 1


        instance_dict["model_patch"] = model_patch
        if retry_times > 0:
            instance_dict["retry_times"] = total_retry_times - retry_times + 1
        else:
            instance_dict["retry_times"] = total_retry_times - retry_times

        # Write the instance data to the jsonl file
        output_file.write(json.dumps(instance_dict, ensure_ascii=False) + '\n')
        return 1



if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Select the dataset for processing")
    parser.add_argument('--dataset', type=str, choices=["princeton-nlp/SWE-bench_Verified"],
                        default='princeton-nlp/SWE-bench_Verified', help="Choose the dataset")
    parser.add_argument("--enable_gt", action= "store_true", help = "enable gt locolization files in llm chat")
    parser.add_argument('--selected_num', type=int, default=500, help="Select how much data")
    parser.add_argument('--model_name', type=str, default="rl-best-007-iter175-128k", help="Choose the llm model")
    parser.add_argument('--max_tokens', type=int, default=16*1024, help="Max tokens for the llm model")
    parser.add_argument('--temp', type=float, default=0.0, help="Choose the temperature")
    parser.add_argument('--passk', type=int, default=1, help="Choose the passk number")
    parser.add_argument('--max_workers', type=int, default=100, help="Number of threads for parallel processing")
    parser.add_argument('--save_dir', type=str, default="./results/", help="Save rollout dir")
    parser.add_argument('--retry_times', type=int, default=5, help="Retry times")
    
    args = parser.parse_args()

    # Now use args.dataset to control the selected dataset
    selected_dataset = args.dataset.split("/")[-1]
    swe_bench_data = load_dataset(args.dataset, split="test")
    instance_data_list = list(swe_bench_data)

    # get repostructure dir
    repostructure_dir = os.environ.get('PROJECT_FILE_LOC', '')


    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    for pass_idx in range(args.passk):
        args.temp = 0.0 if pass_idx == 0 else 1.0
        
        if args.enable_gt:
            output_dir = os.path.join(args.save_dir,f"{selected_dataset}-{args.model_name}-testwriter-enable-gt-pass{pass_idx}")
        else:   
            output_dir = os.path.join(args.save_dir,f"{selected_dataset}-{args.model_name}-testwriter-disable-gt-pass{pass_idx}")


        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # make_model
        try:
            llm_model = make_model(model=f"{args.model_name}", backend="kimidev", max_tokens=args.max_tokens, temperature=args.temp)
        except:
            raise ValueError(f"Unknown model name: {args.model_name}")


        # print enable_gt
        if args.enable_gt:
            cprint("Enable ground truth in chat \n", "green")
        else:
            cprint("Disable ground truth in chat \n", "yellow")

        # Cannot find the repo structure
        bad_list = []
        # Instance IDs that have been processed
        processed_instance_ids = set()

        # Step 2: Use ThreadPoolExecutor to process each instance_data in parallel with tqdm progress bar
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            # Initialize tqdm for the progress bar
            with tqdm(total=len(instance_data_list), desc="Processing instances") as pbar:
                for instance_data in instance_data_list:
                    # Submit each instance processing task with required arguments using a lambda
                    futures.append(executor.submit(
                        solve_instance_data, 
                        instance_data, 
                        repostructure_dir, 
                        output_dir, 
                        processed_instance_ids, 
                        bad_list, 
                        llm_model,
                        args.enable_gt,
                        args.retry_times,
                    ))
                # Update the progress bar for each completed task
                for future in as_completed(futures):
                    future.result()  # Wait for the result and complete the task
                    pbar.update(1)  # Update the progress bar by one step

        cprint(f"Data successfully saved to {output_dir}", 'green')

        # Save bad_list as a text file, appending to it if it already exists
        with open(f'{output_dir}/bad_list.txt', 'a') as f:
            for item in bad_list:
                f.write(f"{item}\n")

