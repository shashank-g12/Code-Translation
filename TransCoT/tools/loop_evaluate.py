# changes
'''
# cache dirs
bash scripts/evaluate.sh translations deepseek-coder-1.3b-instruct codenet Java Python 0.2 8
python3 /home/cse/dual/cs5190439/MTP1/codetlingua/tools/loop_evaluate.py   --samples="/scratch/cse/dual/cs5190439/MTP1/PLTranslation_data/codetlingua/autocot2d/avatar/starcoder/Java/Python/temperature_0.2-sanitized/" \
                            --dataset="avatar" \
                            --source_lang="Java" \
                            --target_lang="Python" \
                            --parallel=3
                            --re_run
'''
import argparse
import json
import multiprocessing
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import numpy as np

from termcolor import cprint
from tqdm import tqdm
from utils import (
    SUCCESS,
    check_correctness,
    estimate_pass_at_k,
    get_problem,
)
from datasets import load_dataset
from utils import load_solutions


def evaluate(flags):
    if flags.parallel is None:
        n_workers = max(1, multiprocessing.cpu_count() // 2)
    else:
        n_workers = flags.parallel
    while True:
        result_path = os.path.join(flags.samples, "eval_results.json")

        if os.path.isfile(result_path) and not flags.re_run:
            print(f"Load from previous results from {result_path}")
            with open(result_path, "r") as f:
                results = json.load(f)

            for task_results in results["eval"].values():
                # update the "files" field to "nfiles"
                if "files" in task_results and "nfiles" not in task_results:
                    task_results["nfiles"] = len(task_results.pop("files"))

        else:
            if flags.dataset == "codenet":
                problems = load_dataset("iidai/codenet",cache_dir="/home/cse/dual/cs5190439/.cache/huggingface/datasets")['train']
                problems = [p for p in problems if p['language'] == flags.source_lang]
            elif flags.dataset == "avatar":
                problems = load_dataset("iidai/avatar",cache_dir="/home/cse/dual/cs5190439/.cache/huggingface/datasets")['train']
                if (flags.source_lang=="Java"):#special 249 case prev codeforces_373_B
                    problems = [p for p in problems if (p['language'] == flags.source_lang )]#and p['id'] !="codeforces_373_B")]
                else:
                    problems = [p for p in problems if p['language'] == flags.source_lang]


            results = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "eval": {},
            }
            print("ProcessPoolExecutor started-")
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                completion_id = Counter()
                n_samples = 0
                eval_results = defaultdict(list)
                remainings = set()

                print("Reading samples...")
                for sample in tqdm(load_solutions(flags)):
                    task_id = sample["task_id"]
                    solution = sample["solution"]
                    remainings.add(sample["_identifier"])
                    args = (
                        sample["_identifier"].split("/")[-1].split(".")[0],
                        get_problem(problems, task_id),
                        solution,
                        sample["_identifier"],
                        flags,
                    )

                    futures.append(executor.submit(check_correctness, *args))
                    completion_id[task_id] += 1
                    n_samples += 1
                print("debug","n_samples",n_samples,"remainings",len(remainings),"completion_id",len(completion_id),"problems",len(problems))
                print("completion_id[task_id]",task_id,completion_id[task_id])
                print("remainings-0",next(iter(remainings)))

                for future in tqdm(as_completed(futures), total=n_samples):
                    result = future.result()
                    remainings.remove(result["_identifier"])
                    eval_results[result["task_id"]].append(result)

            # sort the results for each problem by completion_id
            for task_id, task_results in eval_results.items():
                task_results.sort(key=lambda x: x["completion_id"])
                results["eval"][task_id] = {
                    "nfiles": len(task_results),
                    "base": [x["base"] for x in task_results]
                }




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, type=str, choices=["codenet", "avatar"]
    )
    parser.add_argument("--samples", required=True, type=str)
    parser.add_argument("--parallel", default=None, type=int)
    parser.add_argument("--re_run", action="store_true")
    parser.add_argument("--source_lang", required=True, type=str, choices=["C", "C++", "Java", "Python", "Go"])
    parser.add_argument("--target_lang", required=True, type=str, choices=["C", "C++", "Java", "Python", "Go"])
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
