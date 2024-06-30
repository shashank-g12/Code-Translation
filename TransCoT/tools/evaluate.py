# changes
'''
# cache dirs
bash scripts/evaluate.sh translations deepseek-coder-1.3b-instruct codenet Java Python 0.2 8
python3 /home/cse/dual/cs5190439/MTP1/codetlingua/tools/evaluate.py   --samples="/scratch/cse/dual/cs5190439/MTP1/PLTranslation_data/codetlingua/vanilla/avatar/starcoder/Python/Java/temperature_0.2-sanitized/" \
                            --dataset="avatar" \
                            --source_lang="Python" \
                            --target_lang="Java" \
                            --parallel=20
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
from datasets import load_from_disk

import logging

def evaluate(flags):
    if flags.parallel is None:
        n_workers = max(1, multiprocessing.cpu_count() // 2)
    else:
        n_workers = flags.parallel

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
        if flags.dataset == "codenet":#load_from_disk
            # problems = load_dataset("iidai/codenet",cache_dir="/home/cse/dual/cs5190439/.cache/huggingface/datasets")['train']
            problems = load_from_disk("/scratch/cse/dual/cs5190439/MTP1/PLTranslation_data/dataset/codenet_dataset")['train']
            problems = [p for p in problems if p['language'] == flags.source_lang]
        elif flags.dataset == "avatar":
            # problems = load_dataset("iidai/avatar",cache_dir="/home/cse/dual/cs5190439/.cache/huggingface/datasets")['train']
            problems = load_from_disk("/scratch/cse/dual/cs5190439/MTP1/PLTranslation_data/dataset/avatar_dataset")['train']
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

            assert n_samples == len(remainings), "Missing problems in unfinished"
            # according to sample 10 , it will be 2490 in avatar j2p
            assert len(completion_id) == len(problems), "Missing problems in samples"
            #number of problems -249 in avatar j2P 


            for future in tqdm(as_completed(futures), total=n_samples):
                # try:
                result = future.result()  # Adjust the timeout as needed (100+30+20relax)
                # if we use timeout than there will be problem as let's assume one test case passes and other will be failed in future after timout than it will be considered as correct
                remainings.remove(result["_identifier"])
                eval_results[result["task_id"]].append(result)
                # except Exception as e:
                #     print(f"Error processing future: {e}")
                # PREV-
                # result = future.result()
                # remainings.remove(result["_identifier"])
                # eval_results[result["task_id"]].append(result)

        # sort the results for each problem by completion_id
        for task_id, task_results in eval_results.items():
            task_results.sort(key=lambda x: x["completion_id"])
            results["eval"][task_id] = {
                "nfiles": len(task_results),
                "base": [x["base"] for x in task_results]
            }

    if os.path.isfile(result_path) and flags.re_run:
        decision = ""
        while decision.lower() not in ["y", "n"]:
            print(f"{result_path} already exists. Press [Y/N] to overwrite or exit...")
            decision = input()

        if decision.lower() == "y":
            # mv the file to a backup
            new_path = result_path + ".bak"
            while os.path.isfile(new_path):
                new_path += ".bak"
            os.rename(result_path, new_path)
            print(f"Backup {result_path} to {new_path}")

    if not os.path.isfile(result_path):
        with open(result_path, "w") as f:
            json.dump(results, f)

    # Calculate pass@k.
    total = np.array([r["nfiles"] for r in results["eval"].values()])
    total_correct = []

    for res in results["eval"].values():
        bc = sum([r == SUCCESS for r in res["base"]])
        total_correct.append(bc)
    total_correct = np.array(total_correct)

    pass_at_k = {
        f"pass @ {k}": estimate_pass_at_k(total, total_correct, k).mean()
        for k in [1, 5]
        if total.min() >= k
    }
    cprint(f"{flags.dataset} ({flags.source_lang} -> {flags.target_lang})", "red")
    logging.info(f"{flags.dataset} ({flags.source_lang} -> {flags.target_lang})")
    for k, v in pass_at_k.items():
        cprint(f"{k} :\t{v:.3f}", "red")
        logging.info(f"{k} :\t{v:.3f}")

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
    workdir ="/home/cse/dual/cs5190439/MTP1/codetlingua"
    logging.basicConfig(filename=os.path.join(workdir, 'evaluation_log.log'), level=logging.INFO, format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(f"Evaluation started with args: {args}")

    evaluate(args)

    logging.info(f"Evaluation finished")


if __name__ == "__main__":
    main()
