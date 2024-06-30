import os
from tqdm import tqdm
from checker import load_solutions
from utils import write_directory


# changes
from datasets import load_from_disk
import re

'''
game of dataset = load_dataset("iidai/avatar")['train']
-dataset[task_id]["code"] which code ?
-for x in dataset ("id", "language")

/home/cse/dual/cs5190439/MTP1/codetlingua/tools/sanitize.py --samples="/scratch/cse/dual/cs5190439/MTP1/PLTranslation_data/codetlingua/autocot2d/codenet/granite-8b-code-instruct/Python/Java/temperature_0.2/" \
    --source_lang="Python" \
    --target_lang="Java" \
    --remove_prompt \
    --eofs "# <END-OF-CODE>" "// <END-OF-CODE>" "<END-OF-CODE>" "END-OF-CODE" "C++" "cpp" "c++" "Python" "python" "Java" "Python:" "Java:" "C:" "C++:" "Go:" "# Answer:" "C Code:" "C++ Code:" "Java Code:" "Python Code:" "Go Code:" "Translate the above" \
    "# 2nd solution:" "Note" "Input:" "Explanation" "A:" \
    --rm-prefix-lines '"""' "Here" "[PYTHON]" "[JAVA]" "[C]" "[C++]" "[GO]" "The" "Please" "This" "That" "After" "You" "-" "In" "Also" "To" "Key" "```"

removed from rm_prefix list-
"Translate the above" 
"C++" "cpp" "c++" "C" "Python" "python" "Java" "java" "Go" "go"
'''

if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str, required=True)
    parser.add_argument("--eofs", nargs="+", type=str, default=[])
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--remove_prompt", action="store_true")
    parser.add_argument("--source_lang", required=True, type=str, choices=["C", "C++", "Java", "Python", "Go"])
    parser.add_argument("--target_lang", required=True, type=str, choices=["C", "C++", "Java", "Python", "Go"])
    parser.add_argument(
        "--debug-task", type=str, help="Enter the task ID to only sanitize that task."
    )
    parser.add_argument("--rm-prefix-lines", nargs="+", type=str, help="Remove lines starting with these string.", default=[])
    args = parser.parse_args()

    EXTENSIONS = { "C": ".c", "C++": ".cpp", "Java": ".java", "Python": ".py", "Go": ".go" }

    # make a new folder with "-sanitized" suffix
    is_folder = os.path.isdir(args.samples)
    target_path = pathlib.Path(args.samples)
    if not args.inplace:
        if is_folder:
            new_name = target_path.name + "-sanitized"
        else:
            new_name = target_path.name.replace(".jsonl", "-sanitized.jsonl")
        target_path = target_path.parent / new_name
    target_path = str(target_path)

    nsan = 0
    ntotal = 0

    new_solutions = []

    for solution in tqdm(load_solutions(args)):
        task_id = solution["task_id"]
        dbg_identifier = solution["_identifier"]
        if args.debug_task is not None and task_id != args.debug_task:
            continue

        ntotal += 1
        old_code = solution["solution"]

        # start to modify old_code | old_code should not be re-defined

        new_code = old_code

        if (args.remove_prompt and 'vanilla' in target_path):
            data =new_code
            last_idx =data.find(f"{args.target_lang}:")
            if (last_idx!=-1):
                # last_idx=data.find("4. Whole Java Code:")
                last_idx+=len(f"{args.target_lang}:")
                data=data[last_idx:]

            last_idx=data.find(f"{args.source_lang}:")
            # if (last_idx==-1):
            #     last_idx =data.rfind("main()")
            #     if (last_idx!=-1):
            #         last_idx+=len("main()")
            if (last_idx!=-1):
                data =data[:last_idx]

            new_code =data
        if (args.remove_prompt and ('autocot2d' in target_path or 'program' in target_path)):
            data =new_code
            data = data[data.find(f"3. {args.source_lang} Code:")+len(f"3. {args.source_lang} Code:"):]
            data = data[data.find(f"3. {args.target_lang} Code:")+len(f"3. {args.target_lang} Code:"):]

            last_idx =data.rfind(f"{args.target_lang} Code:")
            if (last_idx==-1):
                last_idx =data.rfind(f"{args.target_lang} code:")
            if (last_idx!=-1):
                # last_idx=data.find("4. Whole Java Code:")
                last_idx+=len(f"{args.target_lang} Code:")
                data=data[last_idx:]
            # last_idx=data.find("####################")
            # not need with stopping criteria
            # last_idx=data.find("###$###")
            # if (last_idx!=-1):
            #     data =data[:last_idx]

            new_code =data


        # if args.remove_prompt and 'code-llama' in target_path:
        #     new_code = new_code[new_code.find(f"{args.target_lang}:\n") + len(f"{args.target_lang}:\n"):]
        #     while new_code.strip().startswith(f"{args.target_lang}:"):
        #         new_code = new_code[new_code.find(f"{args.target_lang}:\n") + len(f"{args.target_lang}:\n"):]
        
        if args.remove_prompt and 'magicoder' in target_path:
            new_code = new_code.split("@@ Response")[1].strip()

        if args.remove_prompt and 'octocoder' in target_path:
            new_code = new_code.split("Answer:")[1].strip()
        
        if args.remove_prompt and 'mixtral' in target_path:
            new_code = new_code.split(f"{args.target_lang}:")[1].strip()

        if args.remove_prompt and 'phi2' in target_path:
            if 'Output:' in new_code:
                new_code = new_code.split("Output:")[1].strip()

        if args.remove_prompt and ('gpt-4' in target_path or 'gpt-3.5' in target_path):
            trg_lang = args.target_lang.lower() if args.target_lang.lower() != 'c++' else 'cpp'
            new_code = new_code[new_code.find(f"```{trg_lang}") + len(f"```{trg_lang}"):]
            new_code = new_code[:new_code.find("```")]

        if args.remove_prompt and 'gemini-pro' in target_path:
            new_code_lines = new_code.split("\n")
            generation_index = new_code_lines.index(args.target_lang)
            new_code = "\n".join(new_code_lines[generation_index + 1:]).strip()

        if args.remove_prompt and 'claude' in target_path:
            new_code_lines = new_code.split("\n")
            generation_index = new_code_lines.index(args.target_lang)
            new_code = "\n".join(new_code_lines[generation_index + 1:]).strip()

        if args.remove_prompt and 'deepseek' in target_path:
            trg_lang = args.target_lang.lower()
            if trg_lang == 'c++' and '```c++' in new_code:
                trg_lang = 'c++'
            elif trg_lang == 'c++' and '```cpp' in new_code:
                trg_lang = 'cpp'
            new_code = new_code[new_code.find(f"```{trg_lang}") + len(f"```{trg_lang}"):]
            new_code = new_code[:new_code.find("```")]

        # basic handling of chat output
        new_code = new_code.replace(f"```{args.target_lang.lower()}", "").replace("```", "").strip()

        for prefix in args.rm_prefix_lines:
            new_code = "\n".join(
                [
                    line
                    for line in new_code.splitlines()
                    if not line.startswith(prefix)
                ]
            ).strip()

        if (args.target_lang!="Python"):
            new_code = "\n".join(
                [
                    line
                    for line in new_code.splitlines()
                    if not line.startswith("#")
                ]
            ).strip()

        for eof in args.eofs:
            new_code = new_code.split(eof)[0].strip()

        if args.target_lang == 'Java':
            # data = re.sub('public\s*class\s*.+', 'public class ' + filename.split('.')[0] + ' {', data)
            new_code= re.sub(r'public\s*class\s*[^{]+', r'public class ' + solution["task_id"], new_code)
        new_code = new_code.replace('Main', solution["task_id"])#changes 
        new_solutions.append(
            {
                "task_id": solution["task_id"],
                "solution": new_code.strip(),
            }
        )

    if is_folder:
        write_directory(target_path, new_solutions, ext=EXTENSIONS[args.target_lang])
