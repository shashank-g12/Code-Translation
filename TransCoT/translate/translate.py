# changes
import re
import time
from pathlib import Path
from tqdm import tqdm

# program changes
import ast
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
import tree_sitter_python as tspython
pattern = r'\bglobal\b\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)'

'''
-model locations
-prompt types
-gpus -auto
-EOS
-input/output locations 
    -approach
    -full out loc
-log/print
python3 /home/cse/dual/cs5190439/MTP1/codetlingua/translate/translate.py --model=granite-20b-code-instruct --dataset=avatar --source_lang=Python --target_lang=Java --prompt_type=vanilla --temperature=0.2 --n_samples=10 --batch_size=10 --max_length=4090 --ngpus=2
autocot2d, vanilla, autocot1
granite-20b-code-instruct
'''
import os

import argparse
from os import PathLike
import logging

from datasets import load_dataset
from model import DecoderBase, make_model, compose_prompt
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

def find_file_with_smallest_length(folder_path, prefix):
    # Get a list of all files in the folder
    files = [f for f in os.listdir(folder_path) if f.startswith(prefix) and (f.endswith(".in") or f.endswith("_in.txt")) ]

    if not files:
        print(f"No files found with prefix '{prefix}' and '.in' extension.")
        return None

    # Initialize variables for tracking the smallest read string length and corresponding file name
    smallest_length = float('inf')
    smallest_length_file = None

    # Iterate through the files
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)

        # Read the content of the file
        with open(file_path, 'r') as file:
            content = file.read()

            # Check if the length is smaller than the current smallest length
            if len(content) < smallest_length:
                smallest_length = len(content)
                smallest_length_file = file_name

    return smallest_length_file

def compute_tree(node, globals_vars,level=0):
    # print('  ' * level + str(type(node).__name__))
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store):
                globals_vars.add(target.id)
            elif (isinstance(target, ast.Tuple) or isinstance(target, ast.List)) and isinstance(target.ctx, ast.Store):
                for tuple_target in target.elts:
                    if isinstance(tuple_target, ast.Name) and isinstance(tuple_target.ctx, ast.Store):
                        globals_vars.add(tuple_target.id)
    # (a): int = 1
    elif isinstance(node, ast.AnnAssign):
        if (isinstance(node.target, ast.Name) and node.simple==0):# Annotation with parenthesis
            globals_vars.add(node.target.id)
    # elif isinstance(node, ast.For):
    #     if isinstance(node.target, ast.Name) and isinstance(node.target.ctx, ast.Store):
    #             globals_vars.add(node.target.id)
    #     elif (isinstance(node.target, ast.Tuple) or isinstance(node.target, ast.List)) and isinstance(node.target.ctx, ast.Store):
    #         for child_target in node.target.elts:
    #             if isinstance(child_target, ast.Name) and isinstance(child_target.ctx, ast.Store):
    #                 globals_vars.add(child_target.id)
    elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
        return
    if isinstance(node, ast.AST):
        for child in ast.iter_child_nodes(node):
            compute_tree(child,globals_vars, level + 1)

def get_structure_python(tree, indent, code):
    cursor = tree.walk()
    cursor.goto_first_child()
    res = ''
    while True:
        if cursor.node.type == 'class_definition':
            idx = 0
            res += indent*'\t'
            while cursor.node.children[idx].type != 'block':
                res += code[cursor.node.children[idx].start_byte: cursor.node.children[idx].end_byte] + ' '
                idx += 1
            
            res += '\n'
            res += get_structure_python(cursor.node.child_by_field_name('body'), indent+1, code)
            res += indent*'\t' + '\n'

        if cursor.node.type == 'function_definition':
            idx = 0
            res += (indent)*'\t'
            while cursor.node.children[idx].type != 'block':
                res += code[cursor.node.children[idx].start_byte: cursor.node.children[idx].end_byte] + ' '
                idx += 1

            res += '\n'
            # Add body content of __init__ function
            if cursor.node.children[1].text == b'__init__':
                body_node = cursor.node.child_by_field_name('body')
                for line in body_node.children:
                    res += (indent+1)*'\t' + code[line.start_byte: line.end_byte] + '\n'

            res += get_structure_python(cursor.node.child_by_field_name('body'), indent+1, code)
            res += '\n'
        
        if not cursor.goto_next_sibling():
            break
    return res


def translate(args, workdir: PathLike, model: DecoderBase):

    EXTENSIONS = { "C": ".c", "C++": ".cpp", "Java": ".java", "Python": ".py", "Go": ".go" }
    '''
    with Progress(
        TextColumn(
            f"{args.dataset} •" + "[progress.percentage]{task.percentage:>3.0f}%"
        ),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    ) as p:
        if args.dataset == "codenet":
            dataset = load_dataset("iidai/codenet")

        elif args.dataset == "avatar":
            dataset = load_dataset("iidai/avatar")

        for item in p.track(dataset['train']):

            if item['language'] != args.source_lang:
                continue

            p_name = item['id']
        '''

    in_folder = f'/scratch/cse/dual/cs5190439/MTP1/PLTranslation_data/dataset/{args.dataset}/{args.source_lang}/Code'
    in_files = os.listdir(in_folder)
    print(f'found {len(in_files)} inputs')
    total_time=0
    for f in tqdm(in_files):
        prompt_file = f'{in_folder}/{f}'

        p_name=f[:(-1*(len(EXTENSIONS[args.source_lang])))] #code_id
        os.makedirs(os.path.join(workdir, p_name), exist_ok=True)

        log = f"Translate: {p_name} from {args.source_lang}-{args.dataset} to {args.target_lang} using {args.model}"
        n_existing = 0
        if args.resume:
            # count existing translated files
            n_existing = len(
                [
                    f
                    for f in os.listdir(os.path.join(workdir, p_name))
                    if f.endswith(EXTENSIONS[args.target_lang])
                ]
            )
            if n_existing > 0:
                log += f" (resuming from {n_existing})"

        nsamples = args.n_samples - n_existing
        # p.console.print(log)
        logging.info(log)
        code_data=''
        with open(prompt_file, 'r',encoding="utf-8") as file:
            code_data = file.read()
            code_data= re.sub(r'public\s*class\s*[^{]+', r'public class ' + "Main ", code_data)
            code_data = code_data.replace(p_name,'Main')

        # io add
        prompt = compose_prompt(args.prompt_type, args.source_lang, args.target_lang, code_data)#item['code']
        if (args.prompt_type!="vanilla"):
            # io related
            # '''
            # prompt+=code_data
            code_id = p_name
            common_path=f"/scratch/cse/dual/cs5190439/MTP1/PLTranslation_data/dataset/{args.dataset}/{args.source_lang}/TestCases/"
            input_path = find_file_with_smallest_length(common_path, code_id)
            input_path =common_path + input_path
            if (args.dataset=="codenet"):
                output_path = input_path[:-7] +"_out.txt"
            else:
                output_path = input_path[:-3] +".out"
            input_as_str= Path(input_path).read_text(encoding="utf-8")
            output_as_str = Path(output_path).read_text(encoding="utf-8")
            # io
            prompt+=f"\n3. Sample Input:\n{input_as_str}\n3. Expected Output:\n{output_as_str}\n\n3. Steps: Let's think step by step.\n\n"

            if (args.prompt_type=="program" and args.source_lang=='Python' ):
                code =code_data
                # Find all matches in the code
                matches = re.findall(pattern, code)

                # Extract the variable names
                global_vars = []
                for match in matches:
                    vars_in_match = [var.strip() for var in match.split(',')]
                    global_vars.extend(vars_in_match)

                globals_vars=set(global_vars)
                # Parse the code into an AST
                tree = ast.parse(code)# feature_version=(3, 10)
                compute_tree(tree, globals_vars)
                # print(globals_vars)

                parser = Parser(Language(tspython.language()))
                tree = parser.parse(bytes(code, 'utf-8'))

                res =get_structure_python(tree, 0, code)
                if ("__name__" in code and "__main__" in code ):
                    res+='''if __name__ == "__main__" :\n'''
                # print(res)

                # modified prompt
                prompt+="Step 1: First of all identify global variables. Global variables are variables that are in global scope and not inside scope of any function and class present in the above Python Code-\n"
                prompt+="Global variables:\n"
                for var in globals_vars:#boundary
                    prompt+=var +", "
                prompt+="\n"

                prompt+="\nStep 2: Identify declarations of classes and functions present in the above Python Code-\n"
                prompt+="Identified declarations of classes and functions in the Python code:\n"
                prompt+=res
                prompt+="\nStep 3:\n"
            # '''
            # try:

        t0 = time.perf_counter()

        sidx = args.n_samples - nsamples
        while sidx < args.n_samples:
            # prompt = f"{args.source_lang}:\n{item['code']}\n\nTranslate the above {args.source_lang} code to {args.target_lang} and end with comment \"<END-OF-CODE>\".\n\n{args.target_lang}:\n"
            outputs = model.codegen(prompt,
                do_sample=not args.greedy,
                num_samples=args.n_samples - sidx,
                max_length=args.max_length,
            )

            assert outputs, "No outputs from model!"
            for impl in outputs:
                try:
                    with open(
                        os.path.join(workdir, p_name, f"{sidx}{EXTENSIONS[args.target_lang]}"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        if model.conversational:
                            f.write(impl)
                        else:
                            f.write(prompt + impl)
                except UnicodeEncodeError:
                    continue
                sidx += 1
        
        t1 = time.perf_counter()
        print("Total generation time:", p_name,t1 - t0)
        total_time+=( t1 - t0)

    print("Avg generation time of run :", total_time/len(in_files))
    logging.info(f"Avg generation time of run :{total_time/len(in_files)}")
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--prompt_type", default='vanilla', type=str, choices=["vanilla", "program","autocot1", "autocot2d_p2j","autocot2d","gemini", "claude", "gpt", "codellama", "octocoder", "dolphin", "solar", "wizardcoder","deepseek"])
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--dataset", required=True, type=str, choices=["codenet", "avatar"])
    parser.add_argument("--source_lang", required=True, type=str, choices=["C", "C++", "Java", "Python", "Go"])
    parser.add_argument("--target_lang", required=True, type=str, choices=["C", "C++", "Java", "Python", "Go"])
    parser.add_argument("--root", type=str, default="translations")
    parser.add_argument("--n_samples", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--ngpus", default=1, type=int)
    parser.add_argument("--max_length", default=1024, type=int)
    args = parser.parse_args()

    gpus_str="0"
    for i in range(1,args.ngpus):
        gpus_str+=f",{i}"
    
    # single GPU even-odd game
    # gpus_str =f"{(args.ngpus%2)}"
    os.environ["CUDA_VISIBLE_DEVICES"] =gpus_str

    if args.greedy and (args.temperature != 0 or args.batch_size != 1 or args.n_samples != 1):
        args.temperature = 0
        args.batch_size = 1
        args.n_samples = 1
        print("Greedy decoding ON (--greedy): setting batch_size=1, n_samples=1, temperature=0")

    args.root="/scratch/cse/dual/cs5190439/MTP1/PLTranslation_data/codetlingua/"
    if (args.prompt_type=="vanilla"):
        approach="vanilla"
    elif (args.prompt_type=="program"):
        approach="program"
    elif (args.prompt_type=="autocot2d" or args.prompt_type=="autocot2d_p2j"):
        approach="autocot2d"
    else:
        approach =args.prompt_type
    #vanilla, autocot2d
    # Make project dir
    # approach dir 
    args.root+=approach
    os.makedirs(args.root, exist_ok=True)

    # Make dataset dir
    os.makedirs(os.path.join(args.root, args.dataset), exist_ok=True)
    # Make dir for codes generated by each model
    args.model = args.model.lower()
    model = make_model(
        name=args.model, batch_size=args.batch_size, temperature=args.temperature, ngpus=args.ngpus
    )
    workdir = os.path.join(
        args.root,
        args.dataset,
        args.model,
        args.source_lang,
        args.target_lang,
        f"temperature_{args.temperature}"
    )
    os.makedirs(workdir, exist_ok=True)
    
    with open(os.path.join(workdir, "model.txt"), "w") as f:
        f.write(str(model.__dict__))
    
    logging.basicConfig(filename=os.path.join(workdir, 'log.log'), level=logging.INFO, format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(f"translation started with args: {args}")

    translate(args, workdir=workdir, model=model)

    logging.info(f"translation finished")

# os.makedirs(f'./codetlingua/logs', exist_ok=True)
# logging.basicConfig(filename=f"'./codetlingua/logs/opensource_LLMs.log", level=logging.INFO, format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


if __name__ == "__main__":
    main()
