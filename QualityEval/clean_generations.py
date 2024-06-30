"""
We used a simple heuristic to clean the generations of the open-source models. Feel free to play with the extraction
heuristic to get better results.
"""

import os
import re
import argparse
'''
pseudocode approach cleaning-
1. Start, End
2.# End of Code

#specific to java cleaning 
    # -care public class cleaning 
    # re.sub(r'public\s*class\s*[^{]+', r'public class ' + "atcoder_ABC17_B" , s1)


'''
def find_smallest_index(text, strings=["Code summary:", "Explanation:", "End of Code","pseudocode algorithm:"]):
    text_lower = text.lower()  # Convert text to lowercase
    strings_lower = [s.lower() for s in strings]  # Convert strings to lowercase

    indices = [text_lower.find(s) for s in strings_lower]
    valid_indices = [idx for idx in indices if idx != -1]
    
    if valid_indices:
        return min(valid_indices)
    else:
        return -1

def list_files(startpath):
    files = []
    for root, dirs, walkfiles in os.walk(startpath):
        for name in walkfiles:
            files.append(os.path.join(root, name))

    return files


def clean_codegeex(dataset):
    main_path = f'output/CodeGeeX/{dataset}'
    output_path = 'output/CodeGeeX/'

    files = list_files(main_path)

    for f in files:

        splitted = f.split('/')
        filename = splitted[-1].strip()
        target_lang = splitted[-2].strip()
        source_lang = splitted[-3].strip()

        with open(f, 'r') as file:
            data = file.read()

        valid_lines = []
        for line in data.split('\n'):
            if line.strip() in ['C:', 'C++:', 'Java:', 'Python:', 'Go:', '"""']:
                break
            else:
                valid_lines.append(line)
        
        data = '\n'.join(valid_lines)
        data = data.replace('<|endoftext|>', '')
        data = data.replace(f'```{target_lang.lower()}', '')
        data = data.replace(f'```', '')

        if target_lang == 'Java':
            data = re.sub('public\s*class\s*.+', 'public class ' + filename.split('.')[0] + ' {', data)
        
        if target_lang == 'Java' and dataset == 'evalplus':
            data = 'package com.example;\n' + data

        os.makedirs(output_path + dataset + '/' + source_lang + '/' + target_lang, exist_ok=True)
        with open(output_path + dataset + '/' + source_lang + '/' + target_lang + '/' + filename, 'w') as file:
            file.write(data)


def clean_codegen(dataset):
    main_path = f'output/CodeGen/{dataset}'
    output_path = 'output/CodeGen/'

    files = list_files(main_path)

    for f in files:
        print(f)
        splitted = f.split('/')
        filename = splitted[-1].strip()
        target_lang = splitted[-2].strip()
        source_lang = splitted[-3].strip()

        with open(f, 'r') as file:
            data = file.read()

        data = data[data.find(f'{target_lang} Code:')+len(f'{target_lang} Code:'):].strip()        

        valid_lines = []
        for line in data.split('\n'):
            if line.strip() in ["*/", 'C Code:', 'C++ Code:', 'Java Code:', 'Python Code:', 'Go Code:']:
                break
            else:
                valid_lines.append(line)
        
        data = '\n'.join(valid_lines)
        data = data.replace('<|endoftext|>', '')

        if target_lang == 'Java':
            data = re.sub('public\s*class\s*.+', 'public class ' + filename.split('.')[0] + ' {', data)

        if target_lang == 'Java' and dataset == 'evalplus' and 'package com.example;' not in data:
            data = 'package com.example;\n' + data

        os.makedirs(output_path + dataset + '/' + source_lang + '/' + target_lang, exist_ok=True)
        with open(output_path + dataset + '/' + source_lang + '/' + target_lang + '/' + filename, 'w',encoding="utf-8") as file:
            file.write(data)



def clean_starcoder(dataset):
    main_path = f'/home/scai/mtech/aib222684/MTP/output_cot_orig_full/StarCoder/{dataset}'
    output_path = '/home/scai/mtech/aib222684/MTP/output_cot_orig/StarCoder/'

    files = list_files(main_path)

    for f in files:
        print(f)
        splitted = f.split('/')
        filename = splitted[-1].strip()
        target_lang = splitted[-2].strip()
        source_lang = splitted[-3].strip()

        #specific to java cleaning
        # if (target_lang!="Java"):
        #     continue

        if (target_lang=="__pycache__"):
            continue
        with open(f, 'r',encoding="utf-8") as file:
            data = file.read()
        
        data = data[data.find('<fim_suffix><fim_middle>')+len('<fim_suffix><fim_middle>'):]
        '''
        # cleaning of summary 
        # last_index=data.find('End of Code')
        last_index=find_smallest_index(data)
        if (last_index==-1):
            if (target_lang!="Python"):# else last } in java code
                last_index =data.rfind("}")+1
                data=data[:last_index]
        else:
            data=data[:last_index]
        # some error possibility if not found as -1


        # cleaning of pseudocode 
        last_index=find_smallest_index(data)
        if (last_index==-1):
            if (target_lang!="Python"):# else last } in java code
                last_index =data.rfind("}")+1
                data=data[:last_index]
        else:
            data=data[:last_index]
        # some error possibility if not found as -1
        '''
        # cleaninig of icl
        # 5. Java code:
        '''
        if (target_lang=="Python"):
            last_index=data.find("5. Java code:")
            if (last_index!=-1 ):# 
                data=data[:last_index]
        elif (target_lang=="Java"):
            last_index=data.find("5. Python code:")
            if (last_index!=-1 ):# 
                data=data[:last_index]
        '''
        # cleaning of autocot2d
        data = data[data.find('3. Python Code:')+len('3. Python Code:'):]

        # cleaning of autocot1
        # data = data[data.find('3. Whole Python Code:')+len('3. Whole Python Code:'):]

        last_idx =data.find('####################')
        if (last_idx==-1):
            # last_idx=data.find("4. Whole Java Code:")
            last_idx=data.find("4. Java Code:")
        if (last_idx==-1):
            last_idx =data.find("main()")
            if (last_idx!=-1):
                last_idx+=len("main()")
        if (last_idx!=-1):
            data =data[:last_idx]

        # b/w -substring
        #  or startwith '''
        # last_idx =data.find(f"```{target_lang.lower()}")
        
        # main() -> main(self) in autocot2d
        # data = data.replace("def main()", "def main(self)")

        valid_lines = []
        for line in data.split('\n'):
            # Start-End due to pseudocode part
            if line.strip() in ["'''","```",'End','C Code:', 'C++ Code:', 'Java Code:', 'Python Code:', 'Go Code:', '"""'] or line.strip().startswith("Input") or line.strip().startswith("Output"):
                break
            elif line.strip()=='Start' or line.strip().startswith("```") or line.strip()==target_lang:
                continue
            else:
                valid_lines.append(line)
        
        data = '\n'.join(valid_lines)
        data = data.replace('<|endoftext|>', '')

        if target_lang == 'Java':
            # data = re.sub('public\s*class\s*.+', 'public class ' + filename.split('.')[0] + ' {', data)
            data= re.sub(r'public\s*class\s*[^{]+', r'public class ' + filename.split('.')[0], data)

        if target_lang == 'Java' and dataset == 'evalplus' and 'package com.example;' not in data:
            data = 'package com.example;\n' + data

        os.makedirs(output_path + dataset + '/' + source_lang + '/' + target_lang, exist_ok=True)
        with open(output_path + dataset + '/' + source_lang + '/' + target_lang + '/' + filename, 'w',encoding="utf-8") as file:
            file.write(data)


import os
import glob
# to clean granite and starcoder
def clean_new_models(dataset):#granite-20b-code-instruct
    target_lang = args.target_lang
    source_lang = args.source_lang
    EXTENSIONS = { "C": ".c", "C++": ".cpp", "Java": ".java", "Python": ".py", "Go": ".go" }
    extn= EXTENSIONS[args.target_lang]
    main_path = f"/home/codetrans/callGraphEval/codetlingua/{args.approach}/{dataset}/{args.model}/{args.source_lang}/{args.target_lang}/temperature_0"
    # output_path= "/scratch/cse/dual/cs5190439/MTP1/PLTranslation_data/codetlingua/autocot2d/avatar/starcoder/Python/Java/temperature_0/"
   
    output_path =main_path +'/'
    cur_folder_path = main_path
    # Loop through all subdirectories in the current folder path
    for dir_path in glob.glob(os.path.join(cur_folder_path, '*')):
        if os.path.isdir(dir_path):  # Check if it's a directory
            dir_id = os.path.basename(dir_path)  # Extract folder_id as dir_id
            target_file_path = os.path.join(dir_path, f"0{extn}")
            if os.path.isfile(target_file_path):  # Check if the file exists
                with open(target_file_path, 'r', encoding="utf-8") as file:
                    data = file.read()
                #body
                # filename=dir_id+".java"
                filename=dir_id+f"{extn}"
                # data = data[data.find('<fim_suffix><fim_middle>')+len('<fim_suffix><fim_middle>'):]
                '''
                # cleaning of summary 
                # last_index=data.find('End of Code')
                last_index=find_smallest_index(data)
                if (last_index==-1):
                    if (target_lang!="Python"):# else last } in java code
                        last_index =data.rfind("}")+1
                        data=data[:last_index]
                else:
                    data=data[:last_index]
                # some error possibility if not found as -1


                # cleaning of pseudocode 
                last_index=find_smallest_index(data)
                if (last_index==-1):
                    if (target_lang!="Python"):# else last } in java code
                        last_index =data.rfind("}")+1
                        data=data[:last_index]
                else:
                    data=data[:last_index]
                # some error possibility if not found as -1
                '''
                # cleaninig of icl
                # 5. Java code:
                '''
                if (target_lang=="Python"):
                    last_index=data.find("5. Java code:")
                    if (last_index!=-1 ):# 
                        data=data[:last_index]
                elif (target_lang=="Java"):
                    last_index=data.find("5. Python code:")
                    if (last_index!=-1 ):# 
                        data=data[:last_index]
                '''
                # cleaning of autocot2d 
                if (args.approach!="vanilla"):
                    data = data[data.find(f"3. {source_lang} Code:")+len(f"3. {source_lang} Code:"):]
                    data = data[data.find(f"3. {target_lang} Code:")+len(f"3. {target_lang} Code:"):]

                    last_idx =data.rfind(f"{target_lang} Code:")
                    if (last_idx==-1):
                        last_idx =data.rfind(f"{target_lang} code:")

                    if (last_idx!=-1):
                        # last_idx=data.find("4. Whole Java Code:")
                        last_idx+=len(f"{target_lang} Code:")
                        data=data[last_idx:]

                    # last_idx=data.find("####################")
                    # not need with stopping criteria
                    last_idx=data.find("###$###")
                    if (last_idx!=-1):
                        data =data[:last_idx]

                # vanilla
                if (args.approach=="vanilla"):
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

                # cleaning of autocot1
                # data = data[data.find('3. Whole Python Code:')+len('3. Whole Python Code:'):]

                # last_idx =data.find('####################')
                # if (last_idx==-1):
                #     # last_idx=data.find("4. Whole Java Code:")
                #     last_idx=data.find("4. Java Code:")
                # if (last_idx==-1):
                #     last_idx =data.find("main()")
                #     if (last_idx!=-1):
                #         last_idx+=len("main()")
                # if (last_idx!=-1):
                #     data =data[:last_idx]

                # b/w -substring
                #  or startwith '''
                # last_idx =data.find(f"```{target_lang.lower()}")
                
                # main() -> main(self) in autocot2d
                # data = data.replace("def main()", "def main(self)")

                valid_lines = []
                for line in data.split('\n'):
                    # Start-End due to pseudocode part
                    if line.strip() in ["'''","```",'End','C Code:', 'C++ Code:', 'Java Code:', 'Python Code:', 'Go Code:', '"""'] or line.strip().startswith("Input") or line.strip().startswith("Output"):
                        break
                    elif line.strip()=='Start' or line.strip().startswith("```") or line.strip()==target_lang or (line.strip().startswith("#") and target_lang!="Python") :#EOS problem of ###$ or comments
                        continue
                    else:
                        valid_lines.append(line)
                
                data = '\n'.join(valid_lines)
                data = data.replace('<|endoftext|>', '')

                if target_lang == 'Java':
                    # data = re.sub('public\s*class\s*.+', 'public class ' + filename.split('.')[0] + ' {', data)
                    data= re.sub(r'public\s*class\s*[^{]+', r'public class ' + filename.split('.')[0], data)
                    # changes made here
                    data = data.replace('Main', filename.split('.')[0])

                if target_lang == 'Java' and dataset == 'evalplus' and 'package com.example;' not in data:
                    data = 'package com.example;\n' + data

                # os.makedirs(output_path + dataset + '/' + source_lang + '/' + target_lang, exist_ok=True)
                with open(output_path + filename, 'w',encoding="utf-8") as file:
                    file.write(data)


def clean_llama(dataset):
    main_path = f'output/LLaMa/{dataset}'
    output_path = 'output/LLaMa/'

    files = list_files(main_path)

    for f in files:

        splitted = f.split('/')
        filename = splitted[-1].strip()
        target_lang = splitted[-2].strip()
        source_lang = splitted[-3].strip()

        with open(f, 'r',encoding="utf-8") as file:
            data = file.read()

        data = data[data.find(f'{target_lang} Code:')+len(f'{target_lang} Code:'):].strip()

        valid_lines = []
        for line in data.split('\n'):
            if line.strip().startswith("Sure"):
                continue
            if line.strip().startswith("Please") or line.strip().startswith("Note") or line.strip().startswith("Here") or line.strip().startswith("In"):
                break
            else:
                valid_lines.append(line)
        
        data = '\n'.join(valid_lines)

        data = data.replace('</s>', '')
        data = data.replace(f'```{target_lang.lower()}', '')
        data = data.replace(f'```', '')

        if target_lang == 'Java':
            data = re.sub('public\s*class\s*.+', 'public class ' + filename.split('.')[0] + ' {', data)

        if target_lang == 'Java' and dataset == 'evalplus' and 'package com.example;' not in data:
            data = 'package com.example;\n' + data

        os.makedirs(output_path + dataset + '/' + source_lang + '/' + target_lang, exist_ok=True)
        with open(output_path + dataset + '/' + source_lang + '/' + target_lang + '/' + filename, 'w', encoding="utf-8") as file:
            file.write(data)


def clean_airoboros(dataset):
    main_path = f'output/TB-Airoboros/{dataset}'
    output_path = 'output/TB-Airoboros/'

    files = list_files(main_path)

    for f in files:

        splitted = f.split('/')
        filename = splitted[-1].strip()
        target_lang = splitted[-2].strip()
        source_lang = splitted[-3].strip()

        with open(f, 'r',encoding="utf-8") as file:
            data = file.read()

        data = data[data.find(f'{target_lang} Code:')+len(f'{target_lang} Code:'):].strip()

        data = data.replace('</s>', '')

        if target_lang == 'Java':
            data = re.sub('public\s*class\s*.+', 'public class ' + filename.split('.')[0] + ' {', data)

        if target_lang == 'Java' and dataset == 'evalplus':
            data = 'package com.example;\n' + data

        os.makedirs(output_path + dataset + '/' + source_lang + '/' + target_lang, exist_ok=True)
        with open(output_path + dataset + '/' + source_lang + '/' + target_lang + '/' + filename, 'w',encoding="utf-8") as file:
            file.write(data)


def clean_vicuna(dataset):
    main_path = f'output/TB-Vicuna/{dataset}'
    output_path = 'output/TB-Vicuna/'

    files = list_files(main_path)

    for f in files:

        splitted = f.split('/')
        filename = splitted[-1].strip()
        target_lang = splitted[-2].strip()
        source_lang = splitted[-3].strip()

        with open(f, 'r',encoding="utf-8") as file:
            data = file.read()

        data = data[data.find(f'{target_lang} Code:')+len(f'{target_lang} Code:'):].strip()

        valid_lines = []
        for line in data.split('\n'):
            if line.strip().startswith("Translate the above") or line.strip().startswith("Note:") or line.strip().startswith("What is"):
                break
            else:
                valid_lines.append(line)
        
        data = '\n'.join(valid_lines)

        data = data.replace('</s>', '')

        if target_lang == 'Java':
            data = re.sub('public\s*class\s*.+', 'public class ' + filename.split('.')[0] + ' {', data)

        if target_lang == 'Java' and dataset == 'evalplus':
            data = 'package com.example;\n' + data

        os.makedirs(output_path + dataset + '/' + source_lang + '/' + target_lang, exist_ok=True)
        with open(output_path + dataset + '/' + source_lang + '/' + target_lang + '/' + filename, 'w',encoding="utf-8") as file:
            file.write(data)


def clean_gpt4(dataset):
    main_path = f'/scratch/cse/dual/cs5190439/MTP1/PLTranslation_data/gpt4_exp/output_autocot2d_full/gpt-4/{dataset}'
    output_path = '/scratch/cse/dual/cs5190439/MTP1/PLTranslation_data/gpt4_exp/output_autocot2d/gpt-4/'

    files = list_files(main_path)

    for f in files:
        print(f)
        splitted = f.split('/')
        filename = splitted[-1].strip()
        target_lang = splitted[-2].strip()
        source_lang = splitted[-3].strip()
        #specific to java cleaning
        # if (target_lang!="Java"):
        #     continue

        if (target_lang=="__pycache__"):
            continue
        with open(f, 'r',encoding="utf-8") as file:
            data = file.read()

        data = data[data.find('3. Python Code:')+len('3. Python Code:'):]
        last_idx =data.rfind('####################')
        if (last_idx==-1):
            last_idx=data.rfind("4. Java Code:")
        if (last_idx==-1):
            last_idx =data.rfind("main()")
            if (last_idx!=-1):
                last_idx+=len("main()")
        if (last_idx!=-1):
            data =data[:last_idx]

        # b/w -substring
        #  or startwith '''
        # last_idx =data.find(f"```{target_lang.lower()}")
        
        # main() -> main(self)
        data = data.replace("def main()", "def main(self)")

        valid_lines = []
        for line in data.split('\n'):
            # Start-End due to pseudocode part
            if line.strip() in ["'''","```",'End','C Code:', 'C++ Code:', 'Java Code:', 'Python Code:', 'Go Code:', '"""'] or line.strip().startswith("Input") or line.strip().startswith("Output"):
                break
            elif line.strip().startswith("```") or line.strip()==target_lang:#f"```{target_lang.lower()}" case
                continue
            else:
                valid_lines.append(line)
        
        data = '\n'.join(valid_lines)

        if target_lang == 'Java':
            # data = re.sub('public\s*class\s*.+', 'public class ' + filename.split('.')[0] + ' {', data)
            data= re.sub(r'public\s*class\s*[^{]+', r'public class ' + filename.split('.')[0], data)


        if target_lang == 'Java' and dataset == 'evalplus' and 'package com.example;' not in data:
            data = 'package com.example;\n' + data

        os.makedirs(output_path + dataset + '/' + source_lang + '/' + target_lang, exist_ok=True)
        with open(output_path + dataset + '/' + source_lang + '/' + target_lang + '/' + filename, 'w',encoding="utf-8") as file:
            file.write(data)


def clean_gpt(dataset):
    main_path = f'/scratch/cse/dual/cs5190439/MTP1/PLTranslation_data/gpt4_exp/output_t7_noid_full/gpt-4/{dataset}'
    output_path = '/scratch/cse/dual/cs5190439/MTP1/PLTranslation_data/gpt4_exp/output_t7_noid/gpt-4/'

    files = list_files(main_path)

    for f in files:
        print(f)
        splitted = f.split('/')
        filename = splitted[-1].strip()
        target_lang = splitted[-2].strip()
        source_lang = splitted[-3].strip()
        #specific to java cleaning
        # if (target_lang!="Java"):
        #     continue

        if (target_lang=="__pycache__"):
            continue
        with open(f, 'r',encoding="utf-8") as file:
            data = file.read()

        # b/w -substring
        # last_idx =data.find(f"```{target_lang.lower()}")

        # gpt-4 ->
        last_idx =data.lower().find(f"```{target_lang.lower()} response")
        if (last_idx!=-1):
            data = data[last_idx+len(f"```{target_lang} response"):]
        last_idx =data.lower().find(f"```{target_lang.lower()}")
        if (last_idx!=-1):
            data = data[last_idx+len(f"```{target_lang}"):]
        # data = data.replace(f"```{target_lang.lower()}", "").replace("```", "")

        '''
        # simple vanilla-
        # clean_possible
        last_index=find_smallest_index(data)
        if (last_index==-1):
            if (target_lang!="Python"):# else last } in java code
                last_index =data.rfind("}")+1
                data=data[:last_index]
        else:
            data=data[:last_index]
        '''
        # valid_lines = []
        # for line in data.split('\n'):
        #     # Start-End due to pseudocode part
        #     if line.strip() in ["'''","```",'End','C Code:', 'C++ Code:', 'Java Code:', 'Python Code:', 'Go Code:', '"""'] or line.strip().startswith("Input") or line.strip().startswith("Output"):
        #         break
        #     elif line.strip()=='Start':
        #         continue
        #     else:
        #         valid_lines.append(line)

        # icl-


        valid_lines = []
        for line in data.split('\n'):
            # Start-End due to pseudocode part
            if line.strip() in ["'''","```",'End','C Code:', 'C++ Code:', 'Java Code:', 'Python Code:', 'Go Code:', '"""'] or line.strip().startswith("Input") or line.strip().startswith("Output"):
                break
            elif line.strip()==target_lang:
                continue
            else:
                valid_lines.append(line)
        
        data = '\n'.join(valid_lines)

        if target_lang == 'Java':
            # data = re.sub('public\s*class\s*.+', 'public class ' + filename.split('.')[0] + ' {', data)
            data= re.sub(r'public\s*class\s*[^{]+', r'public class ' + filename.split('.')[0], data)


        if target_lang == 'Java' and dataset == 'evalplus' and 'package com.example;' not in data:
            data = 'package com.example;\n' + data

        os.makedirs(output_path + dataset + '/' + source_lang + '/' + target_lang, exist_ok=True)
        with open(output_path + dataset + '/' + source_lang + '/' + target_lang + '/' + filename, 'w',encoding="utf-8") as file:
            file.write(data)


def main(args):
    clean_new_models(args.dataset)
    return

    if args.model == 'CodeGeeX':
        clean_codegeex(args.dataset)
    elif args.model == 'StarCoder':
        clean_starcoder(args.dataset)
    elif args.model == 'LLaMa':
        clean_llama(args.dataset)
    elif args.model == 'CodeGen':
        clean_codegen(args.dataset)
    elif args.model == 'TB-Airoboros':
        clean_airoboros(args.dataset)
    elif args.model == 'TB-Vicuna':
        clean_vicuna(args.dataset)
    elif args.model =="gpt":
        clean_gpt(args.dataset)
    elif args.model =="gpt4":
        clean_gpt4(args.dataset)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='clean open-source model generations given a dataset and a model')
    parser.add_argument('--source_lang', help='source language to use for code translation. should be one of [Python,Java,C,C++,Go]', required=True, type=str)
    parser.add_argument('--target_lang', help='target language to use for code translation. should be one of [Python,Java,C,C++,Go]', required=True, type=str)
    parser.add_argument('--model', help='model to use for code translation.', required=True, type=str)
    # parser.add_argument('--report_dir', help='path to directory to store report', required=True, type=str)
    parser.add_argument('--approach', help='approach info', required=True, type=str)
    parser.add_argument('--dataset', help='dataset info', required=True, type=str)

    args = parser.parse_args()
    main(args)

'''
python3 /home/cse/dual/cs5190439/MTP1/codetlingua/clean_generations.py --source_lang Python --target_lang Java --model starcoder --approach autocot2d --dataset avatar
granite-20b-code-instruct
starcoder
granite-8b-code-instruct
CodeLlama-13b-Instruct-hf
'''