import os
import re
import argparse
import subprocess
import json
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
import networkx as nx

DATASET = Path('/home/codetrans/dataset')

class Evaluate:
    
    def __init__(self, args):
        os.makedirs(f'{args.report_dir}', exist_ok=True)
        logging.basicConfig(filename=f"{args.report_dir}/evaluation.log", level=logging.INFO, format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        
        self.get_source_call_graphs(args)
        self.get_target_call_graphs(args)
        self.evaluate(args)
        
    def evaluate(self, args):
        
        def get_idx(x, categories):
            l,r = -1, len(categories)

            while (l+1<r):
                mid = (l+r)//2
                if categories[mid]<=x:
                    l = mid
                elif categories[mid]>x:
                    r = mid
            return l
        
        def check(x, mean, std):
            return x<100
        
        logging.info('starting evaluation...')
        if args.source_lang == 'Java':
            problems = [f.split('.')[0] for f in os.listdir(str(self.source_dataset)) if f.endswith('.java')]
        
        if args.source_lang == 'Python':
            problems = [f.split('.')[0] for f in os.listdir(str(self.source_dataset)) if f.endswith('.py')]
        
        report_text = Path(args.report_dir) / f'report_{args.model}_{args.dataset}_{args.name}_{args.source_lang}_{args.target_lang}.txt'
        
        categories = [1, 2, 3, 4, 5]
        graph_ed = defaultdict(list)
        results = defaultdict(dict)
        cost_list = []
        threshold = 50
        
        for problem_id in problems:
            logging.info(f'problem: {problem_id}')
            if problem_id in self.error_source: continue
            if args.source_lang == 'Java':
                java_graph = self.get_graph_java(problem_id, args)
                python_graph = self.get_graph_python(problem_id, args)
            
            if args.source_lang =='Python':
                java_graph = self.get_graph_java_p2j(problem_id, args)
                python_graph = self.get_graph_python_p2j(problem_id, args)
            # if java_graph is None or python_graph is None: continue
            
            if args.source_lang == 'Java':
                num_source_nodes, paths, cost = self.edit_distance(java_graph, python_graph)
            
            if args.source_lang == 'Python':
                num_source_nodes, paths, cost = self.edit_distance(python_graph, java_graph)
            
            idx = get_idx(num_source_nodes, categories)
            graph_ed[categories[idx]].append(cost)
            results[problem_id] = {'num_source_nodes':num_source_nodes , 'paths':paths, 'cost':cost}
            cost_list.append(cost)
        
        
        # mean = np.mean(np.array(cost_list))
        # std = np.std(np.array(cost_list))
        
        # print(mean)
        # print(std)
        
        with open(report_text, 'w', encoding='utf-8') as report:
            
            total_ed = 0
            count = 0
            for k,v in results.items():
                # if not check(v['cost'], mean, std):
                #     self.hallucination += 1
                #     continue
                total_ed += min(threshold, v['cost'])
                count += 1
                
            report.writelines(f'Average Graph Edit Distance: {total_ed/count}\n')
            report.writelines(f'Categories: {categories}\n')
            for n in categories:
                if len(graph_ed[n])>0:
                    ed = 0
                    t = 0
                    for cost_ in graph_ed[n]:
                        # if not check(cost_, mean, std): continue
                        ed += min(threshold, cost_)
                        t += 1
                    if t > 0: report.writelines(f'{n}: {ed/t}\n')
            
            report.writelines(f'Errors in source language call graph generations: {self.error_source_cg}\n')
            report.writelines(f'Errors in target language call graph generations: {self.error_target_cg}\n')
            # report.writelines(f'Hallucinations: {self.hallucination}\n')
            report.writelines("=================================================================================================\n")
            
            for k,v in results.items():
                report.writelines(f'{k}: \n')
                for a,b in v.items():
                    report.writelines(f'{a}: {b}\n')

    def edit_distance(self, source_graph, target_graph):

        def nsc(a,b):
            if a['name']==b['name']:
                return 0
            else:
                return int(1e5)


        def esc(a,b):
            if a['name']==b['name']:
                return 0
            else:
                return int(1e5)

        G1 = nx.DiGraph()
        G2 = nx.DiGraph()

        n_source = {}
        for n in source_graph[0]:
            if n not in n_source:
                n_source[n] = len(n_source)


        for k,v in n_source.items():
            G1.add_nodes_from([(v, {'name':k})])

        for e in source_graph[1]:
            G1.add_edges_from([(n_source[e[0]], n_source[e[1]], {'name':e[0]+' '+e[1]})])


        n_target = {}
        for n in target_graph[0]:
            if n not in n_target:
                n_target[n] = len(n_target)

        for k,v in n_target.items():
            G2.add_nodes_from([(v, {'name':k})])

        for e in target_graph[1]:
            G2.add_edges_from([(n_target[e[0]], n_target[e[1]], {'name':e[0]+' '+e[1]})])


        paths, cost = nx.optimal_edit_paths(G2, G1, node_subst_cost=nsc, edge_subst_cost=esc)
        num_source_nodes = len(n_source)
        return num_source_nodes, paths, cost
        
        
    def get_source_call_graphs(self, args):
        self.error_source = set()
        if args.source_lang=='Java':
            self.source_lang_cg = Path(args.report_dir) / 'source_lang_cg'
            self.source_dataset = DATASET / args.dataset / args.source_lang / 'Code'
            self.error_source_cg = 0
            
            logging.info('generating source language call graphs...')
            self.source_lang_cg.mkdir(parents=True, exist_ok=True)
            
            for file in self.source_dataset.iterdir():
                if not str(file).endswith('.java'): continue
                try:
                    subprocess.run(['/bin/bash', 'gen_java_cg.sh', str(file), str(self.source_lang_cg) , str(args.dataset)], timeout=20)
                    
                except Exception as e:
                    self.error_source_cg += 1
                    self.error_source.add(str(file).split('/')[-1].split('.')[0])
                    logging.info(f'error generating call graph for {file}')
            
            logging.info('done')
    
        if args.source_lang=='Python':
            self.source_lang_cg = Path(args.report_dir) / 'source_lang_cg'
            self.source_dataset = DATASET / args.dataset / args.source_lang / 'Code'
            self.error_source_cg = 0
            
            logging.info('generating source language call graphs...')
            self.source_lang_cg.mkdir(parents=True, exist_ok=True)
            
            for file in self.source_dataset.iterdir():
                if not str(file).endswith('.py'): continue
                try:
                    subprocess.run(['/bin/bash', 'gen_python_cg.sh', str(file), str(self.source_lang_cg)], timeout=20)
                    
                except Exception as e:
                    self.error_source_cg += 1
                    self.error_source.add(str(file).split('/')[-1].split('.')[0])
                    logging.info(f'error generating call graph for {file}')
            
            logging.info('done')
    
    
    def get_target_call_graphs(self, args):
        if args.target_lang=='Java':
            self.target_lang_cg = Path(args.report_dir) / 'target_lang_cg'
            self.target_lang_dir = Path(args.target_translation_dir)
            self.error_target_cg = 0
            
            logging.info('generating target language call graphs...')
            self.target_lang_cg.mkdir(parents=True, exist_ok=True)
            
            for file in self.target_lang_dir.iterdir():
                if not str(file).endswith('.java'): continue
                try:
                    subprocess.run(['/bin/bash', 'gen_java_cg.sh', str(file), str(self.target_lang_cg), str("avatar")], timeout=20)
                
                except Exception as e:
                    self.error_target_cg += 1
                    logging.info(f'error generating call graph for {file}')
            
            logging.info('done')
        
        if args.target_lang=='Python':
            self.target_lang_cg = Path(args.report_dir) / 'target_lang_cg'
            self.target_lang_dir = Path(args.target_translation_dir)
            self.error_target_cg = 0
            
            logging.info('generating target language call graphs...')
            self.target_lang_cg.mkdir(parents=True, exist_ok=True)
            
            for file in self.target_lang_dir.iterdir():
                if not str(file).endswith('.py'): continue
                try:
                    subprocess.run(['/bin/bash', 'gen_python_cg.sh', str(file), str(self.target_lang_cg)], timeout=20)
                
                except Exception as e:
                    self.error_target_cg += 1
                    logging.info(f'error generating call graph for {file}')
            
            logging.info('done')
    
    def get_graph_java(self, problem_id, args):
        
        data = None
        file_path_cg = None
        file_path_code = None
        if args.source_lang=='Java':
            file_path_code = self.source_dataset / f'{problem_id}.java'
            file_path_cg = self.source_lang_cg / f'{problem_id}.txt'
        elif args.source_lang=='Python':
            file_path_code = self.target_lang_dir / f'{problem_id}.java'
            file_path_cg = self.target_lang_cg / f'{problem_id}.txt'
            
        try:
            with open(str(file_path_cg), 'r') as f:
                data = f.readlines()
        except Exception as e:
            logging.info(f'error in reading file {file_path_cg}')
            return [],[]
        
        code_as_str = file_path_code.read_text(encoding='utf-8').replace('_', '').lower()
            
        nodes = set()
        edges = set()
        for line in data:
            if line.startswith('C:'): continue
            u,v = line.split()
            u = u[2:].split('(')[0]
            v = v[3:].split('(')[0]

            if u.startswith(f'{problem_id}'): u = u.replace(f'{problem_id}','Main')
            if v.startswith(f'{problem_id}'): v = v.replace(f'{problem_id}','Main')

            u = u.replace('$', '.')
            v = v.replace('$', '.')
            u = u.replace(':', '.')
            v = v.replace(':', '.')
            u = u.replace('<', '')
            v = v.replace('<', '')
            u = u.replace('>', '')
            v = v.replace('>', '')
            u = u.replace('_', '')
            v = v.replace('_', '')
            u = u.lower()
            v = v.lower()

            if not (u.startswith('java.'))  and not ('init' in u) and all([i in code_as_str for i in u.split('.')]):
                nodes.add(u)
            if not (v.startswith('java.'))  and not ('init' in v) and all([i in code_as_str for i in v.split('.')]):
                nodes.add(v)

            #edge check
            u,v = line.split()
            u = u[2:].split('(')[0]
            v = v[3:].split('(')[0]

            if u.startswith(f'{problem_id}'): u = u.replace(f'{problem_id}','Main')
            if v.startswith(f'{problem_id}'): v = v.replace(f'{problem_id}','Main')

            if u.startswith('java.') or v.startswith('java.') or 'init' in u or 'init' in v: continue
            u = u.replace('$', '.')
            v = v.replace('$', '.')
            u = u.replace(':', '.')
            v = v.replace(':', '.')
            u = u.replace('<', '')
            v = v.replace('<', '')
            u = u.replace('>', '')
            v = v.replace('>', '')
            u = u.replace('_', '')
            v = v.replace('_', '')
            u = u.lower()
            v = v.lower()
            
            if all([i in code_as_str for i in u.split('.')]) and all([i in code_as_str for i in v.split('.')]):
                edges.add(u+' '+ v)
        return [n for n in nodes], [[e.split()[0], e.split()[1]] for e in edges]
    
    def get_graph_python(self, problem_id, args):

        def get_keywords(file_path_code):
            keywords = ['<builtin>', 'typing', 'itertools', 'functools', 'abc', 're', 'collections', 'sys', 'types'] 

            #get all imports
            data = None
            try:
                data = file_path_code.read_text(encoding='utf-8')
            except Exception as e:
                logging.info(f'error in reading file {file_path_code}')
                return []
                
            imports = re.findall('import (.*)', data)
            for imps in imports:
                if ',' in imps:
                    keywords += [i.strip() for i in imps.split(',')]
                else:
                    keywords.append(imps.strip())
            return list(set(keywords))
        
        data = None
        code_as_str = None
        file_path_cg = None
        file_path_code = None
        if args.source_lang=='Java':
            file_path_code = self.target_lang_dir / f'{problem_id}.py'
            file_path_cg = self.target_lang_cg / f'{problem_id}.json'
        elif args.source_lang=='Python':
            file_path_code = self.source_dataset / f'{problem_id}.py'
            file_path_cg = self.source_lang_cg / f'{problem_id}.json'

        keywords = get_keywords(file_path_code)

        try:
            with open(str(file_path_cg), 'r') as f:
                data = json.load(f)
        except Exception as e:
            logging.info(f'error in reading file {file_path_cg}')
            return [],[]
        
        code_as_str = file_path_code.read_text(encoding='utf-8').replace('_','').lower()
        
        nodes = set()
        edges = set()

        for key in data.keys():
            if not (key==problem_id or any([key.startswith(word) for word in keywords])):
                u = key
                if u.startswith(f'{problem_id}.'): u = u.replace(f'{problem_id}.','')
                u = u.replace('_', '').lower()
                
                if all([i in code_as_str for i in u.split('.')]) and not 'init' in u:
                    nodes.add(u)

            for item in data[key]:
                if not (item==problem_id or any([item.startswith(word) for word in keywords])):
                    v = item
                    if v.startswith(f'{problem_id}.'): v = v.replace(f'{problem_id}.','')
                    v = v.replace('_', '').lower()
                    
                    if all([i in code_as_str for i in v.split('.')]) and not 'init' in v:
                        nodes.add(v)

            #edge check
            if key==problem_id or any([key.startswith(word) for word in keywords]):
                continue

            for item in data[key]:
                if item==problem_id or any([item.startswith(word) for word in keywords]):
                    continue
                u = key
                v = item
                if u.startswith(f'{problem_id}.'): u = u.replace(f'{problem_id}.','')
                if v.startswith(f'{problem_id}.'): v = v.replace(f'{problem_id}.','')
                u = u.replace('_', '').lower()
                v = v.replace('_','').lower()
                
                if (not all([i in code_as_str for i in u.split('.')])) or 'init' in u: continue
                if (not all([i in code_as_str for i in v.split('.')])) or 'init' in v: continue
                
                edges.add(u+' '+ v)
        return [n for n in nodes], [[e.split()[0], e.split()[1]] for e in edges]


    def get_graph_java_p2j(self, problem_id, args):
        data = None
        file_path_cg = None
        file_path_code = None
        
        if args.source_lang=='Java':
            file_path_code = self.source_dataset / f'{problem_id}.java'
            file_path_cg = self.source_lang_cg / f'{problem_id}.txt'
        elif args.source_lang=='Python':
            file_path_code = self.target_lang_dir / f'{problem_id}.java'
            file_path_cg = self.target_lang_cg / f'{problem_id}.txt'
                
        try:
            with open(str(file_path_cg), 'r') as f:
                data = f.readlines()
        except Exception as e:
            return [],[]
            
        code_as_str = file_path_code.read_text(encoding='utf-8').replace('_', '').lower()
        nodes = set()
        edges = set()
        
        for line in data:
            if line.startswith('C:'): continue
            u,v = line.split()
            u = u[2:].split('(')[0]
            v = v[3:].split('(')[0]

            u = u.replace('$', '.')
            v = v.replace('$', '.')
            u = u.replace(':', '.')
            v = v.replace(':', '.')
            u = u.replace('<', '')
            v = v.replace('<', '')
            u = u.replace('>', '')
            v = v.replace('>', '')
            u = u.replace('_', '')
            v = v.replace('_', '')
            u = u.lower()
            v = v.lower()

            if not (u.startswith('java.'))  and not ('init' in u) and all([i in code_as_str for i in u.split('.')]):
                nodes.add(u)
            if not (v.startswith('java.'))  and not ('init' in v) and all([i in code_as_str for i in v.split('.')]):
                nodes.add(v)

            #edge check
            u,v = line.split()
            u = u[2:].split('(')[0]
            v = v[3:].split('(')[0]

            if u.startswith('java.') or v.startswith('java.') or 'init' in u or 'init' in v: continue
            u = u.replace('$', '.')
            v = v.replace('$', '.')
            u = u.replace(':', '.')
            v = v.replace(':', '.')
            u = u.replace('<', '')
            v = v.replace('<', '')
            u = u.replace('>', '')
            v = v.replace('>', '')
            u = u.replace('_', '')
            v = v.replace('_', '')
            u = u.lower()
            v = v.lower()
                
            if all([i in code_as_str for i in u.split('.')]) and all([i in code_as_str for i in v.split('.')]):
                edges.add(u+' '+ v)
        return [n for n in nodes], [[e.split()[0], e.split()[1]] for e in edges]
    
    def get_graph_python_p2j(self, problem_id, args):

        def get_keywords(file_path_code):
            keywords = ['<builtin>', 'typing', 'itertools', 'functools', 'abc', 're', 'collections', 'sys', 'types'] 

            #get all imports
            data = None
            try:
                data = file_path_code.read_text(encoding='utf-8')
            except Exception as e:
                return []
                    
            imports = re.findall('import (.*)', data)
            for imps in imports:
                if ',' in imps:
                    keywords += [i.strip() for i in imps.split(',')]
                else:
                    keywords.append(imps.strip())
            return list(set(keywords))
        
        data = None
        code_as_str = None
        file_path_cg = None
        file_path_code = None
        
        if args.source_lang=='Java':
            file_path_code = self.target_lang_dir / f'{problem_id}.py'
            file_path_cg = self.target_lang_cg / f'{problem_id}.json'
        elif args.source_lang=='Python':
            file_path_code = self.source_dataset / f'{problem_id}.py'
            file_path_cg = self.source_lang_cg / f'{problem_id}.json'
        
        keywords = get_keywords(file_path_code)
        try:
            with open(str(file_path_cg), 'r') as f:
                data = json.load(f)
        except Exception as e:
            return [],[]
        
        code_as_str = file_path_code.read_text(encoding='utf-8').replace('_','').lower()
        prob = problem_id.replace('_','').lower()
            
        nodes = set()
        edges = set()

        for key in data.keys(): 
            if not any([key.startswith(word) for word in keywords]):
                u = key
                if u in problem_id: u = f'{problem_id}.main'
                u = u.replace('_', '').lower()
                if all([(i in code_as_str or i in prob or i in 'main') for i in u.split('.')]) and not 'init' in u:
                    nodes.add(u)
            
            
            for item in data[key]:
                if not any([item.startswith(word) for word in keywords]):
                    v = item
                    if v in problem_id: v = f'{problem_id}.main'
                    v = v.replace('_', '').lower()
                    if all([(i in code_as_str or i in prob or i in 'main') for i in v.split('.')]) and not 'init' in v:
                        nodes.add(v)
            
            if any([key.startswith(word) for word in keywords]):
                continue
            

            for item in data[key]:
                if any([item.startswith(word) for word in keywords]): continue
                
                u = key
                v = item
                
                if u in problem_id: u = f'{problem_id}.main'
                if v in problem_id: v = f'{problem_id}.main'

                u = u.replace('_', '').lower()
                v = v.replace('_','').lower()
                    
                if (not all([(i in code_as_str or i in prob or i in 'main') for i in u.split('.')])) or 'init' in u: continue
                if (not all([(i in code_as_str or i in prob or i in 'main') for i in v.split('.')])) or 'init' in v: continue
                    
                edges.add(u+' '+ v)
        return [n for n in nodes], [[e.split()[0], e.split()[1]] for e in edges]
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='evaluate translation quality of the cot approach')
    parser.add_argument('--model', help='model used for code translation', required=True, type=str)
    parser.add_argument('--dataset', help='dataset used for translation [codenet, avatar]', required=True, type=str)
    parser.add_argument('--name', help='name describing translation technique', required=True, type=str)
    parser.add_argument('--source-lang', help='Source language of the translation [Java, Python]', required=True, type=str)
    parser.add_argument('--target-lang', help='Target language of the translation [Java, Python]', required=True, type=str)
    parser.add_argument('--target-translation-dir', help='Generated translation directory', required=True, type=str)
    parser.add_argument('--report-dir', help='Output directory of the results' , required=True, type=str)
    args = parser.parse_args()
    
    evaluate = Evaluate(args)

#python qualityEval.py --model StarCoder --dataset avatar --name vanilla --source-lang Java --target-lang Python --target-translationd-dir . --report-dir ,


# python qualityEval.py --model StarCoder --dataset avatar --name vanilla --source-lang Java --target-lang Python --target-translation-dir /home/codetrans/codetlingua/vanilla/avatar/starcoder/Java/Python/temperature_0 --report-dir /home/codetrans/callGraphEval/report
