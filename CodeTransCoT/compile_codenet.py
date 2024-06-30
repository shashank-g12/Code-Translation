import os
import subprocess
import pandas as pd
from pathlib import Path
from subprocess import Popen, PIPE
import argparse

# initial directory location 
# extension
# check translation pair code 
# some rough files in folders (like pycache) while testing but considering according to extension of languages..generalize it ?
def main(args):
    print('testing translations')
    dataset = 'codenet'#granite-20b-code-instruct
    report_dir=f"/scratch/cse/dual/cs5190439/MTP1/PLTranslation_data/codetlingua/{args.approach}/codenet/{args.model}/{args.source_lang}/{args.target_lang}/temperature_0/reports"
    # translation_dir = f"/home/scai/mtech/aib222684/MTP/output_cot_orig/{args.model}/{dataset}/{args.source_lang}/{args.target_lang}"
    translation_dir= f"/scratch/cse/dual/cs5190439/MTP1/PLTranslation_data/codetlingua/{args.approach}/codenet/{args.model}/{args.source_lang}/{args.target_lang}/temperature_0"
    test_dir = f"/scratch/cse/dual/cs5190439/MTP1/PLTranslation_data/dataset/{dataset}/{args.source_lang}/TestCases"
    os.makedirs(report_dir, exist_ok=True)
    EXTENSIONS = { "C": ".c", "C++": ".cpp", "Java": ".java", "Python": ".py", "Go": ".go" }
    extn= EXTENSIONS[args.target_lang]
    files = [f  for f in os.listdir(translation_dir) if (f[(-1*len(extn)):]==extn)]#if f != '.DS_Store']
    # files=files[:100]
    compile_failed = []
    test_passed =[]
    test_failed =[]
    test_failed_details = []
    runtime_failed = []
    runtime_failed_details= []
    infinite_loop = []

    if args.target_lang =="Python":

        for i in range(len(files)):
            if (files[i]=="__pycache__"):
                continue
            try:
                print('Filename: ', files[i])
                subprocess.run("python3 -m py_compile "+translation_dir+"/"+ files[i], check=True, capture_output=True, shell=True, timeout=30)

                with open(test_dir+"/"+ files[i].split(".")[0]+"_in.txt" , 'r') as f:
                    f_in = f.read()
                f_out = open(test_dir+"/"+ files[i].split(".")[0]+"_out.txt", "r").read()

                p = Popen(['python3', translation_dir+"/"+ files[i]], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                try:
                    stdout, stderr_data = p.communicate(input=f_in.encode(), timeout=100)
                except subprocess.TimeoutExpired:
                    infinite_loop.append(files[i])
                    continue

                if(stdout.decode().strip()==f_out.strip()):
                    test_passed.append(files[i])
                else:
                    if stderr_data.decode()=='':
                        test_failed.append(files[i])
                        test_failed_details.append('Filename: '+files[i]+' Actual: '+str(f_out)+' Generated: '+ str(stdout.decode()))  
                    else:
                        runtime_failed.append(files[i])
                        runtime_failed_details.append('Filename: '+ files[i]+' Error_type: '+ str(stderr_data.decode())) 

            except Exception as e:
                compile_failed.append(files[i])

    elif args.target_lang =="Java":  # target language 

        for i in range(len(files)):

            try:
                print('Filename: ', files[i])
                subprocess.run("javac "+translation_dir+"/"+ files[i], check=True, capture_output=True, shell=True, timeout=30)

                with open(test_dir+"/"+ files[i].split(".")[0]+"_in.txt" , 'r') as f:
                    f_in = f.read()
                f_out = open(test_dir+"/"+ files[i].split(".")[0]+"_out.txt", "r").read()
                p = Popen(['java', files[i].split(".")[0]], cwd=translation_dir, stdin=PIPE, stdout=PIPE, stderr=PIPE)    

                try:
                    stdout, stderr_data = p.communicate(input=f_in.encode(), timeout=100)
                except subprocess.TimeoutExpired:
                    infinite_loop.append(files[i])
                    continue

                if(stdout.decode().strip()==f_out.strip()): # stdout is from the translated code , f_out test data from original language 
                    test_passed.append(files[i])
                else:
                    if stderr_data.decode()=='':
                        test_failed.append(files[i])
                        test_failed_details.append('Filename: '+files[i]+' Actual: '+str(f_out)+' Generated: '+ str(stdout.decode()))  
                    else:
                        runtime_failed.append(files[i])
                        runtime_failed_details.append('Filename: '+ files[i]+' Error_type: '+ str(stderr_data.decode())) 
            
            except Exception as e:
                compile_failed.append(files[i])

        #remove all .class files generated
        dir_files = os.listdir(translation_dir)
        for fil in dir_files:
            if ".class" in fil: os.remove(translation_dir +"/"+ fil)


    elif args.target_lang == "C++":

        for i in range(len(files)):

            try:
                print('Filename: ', files[i])
                subprocess.run("g++ -o exec_output -std=c++11 " + translation_dir+ "/"+ files[i], check=True, capture_output=True, shell=True)

                with open(test_dir+"/"+ files[i].split(".")[0]+"_in.txt" , 'r') as f:
                    f_in = f.read()

                f_out = open(test_dir+"/"+ files[i].split(".")[0]+"_out.txt", "r").read()
                p = Popen(['./exec_output'], cwd=os.getcwd(), stdin=PIPE, stdout=PIPE, stderr=PIPE)    

                try:
                    stdout, stderr_data = p.communicate(input=f_in.encode(), timeout=100)
                except subprocess.TimeoutExpired:
                    infinite_loop.append(files[i])
                    continue

                if(stdout.decode().strip()==f_out.strip()): # stdout is from the translated code , f_out test data from original language 
                    test_passed.append(files[i])

                else:
                    if stderr_data.decode()=='':
                        test_failed.append(files[i])
                        test_failed_details.append('Filename: '+files[i]+' Actual: '+str(f_out)+' Generated: '+ str(stdout.decode()))  
                    else:
                        runtime_failed.append(files[i])
                        runtime_failed_details.append('Filename: '+ files[i]+' Error_type: '+ str(stderr_data.decode())) 

            except Exception as e:
                compile_failed.append(files[i])

    elif args.target_lang == "C":

        for i in range(len(files)):

            try:
                print('Filename: ', files[i])
                subprocess.run("gcc "+translation_dir+"/"+ files[i], check=True, capture_output=True, shell=True, timeout=10)

                with open(test_dir+"/"+ files[i].split(".")[0]+"_in.txt" , 'r') as f:
                    f_in = f.read()
                f_out = open(test_dir+"/"+ files[i].split(".")[0]+"_out.txt", "r").read()
                p = Popen(['./a.out'], cwd=os.getcwd(), stdin=PIPE, stdout=PIPE, stderr=PIPE)    

                try:
                    stdout, stderr_data = p.communicate(input=f_in.encode(), timeout=100)
                except subprocess.TimeoutExpired:
                    infinite_loop.append(files[i])
                    continue

                if(stdout.decode().strip()==f_out.strip()): # stdout is from the translated code , f_out test data from original language 
                    test_passed.append(files[i])
                else:
                    if stderr_data.decode()=='':
                        test_failed.append(files[i])
                        test_failed_details.append('Filename: '+files[i]+' Actual: '+str(f_out)+' Generated: '+ str(stdout.decode()))  
                    else:
                        runtime_failed.append(files[i])
                        runtime_failed_details.append('Filename: '+ files[i]+' Error_type: '+ str(stderr_data.decode())) 

            except Exception as e:
                compile_failed.append(files[i])

    elif args.target_lang == "Go":

        for i in range(len(files)):

            try:
                print('Filename: ', files[i])
                subprocess.run("go build "+translation_dir+"/"+ files[i], check=True, capture_output=True, shell=True, timeout=30)
                with open(test_dir+"/"+ files[i].split(".")[0]+"_in.txt" , 'r') as f:
                    f_in = f.read()
                f_out = open(test_dir+"/"+ files[i].split(".")[0]+"_out.txt", "r").read()
                p = Popen(["./"+files[i].split(".")[0]], cwd=os.getcwd(), stdin=PIPE, stdout=PIPE, stderr=PIPE)    

                try:
                    stdout, stderr_data = p.communicate(input=f_in.encode(), timeout=100)
                except subprocess.TimeoutExpired:
                    infinite_loop.append(files[i])
                    continue

                if(stdout.decode().strip()==f_out.strip()): # stdout is from the translated code , f_out test data from original language 
                    test_passed.append(files[i])
                else:
                    if stderr_data.decode()=='':
                        test_failed.append(files[i])
                        test_failed_details.append('Filename: '+files[i]+' Actual: '+str(f_out)+' Generated: '+ str(stdout.decode()))  
                    else:
                        runtime_failed.append(files[i])
                        runtime_failed_details.append('Filename: '+ files[i]+' Error_type: '+ str(stderr_data.decode())) 
                
            except Exception as e:
                compile_failed.append(files[i])

            #remove generated files
            wd =  os.getcwd()
            if os.path.isfile(wd + "/" + files[i].split(".")[0]):
                os.remove(wd + "/" + files[i].split(".")[0])        
    
    elif args.target_lang == "Rust":

        for i in range(len(files)):

            try:
                print('Filename: ', files[i])
                bin_name = files[i].split('.')[0]

                with open(f"dataset/codenet/{args.source_lang}/TestCases/"+ files[i].split(".")[0]+"_in.txt" , 'r') as f:
                    f_in = f.read()
                
                f_out = open(f"dataset/codenet/{args.source_lang}/TestCases/"+ files[i].split(".")[0]+"_out.txt", "r").read()

                subprocess.run(f'rustc {translation_dir}/{files[i]}', check=True, capture_output=True, shell=True, timeout=30)
                p = Popen(['./ ' + bin_name], cwd=os.getcwd(), stdin=PIPE, stdout=PIPE, stderr=PIPE)

                try:
                    stdout, stderr_data = p.communicate(input=f_in.encode(), timeout=100)
                except subprocess.TimeoutExpired:
                    infinite_loop.append(files[i])
                    continue

                if stdout.decode().strip() == f_out.strip():
                    test_passed.append(files[i])
                else:
                    if stderr_data.decode() == '':
                        test_failed.append(files[i])
                        test_failed_details.append('Filename: ' + files[i] + ' Actual: ' + str(f_out) + ' Generated: ' + str(stdout.decode()))  
                    else:
                        runtime_failed.append(files[i])
                        runtime_failed_details.append('Filename: ' + files[i] + ' Error_type: ' + str(stderr_data.decode())) 
        
            except Exception as e:
                compile_failed.append(files[i])

    else:
        print("language:{} is not yet supported. select from the following languages[Python,Java,C,C++,Go,Rust,C#]".format(args.target_lang))
        return

    test_failed = list(set(test_failed))
    test_failed_details = list(set(test_failed_details))
    runtime_failed = list(set(runtime_failed))
    runtime_failed_details = list(set(runtime_failed_details))
    infinite_loop = list(set(infinite_loop))
    test_passed = list(set(test_passed))
    # compile_failed = list(set(compile_failed))
    compile_failed = list(set(files)- (set(test_failed) | set(runtime_failed) | set(infinite_loop) | set(test_passed)))

    txt_fp = Path(report_dir).joinpath(f"{args.model}_{dataset}_compileReport_from_"+str(args.source_lang)+"_to_"+str(args.target_lang)+".txt")
    with open(txt_fp, "w", encoding="utf-8") as report:
        report.writelines("Total Instances: {}\n\n".format(len(files)))#len(test_passed)+len(compile_failed)+len(runtime_failed)+len(test_failed)+len(infinite_loop)))# here union or total 
        report.writelines("Total Correct: {}\n".format(len(test_passed)))
        report.writelines("Total Runtime Failed: {}\n".format(len(runtime_failed)))
        report.writelines("Total Compilation Failed: {}\n".format(len(compile_failed)))
        report.writelines("Total Test Failed: {}\n".format(len(test_failed)))
        report.writelines("Total Infinite Loop: {}\n\n".format(len(infinite_loop)))
        acc_score =(len(test_passed)/len(files)) * 100
        print("Accuracy:",acc_score)
        report.writelines("Accuracy: {}\n".format(acc_score))
        report.writelines("Runtime Rate: {}\n".format((len(runtime_failed)/len(files)) * 100))
        report.writelines("Compilation Rate: {}\n".format((len(compile_failed)/len(files)) * 100))
        report.writelines("Test Failed Rate: {}\n".format((len(test_failed)/len(files)) * 100))
        report.writelines("Infinite Loop Rate: {}\n\n".format((len(infinite_loop)/len(files)) * 100))
        report.writelines("=================================================================================================\n")
        report.writelines("Failed Test Files: {} \n".format(test_failed))
        report.writelines("Failed Test Details: {} \n".format(test_failed_details))
        report.writelines("=================================================================================================\n")
        report.writelines("Runtime Error Files: {} \n".format(runtime_failed))
        report.writelines("Runtime Error Details: {} \n".format(runtime_failed_details))
        report.writelines("=================================================================================================\n")
        report.writelines("Compilation Error Files: {} \n".format(compile_failed))
        report.writelines("=================================================================================================\n")    
        report.writelines("Infinite Loop Files: {} \n".format(infinite_loop))
        report.writelines("=================================================================================================\n")
    df = pd.DataFrame(columns=['Source Language', 'Target Language', 'Filename', 'BugType', 'RootCause', 'Impact', 'Comments'])
    index = 0
    for i in range(0, len(compile_failed)):
        list_row = [args.source_lang, args.target_lang, compile_failed[i], "", "", "Compilation Error", ""]
        df.loc[i] = list_row
        index+=1
    for i in range(0, len(runtime_failed)):
        list_row = [args.source_lang, args.target_lang, runtime_failed[i], "", "", "Runtime Error", ""]
        df.loc[index] = list_row
        index+=1 
    for i in range(0, len(test_failed)):
        list_row = [args.source_lang, args.target_lang, test_failed[i], "", "", "Test Failed", ""]
        df.loc[index] = list_row
        index+=1

    excel_fp = Path(report_dir).joinpath(f"{args.model}_{dataset}_compileReport_from_"+str(args.source_lang)+"_to_"+str(args.target_lang)+".xlsx")
    df.to_excel(excel_fp, sheet_name='Sheet1')

    ordered_unsuccessful_fp = Path(report_dir).joinpath(f"{args.model}_{dataset}_compileReport_from_"+str(args.source_lang)+"_to_"+str(args.target_lang)+"_ordered_unsuccessful.txt")
    with open(ordered_unsuccessful_fp, 'w') as f:
        for unsuccessful_instance in compile_failed + runtime_failed + test_failed + infinite_loop:
            f.write(f"{unsuccessful_instance}\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='execute codenet tests')
    parser.add_argument('--source_lang', help='source language to use for code translation. should be one of [Python,Java,C,C++,Go]', required=True, type=str)
    parser.add_argument('--target_lang', help='target language to use for code translation. should be one of [Python,Java,C,C++,Go]', required=True, type=str)
    parser.add_argument('--model', help='model to use for code translation.', required=True, type=str)
    # parser.add_argument('--report_dir', help='path to directory to store report', required=True, type=str)
    parser.add_argument('--approach', help='approach info', required=True, type=str)
    args = parser.parse_args()

    main(args)
'''
# info-
# python3 /home/cse/dual/cs5190439/MTP1/PLTranslationEmpirical/src/test/compile_codenet.py --source_lang Python --target_lang Java --model StarCoder --report_dir /scratch/cse/dual/cs5190439/MTP1/PLTranslation_data/reports
# python3 /home/cse/dual/cs5190439/MTP1/PLTranslationEmpirical/src/test/compile_codenet_feedback.py --source_lang Java --target_lang Python --model StarCoder --report_dir /scratch/cse/dual/cs5190439/MTP1/PLTranslation_data/reports --attempt 2
#
# report_dir- /scratch/cse/dual/cs5190439/MTP1/PLTranslation_data/reports
# -StarCoder_codenet_compileReport_from_Java_to_Python.txt  StarCoder_codenet_compileReport_from_Java_to_Python.xlsx  StarCoder_codenet_compileReport_from_Java_to_Python_ordered_unsuccessful.txt

python3 /home/cse/dual/cs5190439/MTP1/codetlingua/compile_codenet.py --source_lang Python --target_lang Java --model codellama-13b-instruct-hf --approach vanilla
granite-20b-code-instruct
starcoder
granite-8b-code-instruct
vanilla

'''