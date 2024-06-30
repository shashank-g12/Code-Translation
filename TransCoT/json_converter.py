import json
import argparse
def main(args):
    # Replace 'path' with the actual path to your JSON file
    path=f"/scratch/cse/dual/cs5190439/MTP1/PLTranslation_data/codetlingua/{args.approach}/{args.dataset}/{args.model}/{args.source_lang}/{args.target_lang}/temperature_0/reports/{args.model}_{args.dataset}_errors_from_{args.source_lang}_to_{args.target_lang}_1.json"
    # path=f"/scratch/cse/dual/cs5190439/MTP1/PLTranslation_data/reports_autocot2d/StarCoder_codenet_errors_from_Java_to_Python_1.json"
    # Open the JSON file for reading
    with open(path, 'r') as file:
        # Load the JSON data
        data = json.load(file)

    path_=path[:-5]+"_json.txt"
    with open(path_, 'w') as f:
        for d in data:
            print(d, file=f)
            for i in data[d]:
                for j in i:
                    print(j,file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='execute avatar tests')
    parser.add_argument('--source_lang', help='source language to use for code translation. should be one of [Python,Java,C,C++,Go]', required=True, type=str)
    parser.add_argument('--target_lang', help='target language to use for code translation. should be one of [Python,Java,C,C++,Go]', required=True, type=str)
    parser.add_argument('--model', help='model to use for code translation.', required=True, type=str)
    parser.add_argument('--dataset', help='dataset info', required=True, type=str)
    # parser.add_argument('--report_dir', help='path to directory to store report', required=True, type=str)
    parser.add_argument('--approach', help='approach info', required=True, type=str)
    args = parser.parse_args()

    main(args)

'''
python3 /home/cse/dual/cs5190439/MTP1/json_converter.py --source_lang Python --target_lang Java --model starcoder --dataset codenet --approach program
granite-20b-code-instruct
starcoder
granite-8b-code-instruct
vanilla
'''