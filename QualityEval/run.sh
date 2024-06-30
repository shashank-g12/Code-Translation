#!/usr/bin/env bash

source_lang="$1"
target_lang="$2"

datasets=("avatar" "codenet")
approaches=("vanilla" "autocot2d")
models=("granite-20b-code-instruct" "granite-8b-code-instruct" "starcoder" "codellama-13b-instruct-hf")


for dataset in "${datasets[@]}"
do
    for model in "${models[@]}"
    do  
        for approach in "${approaches[@]}"
        do  
            python clean_generations.py --source_lang "${source_lang}" --target_lang "${target_lang}" --model "${model}" --approach "${approach}" --dataset "${dataset}"
            python qualityEval.py --model "${model}" --dataset "${dataset}" --name "${approach}" --source-lang "${source_lang}" --target-lang "${target_lang}" --target-translation-dir "/home/codetrans/callGraphEval/codetlingua/${approach}/${dataset}/${model}/${source_lang}/${target_lang}/temperature_0" --report-dir "/home/codetrans/callGraphEval/report_${model}_${dataset}_${approach}_${source_lang}2${target_lang}"
            rm atcoder_*
            rm codeforces_*
            rm s*
        done
    done
done