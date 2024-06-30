#!/usr/bin/env bash

cp $1 ./
arrIN=(${1//// })
file=${arrIN[-1]}
arrname=(${file//./ })

pycg --package . --max-iter 0 "${file}" -o "${arrname[0]}.json"

mv "./${arrname[0]}.json" "${2}"
rm "${file}"