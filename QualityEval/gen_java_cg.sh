#!/usr/bin/env bash

if [ "$3" == "avatar" ]; then
    cp $1 ./
    arrIN=(${1//// })
    file=${arrIN[-1]}
    arrname=(${file//./ })

    javac "${file}"
    jar cf "${arrname[0]}.jar" ./*.class
    java -jar "javacg-0.1-SNAPSHOT-static.jar" "${arrname[0]}.jar" > "${arrname[0]}.txt"

    mv "./${arrname[0]}.txt" "${2}"
    rm "${file}"
    rm ./*.class
    rm "${arrname[0]}.jar"

else 
    cp $1 ./
    arrIN=(${1//// })
    file=${arrIN[-1]}
    arrname=(${file//./ })

    mv "${file}" "Main.java"

    javac "Main.java"
    jar cf "Main.jar" ./*.class
    java -jar "javacg-0.1-SNAPSHOT-static.jar" "Main.jar" > "${arrname[0]}.txt"

    mv "./${arrname[0]}.txt" "${2}"
    rm "Main.java"
    rm ./*.class
    rm "Main.jar"
fi