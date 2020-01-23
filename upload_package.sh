#!/bin/sh

if [ "$#" -ne 1 ] && [ "$#" -ne 2 ]
then
    echo "Usage: Minimum 1 (version) and maximum 2 (test, version) arguments supported."
    exit -1
elif [ $1 == "test" ]
then
    if [ "$#" -ne 2 ]
    then
        echo "Version not specified."
        exit -1
    else
        files="dist/torchutils_test-${2}*"
    fi
elif [ "$#" -ne 1 ]
then
    echo "First argument should be either \"test\" or version."
    exit -1
else
    files="dist/torchutils-${1}*"
fi

if [ $1 == "test" ]
then
    python3 -m twine upload --repository-url https://test.pypi.org/legacy/ $files
else
    twine upload $files
fi