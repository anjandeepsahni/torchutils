#!/bin/bash

array=( 3.6 3.7 )
condapath="${HOME}/opt/miniconda3"

# parse inputs
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
        pypiloc="https://test.pypi.io/pypi/"
        version=$2
        pkg='torchutils_test'
    fi
elif [ "$#" -ne 1 ]
then
    echo "First argument should be either \"test\" or version."
    exit -1
else
    pypiloc="https://pypi.io/pypi/"
    version=$1
    pkg='torchutils'
fi

echo "Building conda package ..."
cd ~
conda skeleton pypi $pkg --version $version --pypi-url $pypiloc

# torch is listed as pytorch on conda
cd $pkg
perl -i -pe 's/\btorch\b/pytorch/g' meta.yaml
cd ~

# building conda packages
for i in "${array[@]}"
do
	conda build --python $i $pkg
done

# convert package to other platforms
cd ~
platforms=( osx-64 linux-64 win-64)
find $condapath/conda-bld/osx-64 -name "*${version}*.tar.bz2" | while read file
do
    echo $file
    for platform in "${platforms[@]}"
    do
       conda convert --platform $platform $file  -o $condapath/conda-bld/
    done    
done

echo "Building conda package done!"

# delete pkg directory from home
rm -r $pkg