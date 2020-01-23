#!/bin/bash

condapath="${HOME}/opt/miniconda3"

# upload packages to conda
echo "Uploading conda package ..."
find $condapath/conda-bld -name "*${version}*.tar.bz2" | while read file
do
    echo $filech
    anaconda upload $file
done

echo "Uploading conda package done!"