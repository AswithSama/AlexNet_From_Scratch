#!/bin/bash
# Track files > 50MB with Git LFS automatically

find . -type f -size +50M | while read FILE
do
    git lfs track "$FILE"
done

git add .gitattributes
