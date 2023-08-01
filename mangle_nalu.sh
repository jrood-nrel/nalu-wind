#!/bin/bash

find . -type f -exec sed -i.bak 's/HYPRE/NALU_HYPRE/g' {} \;
find . -name \*.bak | xargs rm
git clean -df
find . -type f -exec sed -i.bak 's/hypre/nalu_hypre/g' {} \;
find . -name \*.bak | xargs rm
git clean -df
find . -type f -exec sed -i.bak 's/[[:<:]]HYPRE[[:>:]]/NALU_HYPRE/g' {} \;
find . -name \*.bak | xargs rm
git clean -df
find . -type f -exec sed -i.bak 's/NALU_NALU/NALU/g' {} \;
find . -name \*.bak | xargs rm
git clean -df
find . -type f -exec sed -i.bak 's/nalu/nalu/g' {} \;
find . -name \*.bak | xargs rm
git clean -df
find . -type f -exec sed -i.bak 's/libHYPRE/libNALU_HYPRE/g' {} \;
find . -name \*.bak | xargs rm
git clean -df
git mv cmake/FindHYPRE.cmake cmake/FindNALU_HYPRE.cmake
