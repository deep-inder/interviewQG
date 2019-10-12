#!/bin/bash
cd data
python formatter.py q
cd ..
python train.py
cd outputs/QG_Classification
tar cvzf QG.tar.gz *.*n  
cd ../../
cp outputs/QG_Classification/QG.tar.gz cache/
python evaluate.py
