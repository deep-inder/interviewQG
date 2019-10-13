#!/bin/bash
cd data
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=None --execute DataPreprocess_standard.ipynb
cd ..
onmt_translate -model squad-model.pt -src data/src-test.txt -output data/squad.txt -replace_unk -verbose
cd data
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=None --execute Result_compile.ipynb
