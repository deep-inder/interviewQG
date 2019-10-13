#!/bin/bash
cd data
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=None --execute DataPreprocess_keywords.ipynb
cd ..
onmt_translate -model embed-model.pt -src data/src-test.txt -output data/embedd_keyword.txt -replace_unk -verbose
cd data
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=None --execute Result_compile.ipynb
