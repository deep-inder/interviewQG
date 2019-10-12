import pandas as pd
import sys
from sklearn.model_selection import train_test_split
args = sys.argv
df = pd.read_csv("followML_extra_ratings_copy.csv", header=None)
train_df, test_df = train_test_split(df, test_size = 0.2, random_state = 0)
if (args[1] == 'a'):
    train_df_bert = pd.DataFrame({
        'id':range(len(train_df)),
        'label':train_df[4],
        'text_a':train_df[1].replace(r'\n', ' ', regex=True),
        'text_b': train_df[2].replace(r'\n', ' ', regex=True)
    })

    test_df_bert = pd.DataFrame({
        'id':range(len(test_df)),
        'label':test_df[4],
        'text_a':test_df[1].replace(r'\n', ' ', regex=True),
        'text_b':test_df[2].replace(r'\n', ' ', regex=True)
    })
else:
    train_df_bert = pd.DataFrame({
        'id':range(len(train_df)),
        'label':train_df[4],
        'text_a':train_df[0].replace(r'\n', ' ', regex=True),
        'text_b': train_df[2].replace(r'\n', ' ', regex=True)
    })

    test_df_bert = pd.DataFrame({
        'id':range(len(test_df)),
        'label':test_df[4],
        'text_a':test_df[0].replace(r'\n', ' ', regex=True),
        'text_b':test_df[2].replace(r'\n', ' ', regex=True)
    })
train_df_bert.to_csv('train.tsv', sep='\t', index=False, header=False)
test_df_bert.to_csv('test.tsv', sep='\t', index=False, header=False)