def check_dup_rows(x):
    print(len(x))
    x1 = x.copy(deep=True)
    print(len(x.drop_duplicates()))
    return None

def check_col(df,s):
    return len(set(df[s].values))
