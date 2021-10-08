def get_value_counts_for_column(data, col_name):
    counts_df = data.groupBy(col_name).count().orderBy('count', ascending=False)
    
    return counts_df.toPandas()


def create_vocab(data, col_name, cutoff=5, unk_token=True, none_token=True):
    
    val_counts = get_value_counts_for_column(data, col_name)
    
    val_counts.columns = [f"{col_name}_token", "count"]
    
    final_vocab_df = val_counts[val_counts['count'] >= cutoff].copy()
    
    if unk_token & none_token:
        token_list = ["[UNK]"] + ["[NONE]"] + list(final_vocab_df[f"{col_name}_token"])
    elif unk_token:
        token_list = ["[UNK]"] + list(final_vocab_df[f"{col_name}_token"])
    elif none_token:
        token_list = ["[NONE]"] + list(final_vocab_df[f"{col_name}_token"])
    else:
        token_list = list(final_vocab_df[f"{col_name}_token"])
        
    index_list = list(range(1, len(token_list)+1))
    
    final_vocab = dict(zip(token_list, index_list))
    
    return final_vocab
    
    