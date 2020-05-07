import numpy.random as Random
import numpy as np
import pandas as pd

def rank_swapping(df, p, columns):
    # sort in ascending
    df_ = df
    df_.sort_values(by=columns)
    print("RankSwapping with p = " + str(p))
    if p == 0:
        print("Not Swapping :)")
        return df_
        
    for j in range(0, len(columns)): # for each attribute (Column)
        length = df_.shape[0]
        #print(length)
        col_values = df_.iloc[:,j].to_numpy()
        for i in range(0, length): # for each row
            rand_i = int(Random.uniform(-p,p))
            #col_values[(i, rand_i)] = col_values[(rand_i, i)] # swap valuestemp = df_[i, j]
            temp = col_values[i]
            col_values[i] = col_values[(i + rand_i)%length]
            col_values[(i + rand_i)%length] = temp
        df_.iloc[:,j] = col_values # place swapped values
        print("swapped " + str( int((j+1)/len(columns)*100)) +"%")
    return df_


if __name__ == "__main__":
    data = "./data/data_for_student_case.csv"
    df1 = pd.read_csv(data)
    numeric_cols = ['bin', 'amount']
    rank_swapping(df1, 3, numeric_cols)