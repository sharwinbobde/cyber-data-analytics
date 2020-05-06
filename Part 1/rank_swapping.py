import numpy.random as Random
import numpy as np

def rank_swapping(df, p, columns):
    # sort in ascending
    df_ = df.copy() 
    df_.sort_values(by=columns)
    df_ = df_.to_numpy(dtype=np.float)
    print("RankSwapping with p = " + str(p))
    if p == 0:
        return df_
    for col in range(0, len(columns)): # for each attribute (Column)
        #print("col = " + str(col+1)+"/"+str(len(columns)))
        length = np.shape(df_)[0]
        for row in range(0, length): # for each row
            rand = int(Random.uniform(-p,p))
            temp = df_[row, col]
            df_[row, col] = df_[(row + rand)%length, col]
            df_[(row + rand)%length, col] = temp 
    return df_