import numpy as np
import scipy.sparse as sp
def adjacency_matrix():

    grid1 = np.array([[125, 126, 112, 103, 31 ,28, 11, 8], 
    				  [123, 124, 110, 102, 29, 26, 9, 5], 
    				  [121, 122, 109, 101, 27, 19, 18, 4],
        			  [119, 120, 108, 100, 25, 15, 12, 6], 
    				  [117, 118, 107, 99,  23, 13, 10, 3],
    				  [115, 116, 106, 97,  21, 20, 7, 2],
        			  [113, 114, 105, 98,  17, 24, 14, 0],
        			  [127, 111, 104, 96,  30, 22, 16, 1],  ])

    grid2 = np.array([
    [62, 51, 43, 35, 94, 87, 79, 78],
    [60, 53, 41, 33, 95, 86, 77, 76],
    [63, 54, 47, 44, 93, 84, 75, 74],
    [58, 55, 48, 40, 92, 85, 73, 72],
    [59, 45, 46, 38, 91, 82, 71, 70],
    [61, 49, 42, 36, 90, 83, 69, 68],
    [56, 52, 39, 34, 89, 81, 67, 66],
    [57, 50, 37, 32, 88, 80, 65, 64], ])
    cnt=0
    grids = [grid1,grid2]  
    for gr in grids:
        grid = gr
        rows_idx, cols_idx = grid.shape
        rows, cols, data = [], [], []

        # Desplazamientos de los 8 vecinos (como en una imagen)
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),          (0, 1),
                     (1, -1),  (1, 0), (1, 1)]

        for i in range(rows_idx):
            for j in range(cols_idx):
                node = grid[i, j]
                for di, dj in neighbors:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows_idx and 0 <= nj < cols_idx:
                        rows.append(node)
                        cols.append(grid[ni, nj])
                        data.append(1)
        if(cnt==0):
            A1 = sp.coo_matrix((data, (rows, cols)), shape=(128, 128))
            cnt = cnt+1
        elif(cnt==1):
            A2 =  sp.coo_matrix((data, (rows, cols)), shape=(128, 128))
            cnt=0
            
    A_global = sp.coo_matrix((128, 128))  # matriz vacía
     # Sumar las dos matrices en la global
    A_global = A_global + A1 + A2     
    

     
    return A_global
