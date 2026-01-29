import numpy as np
import pandas as pd

def read_ground_points(path):
    df = pd.read_csv(path, sep=';', header=None, names=['id', 'X', 'Y', 'Z'])
    df['id'] = df['id'].astype(str).str.strip()
    return df

def read_image_pixels(path):
    df = pd.read_csv(path, sep=';', header=None, names=['id', 'col', 'row'])
    df['id'] = df['id'].astype(str).str.strip()
    return df

def pixels_to_photo_mm(df_pix, pixel_mm=0.008, ncol=4500, nrow=3000, cx_mm=0.0894, cy_mm=0.2168,
                       k1=-9.413e-5, k2=1.091e-7):
    df = df_pix.copy()
    x_mm = (df['col'].to_numpy() - ncol/2) * pixel_mm
    y_mm = (df['row'].to_numpy() - nrow/2) * (-pixel_mm)
    x = x_mm - cx_mm
    y = y_mm - cy_mm
    r2 = x*x + y*y
    factor = k1*r2 + k2*(r2*r2)
    df['x_photo'] = x * (1 - factor)
    df['y_photo'] = y * (1 - factor)
    return df

def fit_affine_ls(df, x='x_photo', y='y_photo', X='X', Y='Y'):
    xx = df[x].to_numpy()
    yy = df[y].to_numpy()
    XX = df[X].to_numpy()
    YY = df[Y].to_numpy()

    n = len(df)
    A = np.zeros((2*n, 6))
    L = np.zeros((2*n,))

    A[0::2, 0] = xx
    A[0::2, 1] = yy
    A[0::2, 2] = 1
    L[0::2] = XX

    A[1::2, 3] = xx
    A[1::2, 4] = yy
    A[1::2, 5] = 1
    L[1::2] = YY

    p, *_ = np.linalg.lstsq(A, L, rcond=None)
    return dict(a=p[0], b=p[1], c=p[2], d=p[3], e=p[4], f=p[5])

ground_path  = '/Data/ground.txt'
gcp_pix_path = '/Data/gcp_pix.txt'

df_ground = read_ground_points(ground_path)
df_pix    = read_image_pixels(gcp_pix_path)
df_photo  = pixels_to_photo_mm(df_pix)

df_gcp = pd.merge(df_ground, df_photo, on='id', how='inner')

params = fit_affine_ls(df_gcp)
print(params)
