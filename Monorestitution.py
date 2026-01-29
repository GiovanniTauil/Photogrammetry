import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol

def read_image_points_txt(path, sep=';'):
    """
    Lê arquivo no formato: id;coluna;linha
    Retorna DataFrame com colunas: id, col, row
    """
    df = pd.read_csv(path, sep=sep, header=None, names=['id', 'col', 'row'])
    df['id'] = df['id'].astype(str).str.strip()
    return df

def pixels_to_photo_mm(df_pix, *,
                       pixel_mm=0.008,
                       ncol=4500, nrow=3000,
                       cx_mm=0.0894, cy_mm=0.2168,
                       k1=-9.413e-5, k2=1.091e-7):
    """
    Converte pixel (col,row) -> (x_photo,y_photo) em mm no sistema fotogramétrico,
    com origem no centro da imagem, y positivo para cima, centrado no ponto principal
    e corrigido de distorção radial.
    """
    df = df_pix.copy()

    # Pixel -> mm (origem no centro da imagem)
    x_mm = (df['col'].to_numpy() - ncol/2.0) * pixel_mm
    y_mm = (df['row'].to_numpy() - nrow/2.0) * (-pixel_mm)

    # Centralizar no ponto principal
    x = x_mm - cx_mm
    y = y_mm - cy_mm

    # Distorção radial
    r2 = x*x + y*y
    factor = (k1 * r2) + (k2 * r2 * r2)  # k1*r^2 + k2*r^4

    df['x_photo'] = x * (1.0 - factor)
    df['y_photo'] = y * (1.0 - factor)

    return df

def rotation_matrix_from_opk(omega_deg, phi_deg, kappa_deg):
    """
    Monta matriz R = Rz(kappa) * Ry(phi) * Rx(omega) (convenção comum em fotogrametria).
    Ângulos em graus.
    """
    om = np.deg2rad(omega_deg)
    ph = np.deg2rad(phi_deg)
    ka = np.deg2rad(kappa_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(om), np.sin(om)],
        [0, -np.sin(om), np.cos(om)]
    ])

    Ry = np.array([
        [np.cos(ph), 0, -np.sin(ph)],
        [0, 1, 0],
        [np.sin(ph), 0, np.cos(ph)]
    ])

    Rz = np.array([
        [np.cos(ka), np.sin(ka), 0],
        [-np.sin(ka), np.cos(ka), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx

def inverse_collinearity_xy(x_photo_mm, y_photo_mm, Z_ground_m,
                            Xc_m, Yc_m, Zc_m,
                            R, f_mm):
    """
    Calcula (X,Y) no terreno a partir de (x_photo, y_photo) e Z conhecido,
    usando colinearidade inversa.
    """
    # Numeradores / denominador
    num_x = R[0,0]*x_photo_mm + R[1,0]*y_photo_mm - R[2,0]*f_mm
    num_y = R[0,1]*x_photo_mm + R[1,1]*y_photo_mm - R[2,1]*f_mm
    den   = R[0,2]*x_photo_mm + R[1,2]*y_photo_mm - R[2,2]*f_mm

    # Evitar divisão por zero
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)

    X = Xc_m + (Z_ground_m - Zc_m) * (num_x / den)
    Y = Yc_m + (Z_ground_m - Zc_m) * (num_y / den)
    return X, Y

def initial_ground_xy_from_Z0(df_photo, Z0,
                              Xc, Yc, Zc,
                              R, f_mm):
    """
    Calcula (X_ground, Y_ground) inicial assumindo plano Z=Z0 para todos os pontos.
    """
    df = df_photo.copy()
    Xg, Yg = inverse_collinearity_xy(
        df['x_photo'].to_numpy(),
        df['y_photo'].to_numpy(),
        Z0,
        Xc, Yc, Zc,
        R, f_mm
    )
    df['X_ground'] = Xg
    df['Y_ground'] = Yg
    return df

def sample_dem_Z(dem_data, dem_transform, X, Y):
    """
    Amostra Z no DEM (nearest-neighbor) para coordenadas X,Y.
    Faz clamp para dentro do raster.
    """
    r, c = rowcol(dem_transform, X, Y)
    r = int(np.clip(r, 0, dem_data.shape[0]-1))
    c = int(np.clip(c, 0, dem_data.shape[1]-1))
    return float(dem_data[r, c])

def refine_points_with_dem(df_photo_xy, dem_data, dem_transform,
                           Xc, Yc, Zc, R, f_mm,
                           max_iter=100, tol_xy=0.1):
    """
    Para cada ponto:
      - inicia com X_ground, Y_ground (já existentes no df)
      - amostra Z no DEM
      - recalcula X,Y com colinearidade inversa usando esse Z
      - repete até convergir
    Retorna DataFrame com X_refined, Y_refined, Z_refined, iters.
    """
    df = df_photo_xy.copy()

    X_ref = []
    Y_ref = []
    Z_ref = []
    iters = []

    x_photo = df['x_photo'].to_numpy()
    y_photo = df['y_photo'].to_numpy()

    for i in range(len(df)):
        X_cur = float(df.loc[df.index[i], 'X_ground'])
        Y_cur = float(df.loc[df.index[i], 'Y_ground'])

        delta = np.inf
        k = 0
        Z_dem = np.nan

        while delta > tol_xy and k < max_iter:
            Z_dem = sample_dem_Z(dem_data, dem_transform, X_cur, Y_cur)

            X_new, Y_new = inverse_collinearity_xy(
                x_photo[i], y_photo[i], Z_dem,
                Xc, Yc, Zc,
                R, f_mm
            )

            delta = float(np.hypot(X_new - X_cur, Y_new - Y_cur))
            X_cur, Y_cur = float(X_new), float(Y_new)
            k += 1

        X_ref.append(X_cur)
        Y_ref.append(Y_cur)
        Z_ref.append(Z_dem)
        iters.append(k)

    df['X_refined'] = X_ref
    df['Y_refined'] = Y_ref
    df['Z_refined'] = Z_ref
    df['iters'] = iters
    return df

def monorestitution_pipeline(points_path, dem_path,
                             *,
                             Xc, Yc, Zc,
                             omega, phi, kappa,
                             f_mm=34.145, cx_mm=0.0894, cy_mm=0.2168,
                             k1=-9.413e-5, k2=1.091e-7,
                             pixel_mm=0.008, ncol=4500, nrow=3000,
                             Z0=0.0,
                             max_iter=100, tol_xy=0.1):
    """
    Retorna DataFrame final com:
    id, col, row, x_photo, y_photo, X_ground, Y_ground, X_refined, Y_refined, Z_refined, iters
    """
    # 1) pontos (pixel)
    df_pix = read_image_points_txt(points_path)

    # 2) pixel -> mm fotogramétrico corrigido
    df_photo = pixels_to_photo_mm(
        df_pix,
        pixel_mm=pixel_mm, ncol=ncol, nrow=nrow,
        cx_mm=cx_mm, cy_mm=cy_mm, k1=k1, k2=k2
    )

    # 3) rotação
    R = rotation_matrix_from_opk(omega, phi, kappa)

    # 4) chute inicial com Z0
    df_xy0 = initial_ground_xy_from_Z0(df_photo, Z0, Xc, Yc, Zc, R, f_mm)

    # 5) DEM
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
        dem_transform = src.transform
        dem_crs = src.crs

    # 6) refino
    df_final = refine_points_with_dem(
        df_xy0, dem_data, dem_transform,
        Xc, Yc, Zc, R, f_mm,
        max_iter=max_iter, tol_xy=tol_xy
    )

    return df_final, R, dem_crs

Xc, Yc, Zc = 583805.7090, 7225363.1282, 1993.3668
omega, phi, kappa = 1.4036, 0.4421, -163.6311

points_path = '/content/coordenadas_f_054_quadras.txt'
dem_path    = '/content/DEM_NuvemPtsLASER.tif'

df_mono, R, crs = monorestitution_pipeline(
    points_path, dem_path,
    Xc=Xc, Yc=Yc, Zc=Zc,
    omega=omega, phi=phi, kappa=kappa,
    Z0=950.0,          # exemplo: cota aproximada da área
    tol_xy=0.1,
    max_iter=100
)

display(df_mono.head())
print("CRS do DEM:", crs)

