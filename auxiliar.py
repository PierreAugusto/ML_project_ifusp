# Importing all the necessary libraries
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.coordinates import match_coordinates_sky
import math as math
from astropy.io import fits
from astropy.table import Table
import os
import seaborn as sns
from IPython.display import clear_output
import astropy.units as u
from sklearn.neighbors import KernelDensity,KNeighborsClassifier
import os
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, confusion_matrix, matthews_corrcoef, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, RandomizedSearchCV, GridSearchCV, StratifiedKFold, train_test_split
import pickle
import missingno as msno
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.cluster import KMeans
import subprocess
import joblib
import requests
from PIL import Image
import matplotlib.patches as patches
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer
from collections import Counter


pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", None)

SEED = 333
np.random.seed(SEED)

positions = ['X_IMAGE', 'Y_IMAGE', 'X_WORLD', 'Y_WORLD']

mags = ['U', 'F378', 'F395', 'F410', 'F430', 'G', 'F515', 'R', 'F660', 'I', 'F861', 'Z']

mags_petro = ["F378_PETRO", "F395_PETRO", "F410_PETRO", "F430_PETRO", "F515_PETRO", "F660_PETRO", "F861_PETRO", "Z_PETRO", "G_PETRO", "R_PETRO", "I_PETRO", "U_PETRO"]

mags_auto = ["U_AUTO", "F378_AUTO", "F395_AUTO", "F410_AUTO", "F430_AUTO", "G_AUTO", "F515_AUTO", "R_AUTO", "F660_AUTO", "I_AUTO", "F861_AUTO", "Z_AUTO"]

mags_aper_3 = ["U_APER_3", "F378_APER_3", "F395_APER_3", "F410_APER_3", "F430_APER_3", "G_APER_3", "F515_APER_3", "R_APER_3", "F660_APER_3", "I_APER_3", "F861_APER_3", "Z_APER_3"]

mags_aper_6 = ["U_APER_6", "F378_APER_6", "F395_APER_6", "F410_APER_6", "F430_APER_6", "G_APER_6", "F515_APER_6", "R_APER_6", "F660_APER_6", "I_APER_6", "F861_APER_6", "Z_APER_6"]

mags_iso = ["U_ISO", "F378_ISO", "F395_ISO", "F410_ISO", "F430_ISO", "G_ISO", "F515_ISO", "R_ISO", "F660_ISO", "I_ISO", "F861_ISO", "Z_ISO"]

##############################

flux_auto = ["FLUX_AUTO_U", "FLUX_AUTO_F378", "FLUX_AUTO_F395", "FLUX_AUTO_F410", "FLUX_AUTO_F430", "FLUX_AUTO_G", "FLUX_AUTO_F515", "FLUX_AUTO_R", "FLUX_AUTO_F660", "FLUX_AUTO_I", "FLUX_AUTO_F861", "FLUX_AUTO_Z"]

flux_iso = ["FLUX_ISO_U", "FLUX_ISO_F378", "FLUX_ISO_F395", "FLUX_ISO_F410", "FLUX_ISO_F430", "FLUX_ISO_G", "FLUX_ISO_F515", "FLUX_ISO_R", "FLUX_ISO_F660", "FLUX_ISO_I", "FLUX_ISO_F861", "FLUX_ISO_Z"]

flux_petro = ["FLUX_PETRO_U", "FLUX_PETRO_F378", "FLUX_PETRO_F395", "FLUX_PETRO_F410", "FLUX_PETRO_F430", "FLUX_PETRO_G", "FLUX_PETRO_F515", "FLUX_PETRO_R", "FLUX_PETRO_F660", "FLUX_PETRO_I", "FLUX_PETRO_F861", "FLUX_PETRO_Z"]

flux_aper_3 = ["FLUX_APER_3_U", "FLUX_APER_3_F378", "FLUX_APER_3_F395", "FLUX_APER_3_F410", "FLUX_APER_3_F430", "FLUX_APER_3_G", "FLUX_APER_3_F515", "FLUX_APER_3_R", "FLUX_APER_3_F660", "FLUX_APER_3_I", "FLUX_APER_3_F861", "FLUX_APER_3_Z"]

flux_aper_6 = ["FLUX_APER_6_U", "FLUX_APER_6_F378", "FLUX_APER_6_F395", "FLUX_APER_6_F410", "FLUX_APER_6_F430", "FLUX_APER_6_G", "FLUX_APER_6_F515", "FLUX_APER_6_R", "FLUX_APER_6_F660", "FLUX_APER_6_I", "FLUX_APER_6_F861", "FLUX_APER_6_Z"]


################################################################

flux_err_auto = ["FLUXERR_AUTO_U", "FLUXERR_AUTO_F378", "FLUXERR_AUTO_F395", "FLUXERR_AUTO_F410", "FLUXERR_AUTO_F430", "FLUXERR_AUTO_G", "FLUXERR_AUTO_F515", "FLUXERR_AUTO_R", "FLUXERR_AUTO_F660", "FLUXERR_AUTO_I", "FLUXERR_AUTO_F861", "FLUXERR_AUTO_Z"]

flux_err_iso = ["FLUXERR_ISO_U", "FLUXERR_ISO_F378", "FLUXERR_ISO_F395", "FLUXERR_ISO_F410", "FLUXERR_ISO_F430", "FLUXERR_ISO_G", "FLUXERR_ISO_F515", "FLUXERR_ISO_R", "FLUXERR_ISO_F660", "FLUXERR_ISO_I", "FLUXERR_ISO_F861", "FLUXERR_ISO_Z"]

flux_err_petro = ["FLUXERR_PETRO_U", "FLUXERR_PETRO_F378", "FLUXERR_PETRO_F395", "FLUXERR_PETRO_F410", "FLUXERR_PETRO_F430", "FLUXERR_PETRO_G", "FLUXERR_PETRO_F515", "FLUXERR_PETRO_R", "FLUXERR_PETRO_F660", "FLUXERR_PETRO_I", "FLUXERR_PETRO_F861", "FLUXERR_PETRO_Z"]

flux_err_aper_3 = ["FLUXERR_APER_3_U", "FLUXERR_APER_3_F378", "FLUXERR_APER_3_F395", "FLUXERR_APER_3_F410", "FLUXERR_APER_3_F430", "FLUXERR_APER_3_G", "FLUXERR_APER_3_F515", "FLUXERR_APER_3_R", "FLUXERR_APER_3_F660", "FLUXERR_APER_3_I", "FLUXERR_APER_3_F861", "FLUXERR_APER_3_Z"]

flux_err_aper_6 = ["FLUXERR_APER_6_U", "FLUXERR_APER_6_F378", "FLUXERR_APER_6_F395", "FLUXERR_APER_6_F410", "FLUXERR_APER_6_F430", "FLUXERR_APER_6_G", "FLUXERR_APER_6_F515", "FLUXERR_APER_6_R", "FLUXERR_APER_6_F660", "FLUXERR_APER_6_I", "FLUXERR_APER_6_F861", "FLUXERR_APER_6_Z"]


################################################################

mags_e_auto = ["e_U_AUTO", "e_F378_AUTO", "e_F395_AUTO", "e_F410_AUTO", "e_F430_AUTO", "e_G_AUTO", "e_F515_AUTO", "e_R_AUTO", "e_F660_AUTO", "e_I_AUTO", "e_F861_AUTO", "e_Z_AUTO"]

mags_e_iso = ["e_U_ISO", "e_F378_ISO", "e_F395_ISO", "e_F410_ISO", "e_F430_ISO", "e_G_ISO", "e_F515_ISO", "e_R_ISO", "e_F660_ISO", "e_I_ISO", "e_F861_ISO", "e_Z_ISO"]

mags_e_petro = ["e_U_PETRO", "e_F378_PETRO", "e_F395_PETRO", "e_F410_PETRO", "e_F430_PETRO", "e_G_PETRO", "e_F515_PETRO", "e_R_PETRO", "e_F660_PETRO", "e_I_PETRO", "e_F861_PETRO", "e_Z_PETRO"]

mags_e_aper_3 = ["e_U_APER_3", "e_F378_APER_3", "e_F395_APER_3", "e_F410_APER_3", "e_F430_APER_3", "e_G_APER_3", "e_F515_APER_3", "e_R_APER_3", "e_F660_APER_3", "e_I_APER_3", "e_F861_APER_3", "e_Z_APER_3"]

mags_e_aper_6 = ["e_U_APER_6", "e_F378_APER_6", "e_F395_APER_6", "e_F410_APER_6", "e_F430_APER_6", "e_G_APER_6", "e_F515_APER_6", "e_R_APER_6", "e_F660_APER_6", "e_I_APER_6", "e_F861_APER_6", "e_Z_APER_6"]


################################################################

mags_petro_c = ["F378_PETRO_c", "F395_PETRO_c", "F410_PETRO_c", "F430_PETRO_c", "F515_PETRO_c", "F660_PETRO_c", "F861_PETRO_c", "Z_PETRO_c", "G_PETRO_c", "R_PETRO_c", "I_PETRO_c", "U_PETRO_c"]

mags_auto_c = ["U_AUTO_c", "F378_AUTO_c", "F395_AUTO_c", "F410_AUTO_c", "F430_AUTO_c", "G_AUTO_c", "F515_AUTO_c", "R_AUTO_c", "F660_AUTO_c", "I_AUTO_c", "F861_AUTO_c", "Z_AUTO_c"]

mags_aper_6_c = ["U_APER_6_c", "F378_APER_6_c", "F395_APER_6_c", "F410_APER_6_c", "F430_APER_6_c", "G_APER_6_c", "F515_APER_6_c", "R_APER_6_c", "F660_APER_6_c", "I_APER_6_c", "F861_APER_6_c", "Z_APER_6_c"]

################################################################


class_stars = ["CLASS_STAR_U", "CLASS_STAR_F378", "CLASS_STAR_F395", "CLASS_STAR_F410", "CLASS_STAR_F430", "CLASS_STAR_G", "CLASS_STAR_F515", "CLASS_STAR_R", "CLASS_STAR_F660", "CLASS_STAR_I", "CLASS_STAR_F861", "CLASS_STAR_Z"]

isoarea_world = ["ISOAREA_WORLD_U", "ISOAREA_WORLD_F378", "ISOAREA_WORLD_F395", "ISOAREA_WORLD_F410", "ISOAREA_WORLD_F430", "ISOAREA_WORLD_G", "ISOAREA_WORLD_F515", "ISOAREA_WORLD_R", "ISOAREA_WORLD_F660", "ISOAREA_WORLD_I", "ISOAREA_WORLD_F861", "ISOAREA_WORLD_Z"]

isoarea_image = ["ISOAREA_IMAGE_U", "ISOAREA_IMAGE_F378", "ISOAREA_IMAGE_F395", "ISOAREA_IMAGE_F410", "ISOAREA_IMAGE_F430", "ISOAREA_IMAGE_G", "ISOAREA_IMAGE_F515", "ISOAREA_IMAGE_R", "ISOAREA_IMAGE_F660", "ISOAREA_IMAGE_I", "ISOAREA_IMAGE_F861", "ISOAREA_IMAGE_Z"]

flux_max = ["FLUX_MAX_U", "FLUX_MAX_F378", "FLUX_MAX_F395", "FLUX_MAX_F410", "FLUX_MAX_F430", "FLUX_MAX_G", "FLUX_MAX_F515", "FLUX_MAX_R", "FLUX_MAX_F660", "FLUX_MAX_I", "FLUX_MAX_F861", "FLUX_MAX_Z"]

snr_win = ["SNR_WIN_U", "SNR_WIN_F378", "SNR_WIN_F395", "SNR_WIN_F410", "SNR_WIN_F430", "SNR_WIN_G", "SNR_WIN_F515", "SNR_WIN_R", "SNR_WIN_F660", "SNR_WIN_I", "SNR_WIN_F861", "SNR_WIN_Z"]

treshold = ["THRESHOLD_U", "THRESHOLD_F378", "THRESHOLD_F395", "THRESHOLD_F410", "THRESHOLD_F430", "THRESHOLD_G", "THRESHOLD_F515", "THRESHOLD_R", "THRESHOLD_F660", "THRESHOLD_I", "THRESHOLD_F861", "THRESHOLD_Z"]

mu_treshold = ["MU_THRESHOLD_U", "MU_THRESHOLD_F378", "MU_THRESHOLD_F395", "MU_THRESHOLD_F410", "MU_THRESHOLD_F430", "MU_THRESHOLD_G", "MU_THRESHOLD_F515", "MU_THRESHOLD_R", "MU_THRESHOLD_F660", "MU_THRESHOLD_I", "MU_THRESHOLD_F861", "MU_THRESHOLD_Z"]

mu_max = ["MU_MAX_U", "MU_MAX_F378", "MU_MAX_F395", "MU_MAX_F410", "MU_MAX_F430", "MU_MAX_G", "MU_MAX_F515", "MU_MAX_R", "MU_MAX_F660", "MU_MAX_I", "MU_MAX_F861", "MU_MAX_Z"]

background = ["BACKGROUND_U", "BACKGROUND_F378", "BACKGROUND_F395", "BACKGROUND_F410", "BACKGROUND_F430", "BACKGROUND_G", "BACKGROUND_F515", "BACKGROUND_R", "BACKGROUND_F660", "BACKGROUND_I", "BACKGROUND_F861", "BACKGROUND_Z"]



morphology = ['ELONGATION', 'ELLIPTICITY', 'KRON_RADIUS', 'PETRO_RADIUS']

FWHM_IMAGE = ['FWHM_IMAGE_U', 'FWHM_IMAGE_F378', 'FWHM_IMAGE_F395', 'FWHM_IMAGE_F410', 'FWHM_IMAGE_F430', 'FWHM_IMAGE_G', 'FWHM_IMAGE_F515', 'FWHM_IMAGE_R', 'FWHM_IMAGE_F660', 'FWHM_IMAGE_I', 'FWHM_IMAGE_F861', 'FWHM_IMAGE_Z']

FWHM_WORLD = ['FWHM_WORLD_U', 'FWHM_WORLD_F378', 'FWHM_WORLD_F395', 'FWHM_WORLD_F410', 'FWHM_WORLD_F430', 'FWHM_WORLD_G', 'FWHM_WORLD_F515', 'FWHM_WORLD_R', 'FWHM_WORLD_F660', 'FWHM_WORLD_I', 'FWHM_WORLD_F861', 'FWHM_WORLD_Z']

colors_r = ["U-R", "F378-R", "F395-R", "F410-R", "F430-R", "G-R", "F515-R", "R-F660", "R-I", "R-F861", "R-Z"]

colors = ["U-F378", "U-F395", "U-F410", "U-F430", "U-G", "U-F515", "U-R", "U-F660", "U-I", "U-F861", "U-Z",
          "F378-F395", "F378-F410", "F378-F430", "F378-G", "F378-F515", "F378-R", "F378-F660", "F378-I", "F378-F861", "F378-Z",
          "F395-F410", "F395-F430", "F395-G", "F395-F515", "F395-R", "F395-F660", "F395-I", "F395-F861", "F395-Z",
          "F410-F430", "F410-G", "F410-F515", "F410-R", "F410-F660", "F410-I", "F410-F861", "F410-Z",
          "F430-G", "F430-F515", "F430-R", "F430-F660", "F430-I", "F430-F861", "F430-Z",
          "G-F515", "G-R", "G-F660", "G-I", "G-F861", "G-Z",
          "F515-R", "F515-F660", "F515-I", "F515-F861", "F515-Z",
          "R-F660", "R-I", "R-F861", "R-Z",
          "F660-I", "F660-F861", "F660-Z",
          "I-F861", "I-Z",
          "F861-Z"]

flags = ['FLAGS_U', 'FLAGS_F378', 'FLAGS_F395', 'FLAGS_F410', 'FLAGS_F430', 'FLAGS_G', 'FLAGS_F515', 'FLAGS_R', 'FLAGS_F660', 'FLAGS_I', 'FLAGS_F861', 'FLAGS_Z']

Mg = ["Mg"]

flux_radius_list = [ 'flux_radius_90_70_r' ,  'flux_radius_90_50_r', 'flux_radius_90_20_r', 'flux_radius_70_50_r', 'flux_radius_70_20_r', 'flux_radius_50_20_r']

# Cortes compacto e extenso do FWHM
Fcomp = 3.
Fext = 3.5

# Cortes de magnitude
# DM = 31.51
# mglim = 22.5
# m0 = 30
# flag0 = 3
# cut_data = fr"G_APER_6_c > 13 & G_APER_6_c <= {mglim} & FLAGS_U<= {flag0} & FLAGS_F378<= {flag0} & FLAGS_F395<= {flag0} & FLAGS_F410<= {flag0} & FLAGS_F430<= {flag0} & FLAGS_G<= {flag0} & FLAGS_F515<= {flag0} & FLAGS_R<= {flag0} & FLAGS_F660<= {flag0} & FLAGS_I<= {flag0} & FLAGS_F861<= {flag0} & FLAGS_Z<= {flag0}"


def criar_reta(x_min, x_max, coef_angular, intercepto):
    """
    Cria uma reta com base nos limites de x e nos parâmetros fornecidos.

    :param x_min: Valor mínimo de x.
    :param x_max: Valor máximo de x.
    :param coef_angular: Coeficiente angular da reta (m).
    :param intercepto: Intercepto da reta (b).
    :return: Arrays de x e y representando a reta.
    """
    try:
        # Criar os valores de x
        x = np.linspace(x_min, x_max, 100)
        
        # Calcular os valores de y
        y = coef_angular * x + intercepto
        
        return x, y
    except ValueError as e:
        print(f"Erro de valor: {e}")
        return None, None


def fits_open(path, columns=None):  
    """
    Open a FITS file and convert it to a pandas DataFrame.

    :param path: Path to the FITS file.
    :param columns: List of columns to select from the FITS file.
    :return: Pandas DataFrame containing the FITS data.
    """
    with fits.open(path) as hdu:
        table = Table(hdu[1].data)
        df = table.to_pandas()
        
        # Select only the desired columns, if provided
        if columns is not None:
            df = df[columns]
        
        return df
    
    
def save_as_fits(df, output_fits):
    """
    Salva um DataFrame como um arquivo FITS.

    :param df: DataFrame a ser salvo.
    :param output_fits: Caminho para o arquivo FITS de saída.
    """
    try:
        # Converter o DataFrame para uma tabela Astropy
        table = Table.from_pandas(df)
        
        # Escrever a tabela em um arquivo FITS
        table.write(output_fits, format='fits', overwrite=True)
        
        print(f"Arquivo FITS criado com sucesso: {output_fits}")
    except Exception as e:
        print(f"Erro ao salvar DataFrame como FITS: {e}")


def make_colors(df, colors, mag_type="APER_6_c"):
    """
    Cria colunas de cores em um DataFrame com base em magnitudes especificadas.

    :param df: DataFrame de entrada contendo as magnitudes.
    :param colors: Lista de strings representando as cores a serem criadas no formato 'col1-col2'.
    :param mag_type: Sufixo a ser adicionado aos nomes das colunas de magnitudes.
    :return: DataFrame com as colunas de cores adicionadas.
    """
    try:
        # Criar as cores com o sufixo especificado
        for color in colors:
            col1, col2 = color.split("-")
            col1_mag_type = f"{col1}_{mag_type}"
            col2_mag_type = f"{col2}_{mag_type}"
            df[color] = df[col1_mag_type] - df[col2_mag_type]

        return df
    except Exception as e:
        print(f"Error creating color columns: {e}")
        return df
    
    
def create_corrected_mags(df, mags):
    for mag in mags:
        df[f"{mag}_PETRO_c"] = df[f"{mag}_PETRO"] - df[f"{mag}_ext"]
        df[f"{mag}_AUTO_c"] = df[f"{mag}_AUTO"] - df[f"{mag}_ext"]
        df[f"{mag}_APER_3_c"] = df[f"{mag}_APER_3"] - df[f"{mag}_ext"]
        df[f"{mag}_APER_6_c"] = df[f"{mag}_APER_6"] - df[f"{mag}_ext"]
        df[f"{mag}_ISO_c"] = df[f"{mag}_ISO"] - df[f"{mag}_ext"]

    return df


def match_ra_dec(df1, df2, ra_col1, dec_col1, ra_col2, dec_col2, radius=1.0):
    """
    Faz um match entre dois DataFrames pelo RA e DEC com um raio especificado.

    :param df1: Primeiro DataFrame.
    :param df2: Segundo DataFrame.
    :param ra_col1: Nome da coluna de RA no primeiro DataFrame.
    :param dec_col1: Nome da coluna de DEC no primeiro DataFrame.
    :param ra_col2: Nome da coluna de RA no segundo DataFrame.
    :param dec_col2: Nome da coluna de DEC no segundo DataFrame.
    :param radius: Raio de correspondência em arcseconds.
    :return: DataFrame com as correspondências.
    """
    # Remover entradas com NaN nas colunas de RA e DEC
    df1 = df1.dropna(subset=[ra_col1, dec_col1])
    df2 = df2.dropna(subset=[ra_col2, dec_col2])

    # Criar objetos SkyCoord para os dois DataFrames
    coords1 = SkyCoord(ra=df1[ra_col1].values * u.deg, dec=df1[dec_col1].values * u.deg)
    coords2 = SkyCoord(ra=df2[ra_col2].values * u.deg, dec=df2[dec_col2].values * u.deg)

    # Fazer a correspondência
    idx, d2d, _ = match_coordinates_sky(coords1, coords2)

    # Filtrar correspondências dentro do raio especificado
    match_mask = d2d < radius * u.arcsec

    # Criar DataFrame de correspondências
    matched_df = df1[match_mask].copy()
    matched_df['match_idx'] = idx[match_mask]
    matched_df = matched_df.join(df2.iloc[idx[match_mask]].reset_index(drop=True), rsuffix='_matched')

    return matched_df


def match_ra_dec_stilts(input_file1, input_file2, output_file, ra_col1, dec_col1, ra_col2, dec_col2, radius=1.0):
    """
    Faz um match entre dois arquivos pelo RA e DEC com um raio especificado usando o STILTS.

    :param input_file1: Caminho para o primeiro arquivo.
    :param input_file2: Caminho para o segundo arquivo.
    :param output_file: Caminho para o arquivo de saída.
    :param ra_col1: Nome da coluna de RA no primeiro arquivo.
    :param dec_col1: Nome da coluna de DEC no primeiro arquivo.
    :param ra_col2: Nome da coluna de RA no segundo arquivo.
    :param dec_col2: Nome da coluna de DEC no segundo arquivo.
    :param radius: Raio de correspondência em arcseconds.
    """
    try:
        # Comando STILTS para fazer a correspondência espacial
        command = [
            'stilts', 'tmatch2',
            f'in1={input_file1}',
            f'in2={input_file2}',
            f'out={output_file}',
            'matcher=sky',
            f'values1="{ra_col1} {dec_col1}"',
            f'values2="{ra_col2} {dec_col2}"',
            f'params={radius / 3600.0}',  # Convertendo arcseconds para degrees
            'join=1and2',
            'find=best',
            'ofmt=fits'
        ]
        
        # Executa o comando
        subprocess.run(command, check=True)
        print(f"Arquivo de correspondência criado com sucesso: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao realizar a correspondência: {e}")
        
        
def predict_and_save(df, model_path, columns_predict, save_path=None):
    """
    Faz a predição usando um modelo de machine learning e adiciona a coluna de predição ao DataFrame.

    :param df: DataFrame de entrada.
    :param model_path: Caminho para o arquivo do modelo.
    :param save_path: Caminho para salvar o DataFrame resultante (opcional).
    :return: DataFrame com a coluna de predição adicionada.
    """
    try:
        df = df[columns_predict]
        
        # Carregar o modelo
        with open(model_path, 'rb') as file:
            model= pickle.load(file)
        
        # Criar o normalizador
        scaler = MinMaxScaler()
        
        train_norm = scaler.fit_transform(df)
        
        # Fazer a predição
        predictp = model.predict_proba(df)
        
        # Armazenar as previsões de ser da classe 1
        all_predictions_prob = predictp[:, 1]
        
        df['predict_p'] = all_predictions_prob
         
        # Salvar o DataFrame resultante, se o caminho de salvamento for fornecido
        if save_path:
            save_as_fits(df, save_path)
            print(f"DataFrame com predições salvo em: {save_path}")
        
        return df
    except Exception as e:
        print(f"Erro ao fazer a predição: {e}")
        return None
    
def include_flux_radius_sub(df):
    '''
    Função que inclui as diferenças entre os raios de fluxo de 20, 50, 70 e 90%.
    '''
    df['flux_radius_90_70_r'] = df['FLUX_RADIUS_90_R'] - df['FLUX_RADIUS_70_R']
    df['flux_radius_90_50_r'] = df['FLUX_RADIUS_90_R'] - df['FLUX_RADIUS_50_R']
    df['flux_radius_90_20_r'] = df['FLUX_RADIUS_90_R'] - df['FLUX_RADIUS_20_R']
    df['flux_radius_70_50_r'] = df['FLUX_RADIUS_70_R'] - df['FLUX_RADIUS_50_R']
    df['flux_radius_70_20_r'] = df['FLUX_RADIUS_70_R'] - df['FLUX_RADIUS_20_R']
    df['flux_radius_50_20_r'] = df['FLUX_RADIUS_50_R'] - df['FLUX_RADIUS_20_R']
    return df


# Função para baixar a imagem JPEG e opcionalmente desenhar um círculo
def legacy_image_plot(ra, dec, output_dir='legacy_images', layer='ls-dr10', zoom=False, pixscale=False, size=False, draw_circle=False, circle_radius=50, name_img=None):
    
    # Construir a URL
    if zoom and size:
        url = f"https://www.legacysurvey.org/viewer/jpeg-cutout?ra={ra}&dec={dec}&zoom={zoom}&size={size}"
    
    elif size:
        url = f"https://www.legacysurvey.org/viewer/jpeg-cutout?ra={ra}&dec={dec}&size={size}"
    
    elif pixscale:
        url = f"https://www.legacysurvey.org/viewer/jpeg-cutout?ra={ra}&dec={dec}&pixscale={pixscale}"
    else:
        url = f"https://www.legacysurvey.org/viewer/jpeg-cutout?ra={ra}&dec={dec}"
    
    response = requests.get(url)

    # Verificar se a resposta é bem-sucedida
    if response.status_code == 200:
        # Criar o diretório de saída, se não existir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Nome do arquivo JPG de saída
        if name_img:
            jpg_filename = f"{output_dir}/{name_img}_{ra}_dec{dec}.jpg"
        else:
            jpg_filename = f"{output_dir}/{ra}_dec{dec}.jpg"
        
        # Salvar o arquivo JPEG
        with open(jpg_filename, 'wb') as f:
            f.write(response.content)
        # print(f"Imagem JPG salva: {jpg_filename}")

        # Se a opção de desenhar o círculo estiver ativada, chamar a função para desenhar o círculo
        if draw_circle:
            plot_with_circle(jpg_filename, output_dir, circle_radius=circle_radius)
            # Remover a imagem original (sem círculo) se desejar
            os.remove(jpg_filename)
        
        return jpg_filename
    else:
        print(f"Erro ao baixar imagem para RA={ra}, DEC={dec}")
        return None

# Função para desenhar um círculo no centro da imagem sem borda branca
def plot_with_circle(image_path, save_path, circle_radius=50):
    img = Image.open(image_path)
    
    # Configurando o plot
    fig, ax = plt.subplots()
    ax.imshow(img)
    
    # Adicionando um círculo no centro da imagem
    center_x = img.width / 2
    center_y = img.height / 2
    circle = patches.Circle((center_x, center_y), circle_radius, edgecolor='red', facecolor='none', linewidth=2)
    ax.add_patch(circle)
    
    # Remover eixos e margens
    plt.axis('off')  # Desligar eixos
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remover margens
    
    # Salvar a imagem sem borda branca
    name_img = image_path.split('/')[-1]
    circled_filename = f"{save_path}/{name_img.replace('.jpg', '_circle.jpg')}"
    plt.savefig(circled_filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Imagem com círculo salva: {circled_filename}")


# Função para converter coordenadas esféricas para projeção estereográfica
def stereographic_projection(ra, dec, ra_center, dec_center):
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    ra_center_rad = np.radians(ra_center)
    dec_center_rad = np.radians(dec_center)
    
    k = 2 / (1 + np.sin(dec_center_rad) * np.sin(dec_rad) + np.cos(dec_center_rad) * np.cos(dec_rad) * np.cos(ra_rad - ra_center_rad))
    x = k * np.cos(dec_rad) * np.sin(ra_rad - ra_center_rad)
    y = k * (np.cos(dec_center_rad) * np.sin(dec_rad) - np.sin(dec_center_rad) * np.cos(dec_rad) * np.cos(ra_rad - ra_center_rad))
    
    return np.degrees(x), np.degrees(y)


# Função para filtrar itens dentro do círculo
def itens_no_circulo(df, centro_ra, centro_dec, raio):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("A entrada 'df' deve ser um DataFrame do Pandas com colunas 'RA' e 'DEC'.")

    if 'RA' not in df.columns or 'DEC' not in df.columns:
        raise ValueError("O DataFrame deve conter as colunas 'RA' e 'DEC'.")

    # Converter graus para radianos
    ra_rad = np.radians(df['RA'])
    dec_rad = np.radians(df['DEC'])
    centro_ra_rad = np.radians(centro_ra)
    centro_dec_rad = np.radians(centro_dec)
    
    # Calcular a distância angular
    delta_ra = ra_rad - centro_ra_rad
    delta_dec = dec_rad - centro_dec_rad
    a = np.sin(delta_dec / 2)**2 + np.cos(centro_dec_rad) * np.cos(dec_rad) * np.sin(delta_ra / 2)**2
    distancia_angular = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Converter a distância angular de radianos para graus
    distancia_angular_graus = np.degrees(distancia_angular)
    
    # Filtrar os itens que estão dentro do raio
    itens_dentro = df[distancia_angular_graus <= raio]
    
    return itens_dentro


# Função para filtrar itens fora do círculo
def itens_fora_circulo(df, centro_ra, centro_dec, raio):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("A entrada 'df' deve ser um DataFrame do Pandas com colunas 'RA' e 'DEC'.")

    if 'RA' not in df.columns or 'DEC' not in df.columns:
        raise ValueError("O DataFrame deve conter as colunas 'RA' e 'DEC'.")

    # Converter graus para radianos
    ra_rad = np.radians(df['RA'])
    dec_rad = np.radians(df['DEC'])
    centro_ra_rad = np.radians(centro_ra)
    centro_dec_rad = np.radians(centro_dec)
    
    # Calcular a distância angular
    delta_ra = ra_rad - centro_ra_rad
    delta_dec = dec_rad - centro_dec_rad
    a = np.sin(delta_dec / 2)**2 + np.cos(centro_dec_rad) * np.cos(dec_rad) * np.sin(delta_ra / 2)**2
    distancia_angular = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Converter a distância angular de radianos para graus
    distancia_angular_graus = np.degrees(distancia_angular)
    
    # Filtrar os itens que estão dentro do raio
    itens_fora = df[distancia_angular_graus > raio]
    
    return itens_fora


# Cortes do gráfico de f_90_20 vs R_PETRO_c
a_1, b_1 = -4.5, 97 # Reta maior
a_2, b_2 = -0.8, 17 # Reta menorr

# Cortes do gráfico de f_90_70 vs FLUX_RADIUS_90_R
a_3, b_3 = 0.7, -0.8 # Reta maior 
a_4, b_4 = 0.3, -0.45 # Reta menor

# Cortes do gráfico de f_90_70 vs f_70_20
a_5, b_5 = 1.6, 1. # Reta maior
a_6, b_6 = 0.4, 0.05 # Reta menor

cuts_selection_candidates_1 = fr"R_PETRO_c >= 16 and R_PETRO_c <= 21 and flux_radius_90_20_r >= 2 and flux_radius_90_20_r <= {a_1} * R_PETRO_c + {b_1} and flux_radius_90_20_r >= {a_2} * R_PETRO_c + {b_2}"

cuts_selection_candidates_2 = fr"flux_radius_90_70_r >= 0 and flux_radius_90_70_r <= 4.5 and FLUX_RADIUS_90_R>= 0 and FLUX_RADIUS_90_R<=8.5 and flux_radius_90_70_r <= {a_3} * FLUX_RADIUS_90_R + {b_3} and flux_radius_90_70_r >= {a_4} * FLUX_RADIUS_90_R + {b_4}"

cuts_selection_candidates_3 = fr"flux_radius_70_20_r >= 0 and flux_radius_70_20_r <= 3.5 and flux_radius_70_20_r <= {a_5} * flux_radius_90_70_r + {b_5} and flux_radius_70_20_r >= {a_6} * flux_radius_90_70_r + {b_6}"

cuts_selection_candidates_4 = fr"FWHM_IMAGE_R <= 4.5"
# cuts_selection_candidates_4 = fr"FWHM_IMAGE_R <= 5.5"

cuts_selection_candidates_5 = fr"R_PETRO_c>=17"

# cuts_selection_candidates_6 = fr"PSS < 0.5 or PSS != PSS"

# Mudar para 0.0045 o maior
cuts_selection_candidates_6 = fr"((z < 0.0055) and (z > 0.001)) or (z != z)"

cuts_selection_candidates_7 = fr"ELLIPTICITY <= 0.2"