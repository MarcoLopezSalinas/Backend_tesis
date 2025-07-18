import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from geopy.distance import geodesic
import re

EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

LIST_COLUMNS = [
    'palabras_clave',
    'tipo_lugar',
    'transporte_opciones',
    'actividades_principales',
    'opciones_comida',
    'epoca_recomendada'
]

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, delimiter=';')
    # fill descripcion_corta if it exists
    if 'descripcion_corta' in df.columns:
        df['descripcion_corta'] = df['descripcion_corta'].fillna('')
    else:
        df['descripcion_corta'] = ''
    for col in LIST_COLUMNS:
        df[col] = df[col].apply(
            lambda x: [item.strip() for item in str(x).split(',')] if pd.notnull(x) else []
        )
    df['url_imagen_principal'] = df['url_imagen_principal'].fillna('xnxx.com')
    df['url_imagen_sec'] = df['url_imagen_sec'].fillna('rubias19.com')
    from .config import costo_mapping
    df['rango_costo_numerico'] = df['rango_costo'].apply(
        lambda x: costo_mapping.get(str(x).split(' ')[0], 0)
    )
    df['texto_para_embedding'] = df.apply(
        lambda row: (
            f"{row.get('descripcion_larga','')} [SEP] "
            f"Palabras clave: {', '.join(row.get('palabras_clave',[]))}. [SEP] "
            f"Actividades: {', '.join(row.get('actividades_principales',[]))}. [SEP] "
            f"Tipo de lugar: {', '.join(row.get('tipo_lugar',[]))}"
        ),
        axis=1
    )
    return df

def get_embedding(text: str) -> np.ndarray:
    if not isinstance(text, str) or text.strip() == '':
        return np.zeros(EMBEDDING_MODEL.get_sentence_embedding_dimension())
    return EMBEDDING_MODEL.encode(text, convert_to_numpy=True)

def calculate_cosine_similarity(e1: np.ndarray, e2: np.ndarray) -> float:
    if np.all(e1 == 0) or np.all(e2 == 0):
        return 0.0
    return float(cosine_similarity([e1], [e2])[0][0])

def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return float(geodesic((lat1, lon1), (lat2, lon2)).km)

def greedy_tsp_route(start_coords: tuple, places: list) -> tuple:
    current = start_coords
    route = []
    total_dist = 0.0
    remaining = places.copy()
    while remaining:
        dists = [
            calculate_haversine_distance(current[0], current[1], p['latitud'], p['longitud'])
            for p in remaining
        ]
        idx = int(np.argmin(dists))
        next_place = remaining.pop(idx)
        total_dist += dists[idx]
        route.append(next_place)
        current = (next_place['latitud'], next_place['longitud'])
    return route, total_dist

def detect_explicit_place_mentions(text: str, place_dict: dict) -> list:
    found = []
    txt = text.lower()
    for pid, name in place_dict.items():
        if re.search(rf"\b{re.escape(name.lower())}\b", txt):
            found.append(pid)
    return found
