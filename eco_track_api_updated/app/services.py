import itertools
import numpy as np
import pandas as pd
from fastapi import HTTPException
from .schemas import (
    RecommendationRequest, ItineraryResponse, ItineraryItem,
    RoutePlace, DesgloseItem
)
from .utils import (
    load_data, get_embedding, detect_explicit_place_mentions,
    greedy_tsp_route, calculate_cosine_similarity,
    calculate_haversine_distance
)
from .config import config, costo_mapping

# Precomputación
DF = load_data('data/lugares_turisticos_temp.csv')
DF['embedding_descripcion'] = DF['texto_para_embedding'].apply(get_embedding)
unique_tipos = set(sum(DF['tipo_lugar'], []))
unique_acts  = set(sum(DF['actividades_principales'], []))
EMB_TIPOS = {t: get_embedding(t) for t in unique_tipos}
EMB_ACTS  = {a: get_embedding(a) for a in unique_acts}

def score_places(df: pd.DataFrame, prefs: dict, start: tuple) -> pd.DataFrame:
    rows = []
    budget = costo_mapping.get(prefs['presupuesto'], 0)
    emb_det = get_embedding(prefs['detalles_adicionales'])
    emb_tipo = get_embedding(prefs['tipo_viaje'])
    emb_acts = [get_embedding(a) for a in prefs['actividades']]
    for _, r in df.iterrows():
        data = r.to_dict()
        s_tipo = max(
            [calculate_cosine_similarity(emb_tipo, EMB_TIPOS.get(t, np.zeros_like(emb_tipo))) for t in r['tipo_lugar']],
            default=0
        )
        s_acts = max(
            [calculate_cosine_similarity(ua, EMB_ACTS.get(act, np.zeros_like(ua))) for ua in emb_acts for act in r['actividades_principales']],
            default=0
        )
        s_det = calculate_cosine_similarity(emb_det, r['embedding_descripcion'])
        score = (
            config['pesos_lugar']['semantica_tipo'] * s_tipo +
            config['pesos_lugar']['semantica_actividades'] * s_acts +
            config['pesos_lugar']['bert_detalles'] * s_det
        )
        if r['rango_costo_numerico'] <= budget:
            score += config['pesos_lugar']['presupuesto']
        elif r['rango_costo_numerico'] == budget + 1:
            score += config['pesos_lugar']['presupuesto'] * 0.05
        dist = calculate_haversine_distance(
            start[0], start[1], r['latitud'], r['longitud']
        )
        data.update({
            'puntuacion_relevancia': score,
            'dist_a_punto_partida': dist,
            'similitud_bert_detalles': s_det,
            'max_sim_tipo': s_tipo,
            'max_sim_actividad': s_acts
        })
        rows.append(data)
    return pd.DataFrame(rows)

def generate_itineraries_combos(scored: pd.DataFrame, requested_ids: list) -> list:
    req_df = scored[scored['ID'].isin(requested_ids)]
    others = scored[~scored['ID'].isin(requested_ids)].sort_values(
        'puntuacion_relevancia', ascending=False
    ).head(config['max_candidatos_inicial'])
    combos = []
    for n in range(config['num_lugares_min'], config['num_lugares_max'] + 1):
        for combo in itertools.combinations(
            others.to_dict('records'), n - len(req_df)
        ):
            combos.append(list(req_df.to_dict('records')) + list(combo))
    return combos

def get_itineraries(req: RecommendationRequest) -> ItineraryResponse:
    try:
        start = (req.user_location.lat, req.user_location.lon)
        place_map = dict(zip(DF['ID'], DF['nombre_lugar']))
        requested_ids = detect_explicit_place_mentions(req.detalles_adicionales, place_map)
        scored = score_places(DF, req.dict(), start)
        combos = generate_itineraries_combos(scored, requested_ids)[:config['max_itinerarios_crudos']]

        items = []
        for idx, combo in enumerate(combos):
            route, dist = greedy_tsp_route(start, combo)
            rel_score = sum(p['puntuacion_relevancia'] for p in route)

            ruta = [
                RoutePlace(
                    nombre_lugar=p['nombre_lugar'],
                    relevancia=p['puntuacion_relevancia'],
                    url_imagen_principal=p.get('url_imagen_principal',''),
                    url_imagen_secundaria=p.get('url_imagen_sec',''),
                    descripcion_corta=p.get('descripcion_corta','')
                )
                for p in route
            ]

            desglose = []
            for i, p in enumerate(route, start=1):
                reasons = []
                if p['ID'] in requested_ids:
                    reasons.append("lo solicitaste explícitamente")
                sim_det = p.get('similitud_bert_detalles', 0)
                if sim_det > 0.6:
                    reasons.append(f"encaja muy bien con tu búsqueda (similitud {sim_det:.2f})")
                if p.get('max_sim_tipo', 0) > 0.7:
                    reasons.append(f"ideal para un viaje de '{req.tipo_viaje}'")
                if p.get('max_sim_actividad', 0) > 0.7:
                    reasons.append("ofrece actividades de tu interés")
                budget_num = costo_mapping.get(req.presupuesto, 0)
                if p['rango_costo_numerico'] <= budget_num:
                    reasons.append(f"se ajusta a tu presupuesto '{req.presupuesto}'")
                elif p['rango_costo_numerico'] == budget_num + 1:
                    reasons.append("ligeramente superior a tu presupuesto, pero viable")
                if not reasons:
                    reasons.append("buena combinación de atributos y cercanía a otros lugares")
                desglose.append(
                    DesgloseItem(
                        orden=i,
                        nombre_lugar=p['nombre_lugar'],
                        explicacion="; ".join(reasons)
                    )
                )

            items.append(
                ItineraryItem(
                    opcion=idx+1,
                    puntuacion_relevancia=rel_score,
                    distancia_total_km=dist,
                    ruta=ruta,
                    desglose_por_lugar=desglose,
                    justificacion=f"Este itinerario tiene la máxima puntuación de relevancia ({rel_score:.2f}) según tus preferencias."
                )
            )

        return ItineraryResponse(itineraries=items)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
