# Configuración e hiperparámetros del sistema
enable_logging = True

config = {
    'pesos_lugar': {
        'semantica_tipo': 0.2,
        'semantica_actividades': 0.05,
        'bert_detalles': 0.7,
        'presupuesto': 0.05,
    },
    'max_candidatos_inicial': 15,
    'num_lugares_min': 3,
    'num_lugares_max': 5,
    'max_itinerarios_crudos': 15,
    'cantidad_final_itinerarios': 25,
}

# Mapeo de rangos de costo a valor numérico
costo_mapping = {
    'Free': 0,
    'Low': 1,
    'Low-Medium': 2,
    'Medium': 3,
    'High': 4,
    'Premium': 5
}
