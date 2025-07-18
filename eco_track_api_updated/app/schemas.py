from pydantic import BaseModel
from typing import List, Optional

class Location(BaseModel):
    lat: float
    lon: float

class RecommendationRequest(BaseModel):
    user_location: Location
    tipo_viaje: str
    actividades: List[str]
    presupuesto: str
    compania: Optional[str]
    detalles_adicionales: str

class RoutePlace(BaseModel):
    nombre_lugar: str
    relevancia: float
    url_imagen_principal: str
    url_imagen_secundaria: str
    descripcion_corta: Optional[str]

class DesgloseItem(BaseModel):
    orden: int
    nombre_lugar: str
    explicacion: str

class ItineraryItem(BaseModel):
    opcion: int
    puntuacion_relevancia: float
    distancia_total_km: float
    ruta: List[RoutePlace]
    desglose_por_lugar: List[DesgloseItem]
    justificacion: str

class ItineraryResponse(BaseModel):
    itineraries: List[ItineraryItem]
