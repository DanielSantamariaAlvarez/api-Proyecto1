from pydantic import BaseModel
from typing import List

class DataModel(BaseModel):

# Estas varibles permiten que la librería pydantic haga el parseo entre el Json recibido y el modelo declarado
    study_and_condition: str

    # Esta función retorna los nombres de las columnas correspondientes con el modelo exportado en joblib
    def columns():
        return ["study_and_condition"]


class DListar(BaseModel):
    texto: List[DataModel]
