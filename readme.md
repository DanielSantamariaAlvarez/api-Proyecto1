# Instrucciones de Instalación

1. Clonar el proyecto o descargar el archivo .zip
2. en caso de haber descargado el archivo zip se debe descomprimir.
3. Abrir la carpeta del proyecto en la terminal e ingresar el comando 'pip install -r requirements.txt' para garatizar que se instalen todas las dependencias necesarias.

## Despliegue del proyecto:
Abrir la carpeta del proyecto en la terminal e ingresar el comando 'uvicorn main:app --reload' y correrá el servidor de forma local en el puerto http://127.0.0.1:8000

## Endpoints 

1. El primer endpoint es http://127.0.0.1:8000/predict
2. El segundo endpoint es http://127.0.0.1:8000/predict-r2