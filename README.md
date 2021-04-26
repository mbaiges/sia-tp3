# Trabajo Práctico Perceptrón Simple y Multicapa para la materia Sistemas de Inteligencia Artificial

## Instalación

Para correr el programa debe ser necesario instalar python 3

[Descargar Python 3](https://www.python.org/downloads/)

Una vez instalado python, se necesitan la librería pyyaml.
Para eso, se debe tener instalado pip para python
La guía de instalación se encuentra en el siguiente link:

[Instalar Pip](https://tecnonucleous.com/2018/01/28/como-instalar-pip-para-python-en-windows-mac-y-linux/)

Una vez instalado pip, se debe correr, dentro de la carpeta del repositorio, el comando:

```python
pip install -r requirements.txt
```

## Guía de uso

### Configuración

Antes de ejectuar el programa, se debe modificar el archivo `config.yaml`.
En este archivo se deben configurar la ubicaciones y los nombres de los sets de datos para los distintos ejercicios. Tambien especificar si se desea graficar o no y si se desea que la red utilice momentum y de ser así, cuanto porcentaje usar.

A continuación, se muestra un ejemplo de la configuración:

```yaml

# data folder

data_folder: data

# data files

# ej2
ej2_training: ej2-Conjuntoentrenamiento.txt
ej2_outputs: ej2-Salida-deseada.txt

# ej3
ej3_pair_training: ej3-mapa-de-pixeles-digitos-decimales.txt

# extra modes
momentum:
  opt: False
  alpha: 0.8

# plotting
plotting: False

```

### Ejecución

Finalmente, para correr los distintos puntos del trabajo se debe ejecutar el comando:

```python
python .\main.py
```

A continuación, se observarán un menú para la seleccion del ejercicio especificado.

En el menú de acciones se observarán dos opciones:

* Train and Test: Se entrena a la neurona o red neuronal para aprenda. Al terminar, muestra los resultados del aprendizaje y guarda el estado de la neurona o red en un archivo almacenado en la carpeta `saves`. 

* Predict: Utiliza los resultados de `saves` para permitir que se utilice la red para predecir resultados de puntos que no hayan sido analizados, ingresados por entrada estandar.

