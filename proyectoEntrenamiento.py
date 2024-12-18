# -*- coding: utf-8 -*-
"""ProyectoProgramacion3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PA7UHQmcDndZUsTiXi-7kKts6_wExa2k
"""

import tensorflow as tf
import numpy as np
import os
import pickle
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from string import punctuation

contenido = requests.get('https://www.gutenberg.org/cache/epub/68566/pg68566.txt').text
open('data/fausto.txt', 'w', encoding="utf-8").write(contenido)

sequence_length = 100
BATCH_SIZE = 128
EPOCHS = 30

# Tenemos el path del archivo de datos

FILE_PATH = "data/fausto.txt"
BASENAME = os.path.basename(FILE_PATH)

# Leemos la data
text = open(FILE_PATH, encoding='utf-8').read()
# removemos las letras mayusculas para un estilo mas uniforme
text = text.lower()
# removemos puntuacion
text = text.translate(str.maketrans('', '', punctuation))
text = text.translate(str.maketrans('', '', '¡ª«·»¿ßáäæèéëíñóöúü—‘’“”•™﻿'))


# Esto nos ayuda basicamente a hacer que el entrenamiento sea mas rapido al reducir el vocabulario y hacer mas facil de digerir el texto

# Veamos algunos stats del dataset

n_chars = len(text)
vocab = ''.join(sorted(set(text)))
print('unique_chars:', vocab)
n_unique_chars = len(vocab)
print('Numero de caracteres:', n_chars)
print('Numero de caracteres unicos:', n_unique_chars)

# Hacemos dos diccionarios, dado que tenemos un string con todos los caracteres unicos de nuestro dataset, podemos hacer un diccionario que mapee a cada caracter un numero y viceversa

char2int = {c: i for i, c in enumerate(vocab)}
int2char = {i: c for i, c in enumerate(vocab)}

# Los guardamos en un archivo
pickle.dump(char2int, open(f'{BASENAME}-char2int.pickle', 'wb'))
pickle.dump(int2char, open(f'{BASENAME}-int2char.pickle', 'wb'))

# Ahora vamos a codificar nuestro dataset, o sea, convertir cada caracter en su entero correspondiente

encoded_text = np.array([char2int[c] for c in text])

# Construimos un tf.data.Dataset para nuestro encoded_text, si necesitamos escalar nuestro codigo a datasets mas grandes

char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)

for char in char_dataset.take(8):
  print(char.numpy(), int2char[char.numpy()])

# Ahora construimos nuestras oraciones, queremos que cada muestra de entrada sea la secuencia de caracteres de longitud sequence_length, y para eso usamos el metodo batch de tf.data.Dataset

sequences = char_dataset.batch(2*sequence_length + 1, drop_remainder=True)

# Ahora lo mostramos

for sequence in sequences.take(2):
  print(''.join([int2char[i] for i in sequence.numpy()]))

# Preparamos nuestros inputs y targets, necesitamos un modo de convertir una muestra (secuencia de caracteres) en multiples muestras. Y para eso podemos utilizar el metodo flat_map()

def split_sample(sample):
  ds = tf.data.Dataset.from_tensors((sample[:sequence_length], sample[sequence_length]))
  for i in range(1, (len(sample)-1) // 2):
    input_ = sample[i: i+sequence_length]
    target = sample[i+sequence_length]
    #Extendemos el datasete con concatenación
    other_ds = tf.data.Dataset.from_tensors((input_, target))
    ds = ds.concatenate(other_ds)
  return ds

#Y ahora preparamos inputs y targets

dataset = sequences.flat_map(split_sample)

#Esto basicamente nos entrega una tupla de inputs y targets, donde conseguimos una gran cantidad de muestras de entrenamiento, y concatenamos para añadirlas juntas

# Hagamos entonces one-hot code de los inputs y las labels (targets)

def one_hot_samples(input_, target):
  # Por ejemplo, de tener el caracter d (que se encuentra codificado como 3, con 5 caracteres unicos)
  # Eso nos retorna el vector: [0, 0, 0, 1, 0], dado que 'd' es el 4to caracter
  return tf.one_hot(input_, n_unique_chars), tf.one_hot(target, n_unique_chars)

dataset = dataset.map(one_hot_samples)

# ahora hemos usado el conveniente metodo "map()", para hacer one-hot encode en cada muestra de nuestro dataset.

# print las primeras 2 muestras
for element in dataset.take(2):
    print("Input:", ''.join([int2char[np.argmax(char_vector)] for char_vector in element[0].numpy()]))
    print("Target:", int2char[np.argmax(element[1].numpy())])
    print("Input shape:", element[0].shape)
    print("Target shape:", element[1].shape)
    print("="*50, "\n")

# repetimos, cambiamos y juntamos el dataset
ds = dataset.repeat().shuffle(1024).batch(BATCH_SIZE, drop_remainder=True) # Con drop_remainder = True para eliminar las muestras con menor tamaño que el batch size

# Armamos el modelo, el cual basicamente tiene dos capas LSTM con un numero de 128 unidades de LSTM arbitrario.

model = Sequential([
    LSTM(256, input_shape=(sequence_length, n_unique_chars), return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dense(n_unique_chars, activation="softmax"),
])

# Definimos el path del modelo

model_weights_path = f'results/{BASENAME}-{sequence_length}.h5'
model.summary()
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])

# Entrenamos al modelo

#Hacemos la carpeta results si todavia no existe

if not os.path.isdir("results"):
  os.mkdir("results")

#Entrenamos al modelo

model.fit(ds, steps_per_epoch=(len(encoded_text) - sequence_length) // BATCH_SIZE, epochs=EPOCHS)

# Guardamos el modelo

model.save(model_weights_path)

import numpy as np
import pickle
import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation
import os

sequence_length = 100
# dataset file path
FILE_PATH = "data/fausto.txt"
# FILE_PATH = "data/python_code.py"
BASENAME = os.path.basename(FILE_PATH)
# Ahora probemos a generar nuevo texto

# Como necesitamos una muestra, tomemos una semilla o alguna sentencia de la data de entrenamiento.

seed = "hermosas flores"

char2int = pickle.load(open(f'{BASENAME}-char2int.pickle', 'rb'))
int2char = pickle.load(open(f'{BASENAME}-int2char.pickle', 'rb'))
vocab_size = len(char2int)

# Construimos el modelo nuevamente

model = Sequential([
    LSTM(256, input_shape=(sequence_length, vocab_size), return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dense(vocab_size, activation='softmax')
])

#Y cargamos el set optimo de pesos del modelo.

model.load_weights(f'results/{BASENAME}-{sequence_length}.h5')

#y generamos

s = seed
n_chars = 400

#Generamos 400 caracteres

generated = ''

for i in tqdm.tqdm(range(n_chars), 'Generando texto'):
  #Hagamos la input de entrada
  X = np.zeros((1, sequence_length, vocab_size))
  for t, char in enumerate(seed):
    X[0, (sequence_length - len(seed)) + t, char2int[char]] = 1
#Predecimos el siguiente caracter
  predicted = model.predict(X, verbose=0)[0]
  #Convertimos el vector a un entero
  next_index = np.argmax(predicted)
  #convertimos el entero a un caracter
  next_char = int2char[next_index]
  #añadimos el caracter a los resultados
  generated += next_char
  #Cambiamos la seed y el caracter predicho

  seed = seed[1:] + next_char

print('Seed:', s)
print('Texto generado:')
print(generated)