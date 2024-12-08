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

seed = ""

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

seed = input("ingrese una entrada de la cual se va a generar el texto de manera predictiva (minusculas solamente, espacios permitidos y numeros tambien)")
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