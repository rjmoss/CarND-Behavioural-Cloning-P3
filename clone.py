import csv
import cv2
import numpy as np
from keras.layers import Flatten, Dense, Lambda
from keras.models import Sequential

DATA_PATH = './SimulationData/'

lines = []
with open('./SimulationData/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
steering_angles  = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = DATA_PATH + 'IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)

    angle = float(line[3])
    steering_angles.append(angle)

X_train = np.array(images)
y_train = np.array(steering_angles)

# Network
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Flatten())
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model.h5')
