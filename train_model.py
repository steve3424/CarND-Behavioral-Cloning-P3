import csv
import cv2
import numpy as np

lines = []
with open('first_track_data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:
	source_path = line[0]
	filename = source_path.split('\\')[-1]
	current_path = 'first_track_data/IMG/' + filename
	image = cv2.imread(current_path)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)



from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D

model = Sequential()#model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=[160, 320, 3]))
model.add(Conv2D(6, 5, 5, activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(16, 5, 5, activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(400))
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
