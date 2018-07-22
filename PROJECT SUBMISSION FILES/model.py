import csv
import cv2
import numpy as np


'''
WATCH SLASHES IN FILE PATHS WHEN MOVING FROM LINUX (/) TO WINDOWS (\\)
'''


data_folders = ['track1_lap1', 'track1_lap2', 'track1_backlap1', 'track1_backlap2']

csv_lines = []
for folder in data_folders:
	with open(folder + '/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		lines = []
		for line in reader:
			lines.append(line)
		csv_lines.append(lines)


correction = 0.2
corrections = [0, correction, -correction]

images = []
measurements = []
for f in range(len(data_folders)):
	folder = data_folders[f]
	for line in csv_lines[f]:
		for i in range(3):
			source_path = line[i]
			filename = source_path.split('\\')[-1]
			current_path = folder + '/IMG/' + filename
			image = cv2.imread(current_path)
			images.append(image)
			measurement = float(line[3]) + corrections[i]
			measurements.append(measurement)

'''
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images,measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image,1))
	augmented_measurements.append(measurement * -1.0)
'''

X_train = np.array(images)
y_train = np.array(measurements)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# preprocessing
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

# 3 convolution layers
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3, 3, activation='relu'))

# 3 fc layers and 1 output
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)


model.save('model.h5')


'''
TECHNIQUES

-Build robust network (Nvidia's model)
-Drive multiple laps
-Train on all 3 camera images with correction

-- USING 2 REGULAR LAPS AND 2 BACKWARD LAP WAS A SUCCESS!!
'''
