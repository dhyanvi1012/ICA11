import tensorflow as tf
from src.data_preparation import load_cifar10_data
from src.model import create_cnn_model

# Load CIFAR-10 data
x_train, y_train, x_test, y_test = load_cifar10_data()

# Create and compile the model
model = create_cnn_model(x_train[0].shape)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))


# Save the trained model
model.save('model/cifar10_model.h5')
