import pytest
import tensorflow as tf
import numpy as np
from keras.models import load_model

# import sys
# sys.path.append(r"C:\Users\dhyan\Downloads\harshil\\classification_dhyanvi\\classification_project\\classification_project")


import logging
logging.basicConfig(filename='test.log', filemode='w', level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')


from src.data_preparation import load_cifar10_data
from src.model import create_cnn_model
from src.train import train_model
from src.predict import predict_image

@pytest.fixture
def cifar10_data():
    logging.info(f'loadng data')
    x_train, y_train, x_test, y_test = load_cifar10_data()
    return x_train, y_train, x_test, y_test

@pytest.fixture
def cifar10_model(cifar10_data):
    logging.info(f'training model')
    x_train, y_train, _, _ = cifar10_data
    # model = create_cnn_model(x_train[0].shape)
    # train_model(model, x_train, y_train, 5, 64)
    model = load_model('model/cifar10_model.h5')
    return model

def test_data_preparation(cifar10_data):
    logging.info(f'testing data preparation')
    x_train, y_train, x_test, y_test = cifar10_data
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert x_train.shape[1:] == (32, 32, 3)
    assert x_test.shape[1:] == (32, 32, 3)
    assert y_train.shape[1] == 10
    assert y_test.shape[1] == 10

def test_model_creation(cifar10_model):
    logging.info(f'testing instance of the model')
    assert isinstance(cifar10_model, tf.keras.models.Sequential)

def test_model_training(cifar10_data, cifar10_model):
    logging.info(f'evaluation of the model')
    x_train, y_train, x_test, y_test = cifar10_data
    _, accuracy = cifar10_model.evaluate(x_test, y_test)
    assert accuracy > 0.7  # Adjust as needed

def test_model_prediction(cifar10_model):
    # Generate a random test image for prediction
    logging.info(f'testing model on the input user image')
    test_image = r"cat1.jpg"
    # test_image = np.random.rand(32, 32, 3)
    # predicted_class = predict_image(cifar10_model, test_image)
    predicted_class = predict_image(test_image)
    # assert 0 <= predicted_class < 10
    assert predicted_class


if __name__ == '__main__':
    pytest.main()
