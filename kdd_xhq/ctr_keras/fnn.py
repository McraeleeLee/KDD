import os
import time
import numpy as np
import pandas as pd
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Reshape, Lambda, concatenate, dot, add
from keras.layers import Dropout, GaussianDropout, multiply, SpatialDropout1D, BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.optimizers import Adam
from keras.initializers import TruncatedNormal
print('fnn')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


file_path = "fnn.h5"
batch_size = 256
epochs = 2


def __load_data(train_filename, test_filename):
    train = pd.read_csv(train_filename, header=None, names=[])
    test = pd.read_csv(test_filename)
    numer_columns = [col for col in train.columns if 'numer' in col]
    cat_columns = [col for col in train.columns if 'cat' in col]
    y_train = train['label'].values
    y_test = test['label'].values
    cols = cat_columns + numer_columns
    return cols, train, test, y_train, y_test


# ----------------------------------model-------------
def __base_model(cols, train, test):
    cat_num = len(cols)
    field_cnt = cat_num
    cat_field_input = []
    field_embedding = []
    lr_embedding = []
    for cat in cols:
        input = Input(shape=(1,))
        cat_field_input.append(input)
        nums = pd.concat((train[cat], test[cat])).max() + 1
        # ffm embeddings
        x_embed = Embedding(nums, 10, input_length=1, trainable=True)(input)
        x_reshape = Reshape((10,))(x_embed)
        field_embedding.append(x_reshape)
        # ffm embeddings
    #######ffm layer##########
    embed_layer = concatenate(field_embedding, axis=-1)
    #######dnn layer##########
    embed_layer = Dense(64)(embed_layer)
    embed_layer = BatchNormalization()(embed_layer)
    embed_layer = Activation('relu')(embed_layer)
    embed_layer = Dense(64)(embed_layer)
    embed_layer = BatchNormalization()(embed_layer)
    embed_layer = Activation('relu')(embed_layer)
    embed_layer = Dense(1)(embed_layer)
    ########linear layer##########
    lr_layer = embed_layer
    preds = Activation('sigmoid')(lr_layer)
    opt = Adam(0.001)
    model = Model(inputs=cat_field_input, outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
    return model


def fnn_training(cols, train, test, y_train, y_test):
    # training########################################3
    x_train = train[cols].values
    x_test = test[cols].values
    x_train = list(x_train.T)
    x_test = list(x_test.T)
    early = EarlyStopping(monitor="val_loss", patience=3)
    checkpoint = ModelCheckpoint(
        file_path, save_weights_only=True, verbose=1, save_best_only=True)
    callbacks_list = [early, checkpoint]  # early
    model = __base_model()
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test), callbacks=callbacks_list, shuffle=False)
