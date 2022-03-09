
from ast import Not
import logging
from numpy import float64
from tensorflow import keras
import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import Dropout
from keras.layers import Bidirectional
from sklearn.model_selection import train_test_split
import nni


LOG = logging.getLogger('model')
LOG.setLevel(logging.DEBUG)
LOG.handlers=[logging.FileHandler('log')]
LOG.debug("start")
K.set_image_data_format('channels_last')

TENSORBOARD_DIR = "out"
STANDALONE_LOG_TAG="standlone"

def create_model(hyper_params,X_train):

    input_shape=(X_train.shape[1],X_train.shape[2])
    
    if hyper_params['optimizer'] == 'Adam':
        optimizer = keras.optimizers.Adam(lr=hyper_params['learning_rate'])

    elif hyper_params['optimizer']== 'RMSprop':
        optimizer= keras.optimizers.RMSprop(lr=hyper_params['learning_rate'])

    elif hyper_params['optimizer']== 'Adamax':
        optimizer= keras.optimizers.Adamax(lr=hyper_params['learning_rate'])

    if hyper_params['activation']=='relu':
        activation= 'relu'
    elif hyper_params['activation']=='tanh':
        activation= 'tanh'
    #lstm_node1
    lstm1_nodes= hyper_params['lstm1_nodes']
    lstm2_nodes= hyper_params['lstm2_nodes']
    drop_out= hyper_params['drop_out']

    model = Sequential()

    #auto_encoder
    model.add(LSTM(lstm1_nodes,activation= activation, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(lstm2_nodes,activation= activation, return_sequences=False))
    model.add(RepeatVector(7))
    model.add(LSTM(lstm2_nodes,activation=activation, return_sequences=True))
    model.add(LSTM(lstm1_nodes,activation=activation, return_sequences=True))

    model.add(Bidirectional(LSTM(128,activation=activation, return_sequences=True)))

    #stack Of lstms
    model.add(LSTM(lstm2_nodes,activation=activation, return_sequences=True))
    model.add(Dropout(drop_out))
    model.add(LSTM(lstm2_nodes,activation=activation, return_sequences=False))
    model.add(Dropout(drop_out))

    model.add(Dense(10,activation=activation))
    model.add(Dense(1,activation= activation))
    model.compile(loss='mse', optimizer=optimizer, metrics=[tf.keras.metrics.RootMeanSquaredError(name="RMSE", dtype= float64),tf.keras.metrics.MeanAbsoluteError(name="MAE", dtype=float64),tf.keras.metrics.MeanAbsolutePercentageError(name="MAPE", dtype=float64)])

    return model

class SendMetrics(keras.callbacks.Callback):
    '''
    Keras callback to send metrics to NNI framework
    '''
    def on_epoch_end(self, epoch, logs={}):
        '''
        Run on end of each epoch
        '''
        LOG.debug(logs)
        nni.report_intermediate_result(logs['val_loss'])

    
def train_data(params,X_train,y_train,X_test,y_test,trained_model):

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=None)
    model = create_model(params,X_train)
    
    if trained_model is not None:

        model.set_weights(weights=trained_model.get_weights())

    if nni.get_experiment_id() == "STANDALONE":
        log_path = STANDALONE_LOG_TAG
        TENSORBOARD_DIR = f"{log_path}"
        save_path = f"standlone_models/{log_path}.h5"
    else:
        log_path = f"{nni.get_experiment_id()}/{nni.get_trial_id()}"
        TENSORBOARD_DIR = f"out/{log_path}"
        save_path = f"saved_models/{log_path}.h5"

    #ID=w47oC

    history= model.fit(X_train, y_train, batch_size=params['batch_size'], epochs=params['epochs'], verbose=1,
        validation_data=(X_val,y_val), callbacks=[SendMetrics(), TensorBoard(log_dir=TENSORBOARD_DIR), keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
    
    y_pred= model.predict(X_test) 
    model.save(save_path)

    return y_test,y_pred,history,model,save_path

def transfer_learning(X_train, y_train, X_test, y_test):

    # print('*** Fitting a model without knowledge transfer ***')
    y_test, y_pred_noTransfer, history_noTransfer, model_noTransfer,model= train_data(X_train,y_train,X_test,y_test,trained_model=None)
    # print('\n')
    # print('*** Fitting a model with knowledge transfer ***')
    y_test, y_pred_withTransfer, history_withTransfer, model_withTransfer= train_data(X_train,y_train,X_test,y_test,trained_model=model)
    return model_noTransfer, y_pred_noTransfer, model_withTransfer , y_pred_withTransfer, history_noTransfer, history_withTransfer