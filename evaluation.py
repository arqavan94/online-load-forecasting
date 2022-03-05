from numpy import float64
from sklearn.metrics import r2_score
import tensorflow as tf
from data import cnt_transformer
from keras import backend as K
from sklearn.metrics import r2_score

# def coeff_determination(y_true, y_pred):
#     SS_res =  K.sum(K.square( y_true-y_pred )) 
#     SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
#     return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def evaluate(y_test,y_pred):
    
    # y_test_inv = cnt_transformer.inverse_transform(y_test.reshape(1,-1))
    # y_pred_inv = cnt_transformer.inverse_transform(y_pred.reshape(1,-1))

    RMSE_object= tf.keras.metrics.RootMeanSquaredError(name="RMSE", dtype=float64)
    RMSE= RMSE_object(y_test, y_pred)

    MAE_object= tf.keras.metrics.MeanAbsoluteError(name="MAE", dtype=float64)
    MAE= MAE_object(y_test, y_pred)

    MAPE_object= tf.keras.metrics.MeanAbsolutePercentageError(name="MAPE", dtype=float64)
    MAPE= MAPE_object(y_test, y_pred)

    # R2=coeff_determination(y_test,y_pred)

    return RMSE, MAE, MAPE