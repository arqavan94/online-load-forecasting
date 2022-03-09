from numpy import float64
import tensorflow as tf
from data import scaler

from sklearn.metrics import r2_score


def inverse_transform(series,scaler):
    '''
    Inverse transform of scales series
    Args:
        series: scaled series
        scaler: scaler object
    Returns:
        unscaled series
    '''
    return scaler.inverse_transform(series.reshape(-1,1))

def evaluate(y_test, y_pred_noTransfer, y_pred_withTransfer):
    

    # y_test_inverse = inverse_transform(y_test, scaler)
    # y_pred_inverse = inverse_transform(y_pred, scaler)

    # predictions_noTransfer = inverse_transform(predictions_noTransfer, scaler)
    # predictions_withTransfer = inverse_transform(predictions_withTransfer, scaler)
            

    RMSE_object= tf.keras.metrics.RootMeanSquaredError(name="RMSE", dtype=float64)
    RMSE_noTransfer= RMSE_object(y_test, y_pred_noTransfer)
    RMSE_withTransfer= RMSE_object(y_test, y_pred_withTransfer)

    MAE_object= tf.keras.metrics.MeanAbsoluteError(name="MAE", dtype=float64)
    MAE_noTransfer= MAE_object(y_test, y_pred_noTransfer)
    MAE_withTransfer= MAE_object(y_test, y_pred_withTransfer)

    MAPE_object= tf.keras.metrics.MeanAbsolutePercentageError(name="MAPE", dtype=float64)
    MAPE_noTransfer= MAPE_object(y_test, y_pred_noTransfer)
    MAPE_withTransfer= MAPE_object(y_test, y_pred_withTransfer)

   

    return RMSE_noTransfer, RMSE_withTransfer , MAE_noTransfer, MAE_withTransfer,MAPE_noTransfer,MAPE_withTransfer