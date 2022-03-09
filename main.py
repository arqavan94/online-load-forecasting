
import nni
import logging
from model import transfer_learning
from evaluation import evaluate
from data import X_train, y_train, X_test, y_test
import numpy as np

LOG = logging.getLogger('model')
LOG.setLevel(logging.DEBUG)
LOG.handlers=[logging.FileHandler('log')]
LOG.debug("start")
RMSE_noTransfer_list=[]
RMSE_withTransfer_list=[]
MAE_noTrnasfer_list=[]
MAE_withTransfer_list=[]
MAPE_noTransfer_list=[]
MAPE_withTransfer_list=[]


def generate_default_params():
    '''
    Generate default hyper parameters
    '''
    return {
      
    "optimizer": "Adam",
    "learning_rate": 0.002,
    "activation": "tanh",
    "drop_out": 0.25,
    "lstm1_nodes": 256,
    "lstm2_nodes": 64,
    "batch_size": 64,
    "epochs": 100


    }

if __name__ == '__main__':
   
    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = generate_default_params()
        PARAMS.update(RECEIVED_PARAMS)
        # train
        for i in range(10):
            model_noTransfer, y_pred_noTransfer, model_withTransfer , y_pred_withTransfer,history_noTransfer, history_withTransfer = transfer_learning(X_train, y_train, X_test, y_test)
            
            RMSE_noTransfer, RMSE_withTransfer , MAE_noTransfer, MAE_withTransfer,MAPE_noTransfer,MAPE_withTransfer = evaluate(y_test,y_pred_noTransfer, y_pred_withTransfer)

            RMSE_noTransfer_list.append(RMSE_noTransfer)
            RMSE_withTransfer_list.append(RMSE_withTransfer)

            MAE_noTrnasfer_list.append(MAE_noTransfer)
            MAE_withTransfer_list.append(MAE_withTransfer)

            MAPE_noTransfer_list.append(MAPE_noTransfer)
            MAPE_withTransfer_list.append(MAPE_withTransfer)

            
            RMSE_noTransfer, RMSE_withTransfer , MAE_noTransfer, MAE_withTransfer,
            MAPE_noTransfer,MAPE_withTransfer= np.mean(RMSE_noTransfer_list),np.mean(RMSE_withTransfer_list),np.mean(MAE_noTrnasfer_list),
            np.mean(MAE_noTrnasfer_list),np.mean(MAE_withTransfer_list), np.mean(MAPE_noTransfer_list) ,np.mean(MAPE_withTransfer_list)
            # LOG.debug('Final result is: %.3f %.3f %.3f %.3f', RMSE,MAPE,MAE)

          
            LOG.debug('\n')
            LOG.debug('======== Results for no knowledge transfer =========')
            LOG.debug('The RMSE, MAPE, MAE is {}{}{}'.format(round(np.sqrt(RMSE_noTransfer),4),round(np.sqrt(MAPE_noTransfer),4),round(np.sqrt(MAE_noTransfer),4)))
            print('\n')
            print('======== Results for knowledge transfer =========')
            print('The RMSE, MAPE, MAE is {}{}{}'.format(round(np.sqrt(RMSE_withTransfer),4), round(np.sqrt(MAPE_withTransfer),4), round(np.sqrt(MAE_withTransfer),4)))
        
            # view_predictions(series,predictions_noTransfer,y_test,title='Without Transfer')
            # view_predictions(series,predictions_withTransfer,y_test,title='With Transfer')
        # y_test,y_pred=train_data(PARAMS, 7)
        # RM,MA,MAP=evaluate(y_test,y_pred)
        # LOG.debug('Final result is: %.3f %.3f %.3f', RM,MA,MAP)
        # nni.report_final_result(float(RM))

    except Exception as e:
        LOG.exception(e)
        raise