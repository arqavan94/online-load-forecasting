
import nni
import logging
from model import train_data
from evaluation import evaluate
import numpy as np

LOG = logging.getLogger('model')
LOG.setLevel(logging.DEBUG)
LOG.handlers=[logging.FileHandler('log')]
LOG.debug("start")
RMSE_list=[]
MAE_list=[]
MAPE_list=[]
R2_list=[]

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
            y_test,y_pred=train_data(PARAMS, 7)
            RM,MA,MAP,R2=evaluate(y_test,y_pred)
            RMSE_list.append(RM)
            MAE_list.append(MA)
            MAPE_list.append(MAP)
            # R2_list.append(R2)
            RMSE,MAPE,MAE= np.mean(RMSE_list),np.mean(MAPE_list),np.mean(MAE_list)
            LOG.debug('Final result is: %.3f %.3f %.3f %.3f', RMSE,MAPE,MAE)
        # y_test,y_pred=train_data(PARAMS, 7)
        # RM,MA,MAP=evaluate(y_test,y_pred)
        # LOG.debug('Final result is: %.3f %.3f %.3f', RM,MA,MAP)
        # nni.report_final_result(float(RM))

    except Exception as e:
        LOG.exception(e)
        raise