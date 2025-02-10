import numpy as np

from strategy.abstract_strategy import AbstractStrategy
from model.abstract_model import AbstractModel
from dataset.adaptest_dataset import AdapTestDataset
import random
from tqdm import tqdm

class BECATstrategy(AbstractStrategy):
    
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'BECAT'
    
    def adaptest_select(self, model: AbstractModel, adaptest_data: AdapTestDataset,S_set):
        assert hasattr(model, 'delta_q_S_t'), \
            'the models must implement delta_q_S_t method' 
        assert hasattr(model, 'get_pred'), \
            'the models must implement get_pred method for accelerating' 
        pred_all = model.get_pred(adaptest_data) 
        selection = {} 
        for sid in tqdm(adaptest_data.data.keys(), "Selecting: "): 
            tmplen = (len(S_set[sid])) 
            untested_questions = np.array(list(adaptest_data.untested[sid])) 
            sampled_elements = np.random.choice(untested_questions, tmplen + 5) 
            untested_deltaq = [model.delta_q_S_t(qid, pred_all[sid],S_set[sid],sampled_elements) for qid in untested_questions] 
            j = np.argmax(untested_deltaq) 
            selection[sid] = untested_questions[j] 

        return selection
    