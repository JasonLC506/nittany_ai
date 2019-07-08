import tensorflow as tf
import numpy as np
import _pickle as cPickle
import time
from tqdm import tqdm

from priorCourseGradeTF_Fossil import (
    data_loader,
    PriorCourseGrade as PCG,
)


MAX_ITER = 10

class PriorCourseGrade(PCG):
    def _aggregation(self, grade_weighted_embds):
        rnn_cell = tf.nn.rnn_cell.GRUCell(
            num_units=self.K,   # same size with embd
            name="rnn_cell"
        )
        inner_states, final_state = tf.nn.dynamic_rnn(
            cell=rnn_cell,
            inputs=grade_weighted_embds,
            dtype=tf.float32,
        )
        return tf.reduce_sum(inner_states, axis=1)


if __name__ == "__main__":
    # with open("data/cou_pre", "r") as df:
    #     data_cou_pre, cou_dict_inv, Ord2Grade = cPickle.load(df)
    # # with open("data/cou_pre_test", "r") as df:
    # #     data_cou_pre, cou_dict_inv, Ord2Grade = cPickle.load(df)
    # C = len(cou_dict_inv)
    # data = data_generator(data_cou_pre)
    # # data.C = 8900

    # print(data.C)
    # print(data.N)
    # pcg = PriorCourseGrade(C=data.C)
    # pcg.initialize()
    # pcg.train(data, batch_size=256)

    # with train valid test #
    data_train = data_loader("data/cou_pre_train")
    data_valid = data_loader("data/cou_pre_valid")
    data_test = data_loader("data/cou_pre_test")

    pcg = PriorCourseGrade(C=data_train.C)
    pcg.initialize()
    pcg.train(data_train, data_valid=data_valid, batch_size=256, save_emb=False)
    pcg.restore()
    pcg.evaluate(data_test)
