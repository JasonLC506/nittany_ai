import tensorflow as tf
import _pickle as cPickle
import numpy as np
import time
from tqdm import tqdm

from priorCourseGradeTF import (
    data_loader,
    PriorCourseGrade as PCG,
)


MAX_ITER = 10


class PriorCourseGrade(PCG):
    def setup_network(self):
        with tf.name_scope("embedding"):
            self.embds = tf.Variable(tf.random_uniform([self.C + 1, self.K], minval=0.0, maxval=1.0),
                                     name="embds")  # self.C is bias embedding
            input_grades_enlarge = tf.expand_dims(self.input_grades, axis=-1)
            input_scaled_embds = tf.multiply(input_grades_enlarge, self.embds)
            # maximum pooling #
            self.input_embd = tf.reduce_max(input_scaled_embds, axis=1)
            # output course embd #
            self.output_embd = tf.nn.embedding_lookup(self.embds, self.output_course)

        # with tf.name_scope("compare"):
        #     self.diff = tf.nn.relu(self.output_embd - self.input_embd)
        #
        # with tf.name_scope("predict"):
        #     self.predict_grade = tf.divide(1.0, tf.add(1.0, tf.reduce_sum(self.diff, axis=-1)))
            self.predict_grade = tf.nn.sigmoid(tf.einsum("ij,ij->i", self.input_embd, self.output_embd))


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

    pcg = PriorCourseGrade(C=data_train.C, K=20)
    pcg.initialize()
    pcg.train(data_train, data_valid=data_valid, batch_size=256, save_emb=False)
    pcg.restore()
    pcg.evaluate(data_test)