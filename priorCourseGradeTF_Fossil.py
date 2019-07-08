import tensorflow as tf
import numpy as np
import _pickle as cPickle
import time
from tqdm import tqdm

from priorCourseGradeTF import (
    DataGenerator as DG,
    PriorCourseGrade as PCG,
)


MAX_ITER = 10

class DataGenerator(DG):
    def __init__(self, data, crop_window=40):
        self.crop_window = crop_window
        super(DataGenerator, self).__init__(
            data=data
        )

    def batch_process(self, data_batched):
        batch_input_courses = np.array(
            list(map(
                lambda x: self._crop_padding(x[0]),
                data_batched
            )),
            dtype=np.int32
        )
        batch_input_grades = np.array(
            list(map(
                lambda x: self._crop_padding(x[1], padding=0.0),
                data_batched
            )),
            dtype=np.float32
        )
        batch_input_target_course = np.array(
            list(map(
                lambda x: x[-2],
                data_batched
            ))
        )
        batch_output = np.array(
            list(map(
                lambda x: x[-1],
                data_batched
            ))
        )
        return {
            "batch_input_courses": batch_input_courses,
            "batch_input_grades": batch_input_grades,
            "batch_input_target_course": batch_input_target_course,
            "batch_output": batch_output
        }

    def _crop_padding(self, seq, padding=None):
        length = len(seq)
        if not padding:
            padding = self.C
        if length < self.crop_window:
            padding_seq = [padding for _ in range(self.crop_window - length)]
        else:
            padding_seq = []
        return (padding_seq + seq)[:self.crop_window]


class PriorCourseGrade(PCG):
    def __init__(self, C, K=20, out="exp_result", crop_window=40):
        self.crop_window = crop_window
        super(PriorCourseGrade, self).__init__(
            C=C,
            K=K,
            out=out
        )

    def setup_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope("input"):
                self.input_courses = tf.placeholder(
                    tf.int32,
                    shape=[None, self.crop_window],
                    name="input_courses"
                )
                self.input_grades = tf.placeholder(
                    tf.float32,
                    shape=[None, self.crop_window],
                    name="input_grades"
                )
                self.output_course = tf.placeholder(
                    tf.int32,
                    shape=[None],
                    name="output_course"
                )
                self.target_grade = tf.placeholder(
                    tf.float32,
                    shape=[None],
                    name="target_grade"
                )
            self.setup_network()
            self.setup_loss()
            self.setup_optimizer()
            self.saver = tf.train.Saver(max_to_keep=1000)
            self.init = tf.global_variables_initializer()
        self.graph.finalize()

    def setup_network(self):
        with tf.name_scope("embedding"):
            self.embds = self.source_course_embds = tf.Variable(
                tf.random_uniform([self.C + 1, self.K], minval=0.0, maxval=1.0),
                name="source_course_embds"
            )
            self.target_course_embds = tf.Variable(
                tf.random_uniform([self.C + 1, self.K], minval=0.0, maxval=1.0),
                name="target_course_embds"
            )
            self.target_course_prior = tf.Variable(
                tf.random_uniform([self.C + 1], minval=0.0, maxval=1.0),
                name="target_course_prior"
            )
            self.aggregation_weights = tf.Variable(
                tf.random_uniform([self.crop_window], minval=0.0, maxval=1.0),
                name="aggregation_weights"
            )
            # [None, crop_window, K]
            self.input_courses_embds = tf.nn.embedding_lookup(
                params=self.source_course_embds,
                ids=self.input_courses
            )
            grade_weighted_embds = (
               0.5 - tf.expand_dims(self.input_grades, axis=-1)
            ) * self.input_courses_embds
            # aggregation #
            aggregated_embds = self._aggregation(
                grade_weighted_embds=grade_weighted_embds
            )
            # [None, K]
            self.output_course_embd = tf.nn.embedding_lookup(
                params=self.target_course_embds,
                ids=self.output_course
            )
            self.output_grade_prior = tf.nn.embedding_lookup(
                params=self.target_course_prior,
                ids=self.output_course
            )
            logit = tf.einsum(
                "ij,ij->i",
                self.output_course_embd,
                aggregated_embds
            )
            self.predict_grade = tf.nn.sigmoid(logit + self.output_grade_prior)

    def _aggregation(self, grade_weighted_embds):
        uniform_sum = tf.reduce_sum(grade_weighted_embds, axis=1)
        weighted_sum = tf.einsum(
            "ijk,j->ik",
            grade_weighted_embds,
            self.aggregation_weights
        )
        aggregated_embds = uniform_sum + weighted_sum
        return aggregated_embds

    def _feed_dict_fn(self, data_batched):
        feed_dict = {
            self.input_courses: data_batched["batch_input_courses"],
            self.input_grades: data_batched["batch_input_grades"],
            self.output_course: data_batched["batch_input_target_course"],
            self.target_grade: data_batched["batch_output"]
        }
        return feed_dict


def data_loader(filename):
    with open(filename, "rb") as df:
        data_cou_pre, cou_dict_inv, Ord2Grade = cPickle.load(df)
    data_generator = DataGenerator(data_cou_pre)
    return data_generator


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
    pcg.restore()
    pcg.train(data_train, data_valid=data_valid, batch_size=256, save_emb=False)
    pcg.restore()
    pcg.evaluate(data_test)
