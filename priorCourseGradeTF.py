import tensorflow as tf
import _pickle as cPickle
import numpy as np
import time
from tqdm import tqdm


MAX_ITER = 10

class DataGenerator(object):
    def __init__(self, data):
        self.data = data
        self.C = len(self.data)
        self.data_process()
        self.N = len(self.data)

    def data_process(self):
        data_new = []
        for i in range(self.C):
            course_target = i
            for j in range(len(self.data[i])):
                # course_input, grade_input, course_target, grade_target #
                data_new.append([self.data[i][j][0], self.data[i][j][1], course_target, self.data[i][j][2]])
        self.data = data_new

    def batch_generate(self, batch_size=1, shuffle=True):
        if shuffle:
            np.random.shuffle(self.data)
        batch_cnt = 0
        data_batched = []
        for i in range(self.N):
            sample = self.data[i]
            data_batched.append(sample)
            batch_cnt += 1
            if batch_cnt == batch_size:
                data_batched = self.batch_process(data_batched)
                yield data_batched
                batch_cnt = 0
                data_batched = []
        # if batch_cnt > 0:
        #     yield self.batch_process(data_batched)

    def batch_process(self, data_batched):
        batch_size = len(data_batched)
        batch_input_a = np.zeros([batch_size, self.C + 1])
        # batch_input_b = np.zeros([batch_size, self.C + 1], dtype=np.bool_)
        batch_input_a[:,-1] = 1.0
        for i in range(batch_size):
            # print data_batched[i]
            batch_input_a[i, data_batched[i][0]] = data_batched[i][1]
        batch_input_b = np.array(list(map(lambda x: x[-2], data_batched)))
        batch_output = np.array(list(map(lambda x: x[-1], data_batched)))
        return [batch_input_a, batch_input_b], batch_output


class PriorCourseGrade(object):
    def __init__(self, C, K=20, out="exp_result/"):
        self.C = C
        self.K = K
        self.out = out
        #         self.graph = None
        #         self.loss = None

        self.setup_graph()
        self.sess = tf.Session(graph=self.graph)

    def setup_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope("input"):
                self.input_grades = tf.placeholder(tf.float32, shape=[None, self.C + 1], name="input_grades")
                self.output_course = tf.placeholder(tf.int64, shape=[None], name="output_course")
                self.target_grade = tf.placeholder(tf.float32, shape=[None], name="target_grade")

            self.setup_network()
            self.setup_loss()
            self.setup_optimizer()
            self.saver = tf.train.Saver(max_to_keep=1000)

            self.init = tf.global_variables_initializer()
        self.graph.finalize()

    def setup_network(self):
        with tf.name_scope("embedding"):
            self.embds = tf.Variable(tf.random_uniform([self.C + 1, self.K], minval=0.0, maxval=1.0),
                                     name="embds")  # self.C is bias embedding
            input_grades_enlarge = tf.tile(tf.expand_dims(self.input_grades, axis=-1), [1, 1, self.K])
            input_scaled_embds = tf.multiply(input_grades_enlarge, self.embds)
            # maximum pooling #
            self.input_embd = tf.reduce_max(input_scaled_embds, axis=1)
            # output course embd #
            self.output_embd = tf.nn.embedding_lookup(self.embds, self.output_course)

        with tf.name_scope("compare"):
            self.diff = tf.nn.relu(self.output_embd - self.input_embd)

        with tf.name_scope("predict"):
            self.predict_grade = tf.divide(1.0, tf.add(1.0, tf.reduce_sum(self.diff, axis=-1)))

    def setup_loss(self):
        with tf.name_scope("loss"):
            self.loss = tf.losses.mean_squared_error(self.target_grade, self.predict_grade)

    def setup_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)

    def initialize(self):
        self.sess.run(self.init)

    def save(self, save_path="ckpt/best"):
        self.saver.save(
            sess=self.sess,
            save_path=save_path
        )

    def restore(self, save_path="ckpt/best"):
        self.saver.restore(
            sess=self.sess,
            save_path=save_path
        )

    def _feed_dict_fn(self, data_batched):
        [input_grades_batched, output_course_batched], target_grade_batched = data_batched
        feed_dict = {
            self.input_grades: input_grades_batched,
            self.output_course: output_course_batched,
            self.target_grade: target_grade_batched
        }
        return feed_dict

    def train(self, data, data_valid=None, batch_size=100, max_iter=MAX_ITER, save_emb=True):
        print("start training")
        start = time.time()
        sess = self.sess

        loss_best = None
        for epoch in range(max_iter):
            loss_epoch = 0.0
            N_samples = 0
            pbar = tqdm(data.batch_generate(batch_size), total=(data.N / batch_size))
            for data_batched in pbar:
                feed_dict = self._feed_dict_fn(data_batched)
                _, loss, embds = sess.run([self.optimizer, self.loss, self.embds], feed_dict=feed_dict)

                loss_epoch += loss
                N_samples += batch_size
                pbar.set_postfix(epoch=epoch, loss=loss)

            print("epoch %d: training loss %f" % (epoch, loss_epoch / N_samples))
            if save_emb:
                with open(self.out + "embds_epoch%03d" % epoch, "wb") as of:
                    cPickle.dump(embds, of)
            if data_valid is not None:
                loss_valid = self.evaluate(data_valid, batch_size=batch_size)
                print("epoch %d: valid loss %f" % (epoch, loss_valid))

                if loss_best is None or loss_best > loss_valid:
                    loss_best = loss_valid
                    self.save()

            end = time.time()
            print("use %f s" % (end - start))
        print("done training, using in total %f s" % (time.time() - start))

    def evaluate(self, data, batch_size=256):
        print("start training")
        start = time.time()
        sess = self.sess

        loss_epoch = 0.0
        N_samples = 0
        pbar = tqdm(data.batch_generate(batch_size), total=(data.N / batch_size))
        for data_batched in pbar:
            feed_dict = self._feed_dict_fn(data_batched)
            _, loss, embds = sess.run([self.loss, self.loss, self.embds], feed_dict=feed_dict)

            loss_epoch += loss
            N_samples += batch_size
            pbar.set_postfix(loss=loss)

        loss_result = loss_epoch / N_samples
        print("evaluation loss %f" % (loss_epoch / N_samples))
        print("done evaluation, using in total %f s" % (time.time() - start))
        return loss_result


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

    pcg = PriorCourseGrade(C=data_train.C, K=20)
    pcg.initialize()
    pcg.train(data_train, data_valid=data_valid, batch_size=256)
    pcg.restore()
    pcg.evaluate(data_test)