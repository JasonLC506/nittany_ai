import numpy as np
import cPickle
from tqdm import tqdm
from scipy.sparse import csr_matrix

MAX_ITER = 50

class data_generator(object):
    def __init__(self, data):
        self.data = data
        self.C = len(self.data)
        self.data_process()
        self.N = len(self.data)

    def data_process(self):
        data_new = []
        for i in xrange(self.C):
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
        for i in xrange(self.N):
            sample = self.data[i]
            data_batched.append(sample)
            batch_cnt += 1
            if batch_cnt == batch_size:
                data_batched = self.batch_process(data_batched)
                yield data_batched
                batch_cnt = 0
                data_batched = []
        if batch_cnt > 0:
            yield self.batch_process(data_batched)

    def batch_process(self, data_batched):
        batch_size = len(data_batched)
        batch_input_a = np.zeros([batch_size, self.C + 1])
        batch_input_b = np.zeros([batch_size, self.C + 1], dtype=np.bool_)
        batch_input_a[:,-1] = 1.0
        for i in range(batch_size):
            # print data_batched[i]
            batch_input_a[i, data_batched[i][0]] = data_batched[i][1]
            batch_input_b[i, data_batched[i][-2]] = True
        batch_output = np.array(map(lambda x: x[-1], data_batched))
        return [batch_input_a, batch_input_b], batch_output


class PriorCourseGrade(object):
    def __init__(self, C, K=20,
                 alpha=0.01, lr=0.1, kappa=0.9, step=0,
                 seed=2018
                 ):
        """
        :param C: Number of courses
        :param K: latent vector dimension
        """
        self.C = C
        self.K = K

        self.embds = None                              # [self.C + 1, self.K]
        #self.bias = None                               # [self.K] bias embds measuring students initial level

        # random seed #
        np.random.seed(seed)

        # derivatives #
        self.embds_der = None
        self.bias_der = None

        # learning params #
        self.lr_step = step
        self.lr_kappa = kappa
        self.lr = lr
        self.alpha = alpha                              # L2 normalization scale
        self.batch_size = 100

    def fit(self,
            data,
            data_valid=None,
            alpha=0.01,
            lr=0.1,
            kappa=0.9,
            step=0,
            batch_size=100,
            max_iter=MAX_ITER,
            out="result/course_level"):
        self.lr_step = step
        self.lr_kappa = kappa
        self.lr = lr
        self.alpha = alpha
        self.batch_size = batch_size

        # initialize #
        self.initialize()

        # training #
        for epoch in xrange(max_iter):
            loss_train = self._fit_epoch(data)
            print("epoch: %d, loss_train: %f" % (epoch, loss_train))
            if data_valid is not None:
                loss_valid = self.valid_loss(data_valid)
                print("loss_valid: %f" % loss_valid)
            with open(out + "%03d" % epoch, "w") as of:
                cPickle.dump(self.embds, of)


    def initialize(self):
        self.embds = np.random.random(size=[self.C + 1, self.K])
        self.embds_der = np.zeros(shape=self.embds.shape)
        #self.bias = np.random.random(size=[self.K]) * 0.1
        #self.bias_der = np.zeros(shape=self.bias.shape)

    def _fit_epoch(self, data):
        loss_epoch = 0.0
        N_samp = 0
        # for sample_batched in data.batch_generate(self.batch_size):
        pbar=tqdm(data.batch_generate(self.batch_size))
        for sample_batched in pbar:
            input, output = sample_batched
            pred = self.forward(input)
            loss = self.loss_fn(pred, output, size_average=False)
            self.backward(pred, output)
            self.optimize_step()
            loss_epoch += loss
            N_samp += output.shape[0]
            pbar.set_postfix(loss=loss_epoch/N_samp)
        return loss_epoch / N_samp

    def forward(self, input, test=False):
        """
        :param input: batch of courses grades [batch_size, self.C + 1], dtype=np.float32,
                      where [:, self.C] = 1 as bias
                      and batch of target courses ids [batch_size, self.C], dtype=np.bool_
        :param test: test_mode flag
        """
        # forward pass #
        embds_batch = np.tile(self.embds, [self.batch_size, 1, 1])
        input_embds = [np.tile(np.expand_dims(input[i], -1), [1, 1, self.K]) for i in range(2)]
        v_i_c = np.multiply(input_embds[0], embds_batch)
        v_i = np.max(v_i_c, axis=1)                   # [batch_size, self.K] grade direct scaling
        v_t = np.dot(input[1], self.embds)            # [batch_size, self.K]extract target course embds
        # use v_i and v_t for pred #
        v_diff = (v_t - v_i).clip(min=0)
        diff = np.sum(v_diff, axis=1)
        pred = 1.0 / (1.0 + diff)                     # can be seen as sigmoid(-ln(diff))

        if test:
            return pred

        # calculate derivative for backpropagation #
        ### test ###
        # print("calculate shapes")
        v_diff_der = - np.tile(np.expand_dims(np.power(pred, 2.0), -1), [1, self.K])
        # print(v_diff_der.shape)
        mask_a = v_t > v_i
        # print(mask_a.shape)
        v_t_der = np.where(mask_a, v_diff_der, np.zeros(v_diff_der.shape))
        # print(v_t_der.shape)
        v_i_der = - v_t_der
        mask_b = (v_i_c == np.tile(np.expand_dims(v_i, axis=1), [1, self.C+1, 1]))
        # print(v_i_der.shape)
        v_i_der_expand = np.tile(np.expand_dims(v_i_der, axis=1), [1, mask_b.shape[1], 1])
        embds_batch_der = np.multiply(np.where(mask_b, v_i_der_expand, np.zeros(v_i_der_expand.shape)), input_embds[0])\
                          + np.where(input_embds[1], - v_i_der_expand, np.zeros(v_i_der_expand.shape))
        self.embds_der = embds_batch_der             # [batch_size, self.C, self.K]
        # print(self.embds_der.shape)

        return pred

    def loss_fn(self, pred, output, size_average=False):
        loss = np.power((output - pred), 2.0)
        if size_average:
            return np.average(loss)
        else:
            return np.sum(loss)

    def backward(self, pred, output):
        pred_der = pred - output                    # depend on loss_fn
        embds_der = np.tensordot(pred_der, self.embds_der, axes=(0,0))
        self.embds_der = embds_der

    def optimize_step(self):
        self.lr_step += 1
        lr = self.lr * np.power(self.lr_step, - self.lr_kappa)
        embds_delta = - lr * (self.embds_der + self.alpha)
        self.embds = self.embds + embds_delta

    def valid_loss(self, data):
        loss_valid = 0.0
        N_samp = 0.0
        for sample_batched in data.batch_generate(self.batch_size):
            input, output = sample_batched
            pred = self.forward(input, test=True)
            loss = self.loss_fn(pred, output, size_average=False)
            loss_valid += loss
            N_samp += output.shape[0]
        return loss_valid / N_samp


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


if __name__ == "__main__":
    with open("data/cou_pre", "r") as df:
        data_cou_pre, cou_dict_inv, Ord2Grade = cPickle.load(df)
    # with open("data/cou_pre_test", "r") as df:
    #     data_cou_pre, cou_dict_inv, Ord2Grade = cPickle.load(df)
    C = len(cou_dict_inv)
    data = data_generator(data_cou_pre)
    # data.C = 8900
    print data.C
    print data.N
    pcg = PriorCourseGrade(C=data.C)
    pcg.fit(data, batch_size=256)
