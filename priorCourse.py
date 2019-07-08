import numpy as np
import _pickle as cPickle
from tqdm import tqdm
from scipy.sparse import csr_matrix

MAX_ITER = 50

class data_generator(object):
    def __init__(self, data):
        self.data = data
        self.N = len(self.data)

    def batch_generate(self, batch_size=1, shuffle=True):
        if shuffle:
            np.random.shuffle(self.data)
        batch_cnt = 0
        batch_input = []
        batch_output = []
        for i in range(self.N):
            sample = self.data[i]
            batch_input.append(sample[0])
            batch_output.append(sample[-1])
            batch_cnt += 1
            if batch_cnt == batch_size:
                yield np.array(batch_input, dtype=np.bool_), np.array(batch_output)
                batch_cnt = 0
                batch_input = []
                batch_output = []
        if batch_cnt > 0:
            yield np.array(batch_input, dtype=np.bool_), np.array(batch_output)


class NNegLasso(object):
    def __init__(self, C):
        self.C = C                                                          # number of courses
        self.alpha = 0                                                      # L1 scale

        self.v = None                                                       # raw impact parameter for each course

        self.bias = 0                                                       # bias

        self.v_der = None                                                   # derivative over v
        self.bias_der = None

        self.lr_step = 0
        self.lr_kappa = 0.9
        self.lr = 0.1

    def fit(self, data, alpha=0.3, kappa=0.9, lr=0.1, data_valid=None, batch_size=100, max_iter=MAX_ITER):
        self.alpha = alpha
        self.lr_kappa = kappa
        self.lr = lr
        self.batch_size = batch_size

        # initialize #
        self.v = np.zeros(self.C)
        self.v_der = np.zeros(self.C)

        for epoch in range(max_iter):
            loss_train = self._fit_epoch(data)
            # print "epoch: %d, loss_train: %f" % (epoch, loss_train)
            if data_valid is not None:
                loss_valid = self.valid_loss(data_valid)
                print("loss_valid: %f" % loss_valid)

    def _fit_epoch(self, data):
        loss_epoch = 0.0
        N_samp = 0
        for sample_batched in data.batch_generate(self.batch_size):
            input, output = sample_batched
            pred = self.forward(input)
            loss = self.loss_fn(pred, output, size_average=False)
            self.backward(pred, output)
            self.optimize_step()
            loss_epoch += loss
            N_samp += input.shape[0]
        return loss_epoch / N_samp


    def forward(self, input, test=False):
        """
        :param input: batch of courses [batch_size, self.C] dtype=np.bool_
        """
        v_nn = self.v.clip(min=0)                                           # clamp
        pred = np.dot(input, v_nn) + self.bias
        if not test:
            self.v_der = input.astype(np.float64)
            self.bias_der = np.ones(input.shape[0])
        return pred

    def loss_fn(self, pred, output, size_average=True):
        score = sigmoid(pred)
        loss = np.power((score-output), 2.0)
        if size_average:
            return np.average(loss)
        else:
            return np.sum(loss)

    def backward(self, pred, output):
        score = sigmoid(pred)
        batch_der = (score - output) * score * (1 - score)
        self.v_der = np.dot(batch_der, self.v_der)
        self.bias_der = np.dot(batch_der, self.bias_der)

    def optimize_step(self):
        self.lr_step += 1
        lr = self.lr * np.power(self.lr_step, - self.lr_kappa)
        v_delta = - lr * (self.v_der + self.alpha)
        bias_delta = -lr * (self.bias_der)
        self.v = self.v + v_delta
        self.v = self.v.clip(min=0)
        self.bias = self.bias + bias_delta

    def valid_loss(self, data):
        loss_valid = 0.0
        N_samp = 0.0
        for sample_batched in data.batch_generate(self.batch_size):
            input, output = sample_batched
            pred = self.forward(input, test=True)
            loss = self.loss_fn(pred, output, size_average=False)
            loss_valid += loss
            N_samp += input.shape[0]
        return loss_valid / N_samp

    def restore(self, v, bias):
        self.v = v
        self.bias = bias
        self.C = self.v.shape[0]
        return self

    def predict(self, input, restored=True):
        if restored:
            v_nn = self.v                                                       # clamp no need for restored model
        else:
            v_nn = self.v.clip(min=0)                                           # clamp
        if type(v_nn) == csr_matrix:
            v_nn = v_nn.toarray().squeeze()
        pred = np.dot(input, v_nn) + self.bias
        score = sigmoid(pred)
        return score

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def synthetic(N_sparse, N_samp, C=6000):
    data = []
    for i in range(N_samp):
        input_ind = np.random.choice(C, np.random.randint(0,N_sparse), replace=False)
        input = np.zeros(C, dtype=np.bool_)
        input[input_ind] = True
        output = np.random.random()
        data.append([input, output])
    return data

def ind2onehot(ind, N_range):
    result = np.zeros(N_range, dtype=np.bool_)
    result[ind] = True
    return result

if __name__ == "__main__":
    # C = 6000
    # data = synthetic(20, 50000, C=C)
    with open("data/cou_pre", "r") as df:
        data_cou_pre, cou_dict_inv, Ord2Grade = cPickle.load(df)
    C = len(cou_dict_inv)

    prior_cou_graph = np.zeros([C,C], dtype=np.float64)
    cou_bias = np.zeros(C, dtype=np.float64)

    pbar = tqdm(range(C), total=C)
    for c in pbar:
        data_sparse = data_cou_pre[c]
        data = map(lambda x: [ind2onehot(x[0], C), x[1]], data_sparse)
        model = NNegLasso(C=C)
        model.fit(data_generator(data), alpha=2.0, max_iter=20)
        prior_cou_graph[c] = model.v
        cou_bias[c] = model.bias
        pbar.set_postfix(NNZ=np.sum(model.v>0))

    with open("data/graph_prior_cou", "w") as f:
        cPickle.dump([csr_matrix(prior_cou_graph), cou_bias], f)