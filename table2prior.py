import csv
import numpy as np
import _pickle as cPickle

from common import StrToBytes

N_semaster = 12

Ord2Grade = {"A": 0.90, "A-": 0.80, "B+": 0.70, "B": 0.60, "B-": 0.50, "C+": 0.40, "C": 0.30, "D": 0.1, "F": 0.0}

enroll_stu = {}

# cou_dict = {}
# N_cou = 0
with open("data/cou_n2i_i2n", "r") as df:
    cou_dict, cou_dict_inv = cPickle.load(StrToBytes(df))
    N_cou = len(cou_dict)

    print(cou_dict)
    print(cou_dict_inv)

stu_dict = {}
N_stu = 0

# for i_sem in range(N_semaster):
#     with open("data/enroll_table_sem_%02d.csv" % i_sem, 'rb') as df:
#         reader = csv.reader(df)
#         table = list(reader)
#
#         for record in table:
#             stu, cou, gra = record
#             enroll_stu.setdefault(stu,
#                                   {"grade": [[] for _ in range(N_semaster)],
#                                    "cou_taken": [[] for _ in range(N_semaster)]})
#             enroll_stu[stu]["grade"][i_sem].append(gra)
#             enroll_stu[stu]["cou_taken"][i_sem].append(cou)
#             if cou not in cou_dict:
#                 print("unexpected course", cou)
#                 raise ValueError
#                 cou_dict[cou] = N_cou
#                 N_cou += 1
#             if stu not in stu_dict:
#                 stu_dict[stu] = N_stu
#                 N_stu += 1
#
# print(N_stu, N_cou)
# with open("data/enroll_stu_from_table", "w") as f:
#     cPickle.dump([enroll_stu, stu_dict, cou_dict], f)

with open("data/enroll_stu_from_table", "r") as f:
    enroll_stu, stu_dict, cou_dict = cPickle.load(StrToBytes(f))


# data_cou_pre = [[] for _ in range(len(cou_dict))]
# for stu in enroll_stu:
#     cou_taken = enroll_stu[stu]["cou_taken"]
#     grade = enroll_stu[stu]["grade"]
#
#     for i_sem in range(N_semaster):
#         cou_taken_sem = cou_taken[i_sem]
#         grade_sem = grade[i_sem]
#         for i_cou in range(len(cou_taken_sem)):
#             cou_target = cou_taken_sem[i_cou]
#             grade_target = grade_sem[i_cou]
#             cou_pre = sum(cou_taken[:i_sem], [])
#             grade_taken = sum(grade[:i_sem], [])
#
#             cid_target = cou_dict[cou_target]
#             cid_pre = map(lambda x: cou_dict[x], cou_pre)
#             gra_target = Ord2Grade[grade_target]
#             gra_pre = map(lambda x: Ord2Grade[x], grade_taken)
#             data_cou_pre[cid_target].append([cid_pre, gra_pre, gra_target])
#
# # cou_dict_inv = [0 for _ in range(len(cou_dict))]
# # for cou in cou_dict:
# #     cou_dict_inv[cou_dict[cou]] = cou
# with open("data/cou_pre", "w") as f:
#     cPickle.dump([data_cou_pre, cou_dict_inv, Ord2Grade], f)

# train valid test version #

data_cou_pre_train = [[] for _ in range(len(cou_dict))]
data_cou_pre_valid = [[] for _ in range(len(cou_dict))]
data_cou_pre_test = [[] for _ in range(len(cou_dict))]

np.random.seed(2019)

for stu in enroll_stu:
    cou_taken = enroll_stu[stu]["cou_taken"]
    grade = enroll_stu[stu]["grade"]

    N_semaster_train = np.random.randint(0, N_semaster - 1)
    N_semaster_valid = N_semaster_train + 1
    N_semaster_test = N_semaster_valid + 1
    for i_sem in range(N_semaster_test):
        cou_taken_sem = cou_taken[i_sem]
        grade_sem = grade[i_sem]
        for i_cou in range(len(cou_taken_sem)):
            cou_target = cou_taken_sem[i_cou]
            grade_target = grade_sem[i_cou]
            cou_pre = sum(cou_taken[:i_sem], [])
            grade_taken = sum(grade[:i_sem], [])

            cid_target = cou_dict[cou_target]
            cid_pre = list(map(lambda x: cou_dict[x], cou_pre))
            gra_target = Ord2Grade[grade_target]
            gra_pre = list(map(lambda x: Ord2Grade[x], grade_taken))

            if i_sem < N_semaster_train:
                data_cou_pre_train[cid_target].append([cid_pre, gra_pre, gra_target])
            elif i_sem == N_semaster_train:
                data_cou_pre_valid[cid_target].append([cid_pre, gra_pre, gra_target])
            else:
                data_cou_pre_test[cid_target].append([cid_pre, gra_pre, gra_target])

# cou_dict_inv = [0 for _ in range(len(cou_dict))]
# for cou in cou_dict:
#     cou_dict_inv[cou_dict[cou]] = cou
with open("data/cou_pre_train", "wb") as f:
    cPickle.dump([data_cou_pre_train, cou_dict_inv, Ord2Grade], f)
with open("data/cou_pre_valid", "wb") as f:
    cPickle.dump([data_cou_pre_valid, cou_dict_inv, Ord2Grade], f)
with open("data/cou_pre_test", "wb") as f:
    cPickle.dump([data_cou_pre_test, cou_dict_inv, Ord2Grade], f)

