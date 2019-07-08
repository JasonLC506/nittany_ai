import csv
import xlrd
import cPickle

workbook = xlrd.open_workbook('data/Grading_sourse.xlsm')
print workbook
#
# print len(workbook.sheets())
for sheet in workbook.sheets():
    print sheet.nrows
    print sheet.name
    print sheet.ncols

    data = {}
    # students #
    data["student"] = sheet.col_values(1)[1:]
    data["subject"] = sheet.col_values(3)[1:]
    data["cour_num"] = sheet.col_values(4)[1:]
    data["grade"] = sheet.col_values(16)[1:]

    data["course"] = []
    for i in range(len(data["subject"])):
        data["course"].append([data["subject"][i], data["cour_num"][i]])

    for key in data:
        print key, data[key][0]

    data["student_set"] = set(data["student"])
    data["subject_set"] = set(data["subject"])
    data["course_set"] = set(map(lambda x: x[0] + "_" + x[1], data["course"]))

    print "N_stu, N_sub, N_course", len(data["student_set"]), len(data["subject_set"]), len(data["course_set"])

    with open("data/source", "w") as df:
        cPickle.dump(data, df)
