import csv, random, os
from tqdm import tqdm
import pandas as pd
from util.util import *


random.seed(120)
NULL_USER_FEATURE = [0, 0, 0, 0]


if not os.path.exists("dataset"):
    os.mkdir("dataset")


def split_train_valid(stage, split=.1):
    filename = "./underexpose_train/clean_underexpose_train_click-{}.csv".format(stage)
    if not os.path.exists(filename):
        raise FileNotFoundError("No file: {}".format(filename))

    with open(filename, newline='') as f:
        reader = csv.reader(f)
        all_list = []
        for r in reader:
            all_list.append(r)
        user_id_list = set([i for i, _, _ in all_list])
        user_id_list = [i for i in user_id_list]
        print("stage {}, total user {}".format(stage, len(user_id_list)))
        random.shuffle(user_id_list)
        with open("dataset/clean_split1_click-{}_valid.csv".format(stage), "w", newline="") as fw_valid:
            writer_valid = csv.writer(fw_valid)
            with open("dataset/clean_split1_click-{}_train.csv".format(stage), "w", newline="") as fw_train:
                writer_train = csv.writer(fw_train)
                len_of_valid = int(len(user_id_list)*split)
                user_in_valid = set(user_id_list[:len_of_valid])
                user_in_train = set(user_id_list[len_of_valid:])
                train_ls = []
                valid_ls = []
                for l in all_list:
                    if l[0] in user_in_train:
                        train_ls.append(l)
                    if l[0] in user_in_valid:
                        valid_ls.append(l)
                print("stage {}, split {}, train {}, valid {}".format(stage, 1, len(train_ls), len(valid_ls)))
                writer_train.writerows(train_ls)
                writer_valid.writerows(valid_ls)
        with open("dataset/clean_split2_click-{}_valid.csv".format(stage), "w", newline="") as fw_valid:
            writer_valid = csv.writer(fw_valid)
            with open("dataset/clean_split2_click-{}_train.csv".format(stage), "w", newline="") as fw_train:
                writer_train = csv.writer(fw_train)
                len_of_valid = int(len(user_id_list)*split)
                user_in_train = set(user_id_list[:-len_of_valid])
                user_in_valid = set(user_id_list[-len_of_valid:])
                train_ls = []
                valid_ls = []
                for l in all_list:
                    if l[0] in user_in_train:
                        train_ls.append(l)
                    if l[0] in user_in_valid:
                        valid_ls.append(l)
                print("stage {}, split {}, train {}, valid {}".format(stage, 2, len(train_ls), len(valid_ls)))
                writer_train.writerows(train_ls)
                writer_valid.writerows(valid_ls)


def construct_dataset(stage):
    all_item_feat = pickle_read("dataset/underexpose_item_feat.pkl")
    all_user_feat = pickle_read("dataset/underexpose_user_feat.pkl")

    for split in range(1, 3):
        item_count = dict()
        with open("dataset/clean_split{0}_click-{1}_valid.csv".format(split, stage), newline="") as fr_valid:
            reader_valid = csv.reader(fr_valid)
            validset = []
            count = 0
            
            for line in reader_valid:
                count += 1
                userid, itemid, qtime = line
                if itemid not in all_item_feat:
                    continue
                if userid not in all_user_feat:
                    user_f = NULL_USER_FEATURE
                else:
                    user_f = all_user_feat[userid]
                if len(user_f) < 4:
                    user_f.insert(2, 0)

                item_count[itemid] = item_count.get(itemid,0)+1
                item_f = all_item_feat[itemid][0] + all_item_feat[itemid][1]
                record_feat = [1., qtime] + user_f + item_f
                assert len(record_feat) == 262
                validset.append(record_feat)
            print("split {}, stage {}, total {}, valid kept {}".format(split, stage, count, len(validset)))
            pickle_write(
                "dataset/construct_split{0}_click-{1}_valid.pkl".format(split, stage), validset)

        with open("dataset/clean_split{0}_click-{1}_train.csv".format(split, stage), newline="") as fr_train:
            reader_train = csv.reader(fr_train)
            trainset = []
            count = 0
            click_set = set()
            qtime_ls = []
            for line in reader_train:
                count += 1
                userid, itemid, qtime = line
                qtime_ls.append(qtime)
                click_set.add("{}-{}".format(userid, itemid))
                if itemid not in all_item_feat:
                    continue
                if userid not in all_user_feat:
                    user_f = NULL_USER_FEATURE
                else:
                    user_f = all_user_feat[userid]
                if len(user_f) < 4:
                    user_f.insert(2, 0)

                item_count[itemid] = item_count.setdefault(itemid,0)+1
                item_f = all_item_feat[itemid][0] + all_item_feat[itemid][1]
                record_feat = [1., qtime] + user_f + item_f
                assert len(record_feat) == 262
                trainset.append(record_feat)
            print("split {}, stage {}, total {}, train kept {}".format(
                split, stage, count, len(trainset)))

            user_set = [i for i in all_user_feat.keys()]
            item_set = [i for i in all_item_feat.keys()]
            #get most popular item for negetive subset  
            item_count = sorted(item_count.items(),key=lambda x:x[1],reverse=True)
            # print(item_count)
            item_most_popular_id = [i[0] for i in item_count[:len(item_count)//3]]
            
            # print(item_most_popular_id)
            # print(item_set)
            # exit(0)
            invalid_data_count = 0
            while invalid_data_count < count//2 :
                userid = random.choice(user_set)

                if invalid_data_count % 2 == 0:
                    itemid = random.choice(item_set)
                else:   
                    itemid = random.choice(item_most_popular_id)
                
                #itemid = random.choice(item_set)
                k = "{}-{}".format(userid, itemid)
                if k not in click_set:
                    click_set.add(k)
                    qtime = random.choice(qtime_ls)
                    if itemid not in all_item_feat:
                        continue
                    if userid not in all_user_feat:
                        user_f = NULL_USER_FEATURE
                    else:
                        user_f = all_user_feat[userid]
                    if len(user_f) < 4:
                        user_f.insert(2, 0)
                    item_f = all_item_feat[itemid][0] + \
                        all_item_feat[itemid][1]
                    record_feat = [.0, qtime] + user_f + item_f
                    assert len(record_feat) == 262
                    trainset.append(record_feat)
                    invalid_data_count += 1

            print("split {}, stage {}, total {}, train + invalid data {}".format(split, stage, count, len(trainset)))
            random.shuffle(trainset)
            pickle_write(
                "dataset/construct_split{0}_click-{1}_train.pkl".format(split, stage), trainset)


def split_train_valid_example(stage):
    filename = "./underexpose_train/clean_underexpose_train_click-{}.csv".format(
        stage)

    with open(filename, newline='') as f:
        reader = csv.reader(f)
        ls = []
        for r in reader:
            ls.append(r)
        # print(len(ls))
        random.shuffle(ls)
        with open("dataset/clean_example_click-{}_train.csv".format(stage), "w", newline="") as fw:
            writer = csv.writer(fw)
            for r in ls[:int(len(ls) * .1)]:
                writer.writerow(r)
            print("for example:", len(ls[:int(len(ls) * .1)]))


def statistic_feat():
    filename = "./underexpose_train/underexpose_user_feat.csv"
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        ls_user = []
        for r in tqdm(reader):
            ls_user.append(r)
        user_set = set([i[0] for i in ls_user])
        # print("user_feat, item {}".format(len([i for i in user_set])))

    filename = "./underexpose_train/underexpose_item_feat.csv"
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        ls_item = []
        for r in tqdm(reader):
            ls_item.append(r)
        item_set = set([i[0] for i in ls_item])
        # print("item_feat, item {}".format(len([i for i in item_set])))

    return user_set, item_set


def statistic_T(stage, user_set, item_set):
    filename = "./underexpose_train/underexpose_train_click-{}.csv".format(
        stage)
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        train_list = []
        for r in tqdm(reader):
            train_list.append(r)
        user_set_train = set([i for i, _, _ in train_list])
        item_set_train = set([i for _, i, _ in train_list])
        # print("stage {}, total {}, user {}, item {}".format(stage, len(train_list), len([i for i in user_set_train]), len([i for i in item_set_train])))
    # return

    not_in_item_list = [i for i in item_set_train if i not in item_set]
    not_in_item_set = set(not_in_item_list)
    # print("user not in feat: {}".format(len([i for i in user_set_train if i not  in user_set])))
    # print("item not in feat: {}".format(len(not_in_item_list)))
    print("record whose item not in feat: {}".format(len([i for i in train_list if i[1] in not_in_item_set])))
    # print()
    return not_in_item_set


def read_using_pd():
    df = pd.read_csv("./underexpose_train/underexpose_item_feat.csv")
    # print(df.head(3))
    print(df.shape)


def make_feat_pkl():
    filename = "./underexpose_train/underexpose_item_feat.csv"
    with open(filename, newline='') as f:
        res = {}
        for r in tqdm(f):
            r = r.replace("[", "")
            r = r.replace("]", "")
            r = r.replace(" ", "")
            ls = r.split(",")
            item_id = str(ls[0])
            txt_vec = [float(i) for i in ls[1:129]]
            img_vec = [float(i) for i in ls[129:]]
            res[item_id] = (txt_vec, img_vec)
        pickle_write("dataset/underexpose_item_feat.pkl", res)

    def _calc(x):
        return int(x) if x else 0
    filename = "./underexpose_train/underexpose_user_feat.csv"
    with open(filename, newline='') as f:
        res = {}
        for r in tqdm(f):
            r = r.replace(u"\n", "1")
            r = r.replace("M", "1,0")
            r = r.replace("F", "0,1")
            r = r.replace(" ", "")
            ls = r.split(",")
            user_id = str(ls[0])
            ls_data = [_calc(i) for i in ls[1:]]
            res[user_id] = ls_data
        pickle_write("dataset/underexpose_user_feat.pkl", res)


def data_clean(stage):
    all_item_feat = pickle_read("dataset/underexpose_item_feat.pkl")
    item_set = [i for i in all_item_feat.keys()]

    old_filename = "./underexpose_train/underexpose_train_click-{}.csv".format(
        stage)
    new_filename = "./underexpose_train/clean_underexpose_train_click-{}.csv".format(
        stage)
    if not os.path.exists(old_filename):
        raise FileNotFoundError("No file: {}".format(old_filename))

    with open(old_filename, newline='') as f:
        reader = csv.reader(f)
        with open(new_filename, "w", newline='') as fw:
            count = 0
            for r in tqdm(reader):
                if str(r[1]) not in item_set:
                    count = count + 1
                    continue
                fw.write(",".join(r) + "\r\n")
            print("stage {}, delete record {}".format(stage, count))


if __name__ == "__main__":

    """
    [1,0] for Male
    [0,1] for Female
    """

    released_stage = 4

    # # TODO: 1. transform csv to pkl (dict form)
    # make_feat_pkl()

    # # TODO: 2. data cleaning to original data file
    # for i in range(released_stage + 1):
    #     data_clean(i)

    # # TODO: 3. data split on cleaned data
    # for i in range(released_stage + 1):
    #     split_train_valid(i)
        
    # TODO: 4. construct training set and validation set using data split results
    for i in range(released_stage + 1):
        construct_dataset(i)
