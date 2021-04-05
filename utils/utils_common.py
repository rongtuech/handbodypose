import numpy as np
import pickle
import json
import codecs
import os
import random
import time
import datetime
import shutil
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


class FileTool:
    @staticmethod
    def read_json_file(path):
        """
        read json file
        :param path:
        :return: json data
        """
        path = FileTool.__fix_path(path)
        json_data = open(path,'r').read()
        return json.loads(json_data)

    @staticmethod
    def read_text_file(i_file, encoding='utf-8'):
        """
        get list data in file
        :param i_file: filename need to be read
        :return list data
        """
        i_file = FileTool.__fix_path(i_file)
        with codecs.open(i_file, "r", "utf-8") as f:
            lines = f.read().splitlines()
        return lines

    @staticmethod
    def write_json_file(o_file, data, encoding='utf-8'):
        """
        write json file
        :param path:
        :return: json data
        """
        o_file = FileTool.__fix_path(o_file)
        with open(o_file, 'w') as out_file:
            out_file.write(data.dump())

    @staticmethod
    def write_json_list_file(o_file, data, encoding='utf-8'):
        """
        write json file
        :param path:
        :return: json data
        """
        # o_file = FileTool.__fix_path(o_file)
        with open(o_file, 'w') as out_file:
            out_file.write('\n'.join(map(lambda x: json.dumps(x), data)))

    @staticmethod
    def write_text_file(o_file, data, encoding='utf-8'):
        """
        write data to o_file
        :param o_file: out file path
        :param data:
        :param encoding:
        """
        o_file = FileTool.__fix_path(o_file)
        with open(o_file, 'w') as out_file:
            out_file.write('\n'.join(map(lambda x: str(x), data)))

    @staticmethod
    def __fix_path(path):
        if not os.path.exists(path):
            path = "%s%s" % ("../", path)
        return path

    @staticmethod
    def read_text_csv(i_file,elim = ',', encoding='utf-8'):
        """
        get list data in file
        :param i_file: filename need to be read
        :return list data
        """
        i_file = FileTool.__fix_path(i_file)
        lines = []
        with codecs.open(i_file, "r", "utf-8") as f:
            lines = f.read().splitlines()
        lines = map(lambda x: x.split(elim), lines)
        return lines

    def writeIntoCsvTemp(filename, listinfo):
        def process(data):
            str = "%s" % data[0]
            for ele in data[1:]:
                if type(ele) == type(""):
                    ele = ele.replace("\t", "  ")
                str += "\t%s" % ele
            return str

        # print cvs
        with open(filename, 'w') as out_file:
            out_file.write('\n'.join(map(lambda x: process(x), listinfo)))

    def makeCSVFromData(crawlable, crawling, listdata, title):
        csv = title.replace(",", "\t") + '\n'

        for ele in crawling:
            str = ""
            for e in ele:
                str += "'%s'\t" % e
            str += "'crawling...'"
            csv += str + '\n'

        for ele in crawlable:
            str = ""
            str += "'%s'\t'%s'" % (ele[0], ele[1])
            csv += str + '\n'

        if listdata != None:
            for ele in listdata:
                str = ""
                str += "'%s'\t'%s'" % (ele[0], ele[1])
                csv += str + '\n'

        return csv

    @staticmethod
    def writePickle(filename, data):
        pickle.dump(data, open(filename, "wb"))


    @staticmethod
    def readPickle(filename):
        return pickle.load(open(filename, "rb"))


    @staticmethod
    def move_file(init_path, target_dir):
        file_name = os.path.split(init_path)[1]
        shutil.move(init_path, os.path.join(target_dir,file_name))

    @staticmethod
    def get_all_file_paths_in_dir(root_dir):
        return [os.path.join(root_dir,file)
                for file in os.listdir(root_dir)
                if os.path.isfile(os.path.join(root_dir,file))]


class JsonParser:
    @staticmethod
    def parse_date(strDate="2017-10-05T05:55:32+0000"):
        strDate = strDate.replace("T"," ")
        if "+" in strDate:
            strDate = strDate[0:strDate.index("+")]
        return datetime.datetime.strptime(strDate, '%Y-%m-%d %H:%M:%S')

    @staticmethod
    def get_tag(json, tag):
        if tag in json:
            return json[tag]
        else:
            print("%s not in %s" % (tag, json))
            return None

    @staticmethod
    def get_neat_tags(json, tags):
        if json == None:
            return None
        if len(tags)<=0:
            return json
        elif tags[0] in json:
            return JsonParser.get_neat_tags(json[tags[0]],tags[1:])
        else:
            print("%s not in %s" % (tags[0], json))
            return None

    @staticmethod
    def get_loop_tag(json, tag):
        if "data" in json:
            list = []
            for ele in json["data"]:
                if JsonParser.get_tag(ele,tag) != None:
                    list.append(ele[tag])
            return list
        else:
            return []

    @staticmethod
    def find_all_tag(json, tag):
        res = []
        if type(json) is list:
            for ele in json:
                res.extend(JsonParser.find_all_tag(ele,tag))
        elif type(json) is dict:
            if tag in json:
                res.append(json[tag])
            for key in json:
                res.extend(JsonParser.find_all_tag(json[key],tag))
        return res


class RandomWait:
    @staticmethod
    def wait(numSeconds, range):
        rd = random.uniform(-range, range)
        time.sleep(numSeconds,rd)


class ListTool:
    @staticmethod
    def splitList(initList, numEle):
        resultList = []
        tempList = []
        ind = 0
        for ele in initList:
            ind+=1
            tempList.append(ele)
            if ind >= numEle:
                resultList.append(tempList)
                tempList=[]
                ind =0
        if len(tempList) > 0:
            resultList.append(tempList)

        return resultList




class NNTool:
    @staticmethod
    def cal_out_size(init_size, kernel_size, padding = (0, 0), stride=(1, 1), dilation=(1, 1)):
        temp_0 = (init_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) -1) / stride[0] + 1
        temp_1 = (init_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) -1) / stride[1] + 1

        return (temp_0, temp_1)


class VisualizeTool:
    @staticmethod
    def apply_function_with_random_image(folder_path, num_imgs, func):
        origin_imgs = []
        preprocess_imgs = []
        for i in tqdm(range(num_imgs)):
            origin_imgs.append(cv2.imread(os.path.join(folder_path, random.choice(os.listdir(folder_path)))))
            preprocess_imgs.append(func(origin_imgs[i]))

        fig = plt.figure()
        for i in range(num_imgs):
            fig.add_subplot(2, num_imgs, i*2 +1)
            plt.imshow(origin_imgs[i])
            fig.add_subplot(2, num_imgs, i*2 +2)
            plt.imshow(preprocess_imgs[i])
        plt.show()
