from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys 
import os
import time

abs_path = os.path.abspath('.')

class test_img:

    def __init__(self):
        self.img_list = None

    def get_img_name(self, path=None):
        file_list = os.listdir(path) if path else os.listdir('.')
        img_list = list(map(lambda x:os.path.join(path,x), [x for x in file_list if 'jpg' in x]))
        num  = len(img_list)
        bits = len(str(num))
        count = 1
        res_list = []
        for i in img_list:
            dir_path = os.path.split(i)[0]
            s_c = ('00000' + str(count))[-bits:]
            res_name = os.path.join(dir_path, s_c + ".jpg")
            try:
                os.rename(i, res_name)
            except:
                res_name = i
            res_list.append(res_name)
            count += 1
        return res_list

        # return img_list

    def get_all_test(self, file_path=None):
        img_list = []
        test_dir_list = self.get_dir() if not file_path else [file_path]
        print('--------------test_dir---------------')
        print(test_dir_list)
        print('-------------------------------------')
        for i in test_dir_list:
            temp_list = self.get_img_name(i)
            img_list.extend(temp_list)
        return img_list

    def get_dir(self, path=None):
        tep_list = os.listdir(path) if path else os.listdir('.')
        test_list = []
        for i in tep_list:
             if os.path.isdir(i) and 'test' in i and 'new' in i :
                test_list.append(os.path.join(abs_path,i))
        return test_list

    def img_to_num(self, img_list):
        data = np.array(0)
        for i in img_list:
            im = Image.open(i)
            data_single = np.array(im)
            if data.any():
                data = np.r_[data,[data_single]]
            else:
                data = np.array([data_single])
                print(data.shape)
        return data


class model_predict:
    def __init__(self):
        self.IMG = test_img()
        self.saver = tf.train.import_meta_graph(os.path.join(abs_path, "2019_4_24_98.model-1560.meta"))
        self.sess = tf.Session()
        self.saver.restore(self.sess, tf.train.latest_checkpoint(abs_path))

    def get_dir(self, path=None):
        path = path if path else abs_path

        tep_list = os.listdir(path)
        tep_list = list(map(lambda x:os.path.join(path, x), tep_list))
        train_list = []
        test_list = []
        print('-------------------run_dir----------------------')
        print(path)
        print('-------------------------------------------')
        for i in tep_list:
            if os.path.isdir(i) :
                if 'train' in i and 'new' in i:
                    train_list.append(os.path.join(abs_path,i))
                elif 'test' in i and 'new' in i:
                    test_list.append(os.path.join(abs_path,i))
        return {
            'train':train_list,
            'test':test_list
        }
    def run_predict(self, QT=None, file_path=None):
        pre_time = time.time()
        graph = tf.get_default_graph()
        input_holder = graph.get_tensor_by_name("X:0")
        keep_prob_holder = graph.get_tensor_by_name("keep_prob:0")
        predict_max_idx = graph.get_tensor_by_name("predict_max_idx:0")

        img_list = self.IMG.get_all_test(file_path)

        # y_list = Img.get_all_y()
        qt_data = []
        c = 0
        r_c = 0
        for img_path in img_list:
            img_data = self.IMG.img_to_num([img_path])
            predict_list = []
            for i in range(5):
                temp_data = img_data[:,:,i*32:(i+1)*32,:]

                # print('================', temp_data.shape)

                predict = self.sess.run(predict_max_idx, feed_dict={input_holder:temp_data, keep_prob_holder : 1.0})  
                predict_value = np.squeeze(predict)
                
                # print('-----------------answer--------------', int(predict_value))
                # Image.fromarray(temp_data[0]).show()
                predict_list.append(int(predict_value))

            print(img_path)
            per_img = Image.open(img_path)
            print(' ------------------预测值：{}--------------------------'.format(predict_list))
            if QT:
                qt_data.append(['识别图片---%s---, 识别结果为 : %s \n\n\n'%(os.path.split(img_path)[1], predict_list), img_path])
                
            # plt.imshow(per_img)
            # plt.axis('off')
            # plt.show()

            print('\n')
            c += 1

        spend_time = time.time() - pre_time
        if QT:
            QT.predict_data_format(qt_data)
            QT.predict_show('识别结束, 共识别 %s 张图，用时 %s 秒'%(c, spend_time))
            # print('正确率：%.2f%%(%d/%d)' % (r_c*100/len(img_list), r_c, len(img_list)))


    # def run_predict(self, QT=None, file_path=None):
    #     pre_time = time.time()
    #     graph = tf.get_default_graph()
    #     input_holder = graph.get_tensor_by_name("X:0")
    #     keep_prob_holder = graph.get_tensor_by_name("keep_prob:0")
    #     predict_max_idx = graph.get_tensor_by_name("predict_max_idx:0")

    #     img_list = self.IMG.get_all_test(file_path)

    #     # y_list = Img.get_all_y()
    #     with tf.Session() as sess:
    #         self.saver.restore(sess, tf.train.latest_checkpoint(abs_path))
    #         c = 0
    #         r_c = 0
    #         for img_path in img_list:
    #             img_data = self.IMG.img_to_num([img_path])
    #             predict_list = []
    #             for i in range(5):
    #                 temp_data = img_data[:,:,i*32:(i+1)*32,:]

    #                 # print('================', temp_data.shape)

    #                 predict = sess.run(predict_max_idx, feed_dict={input_holder:temp_data, keep_prob_holder : 1.0})  
    #                 predict_value = np.squeeze(predict)
                    
    #                 # print('-----------------answer--------------', int(predict_value))
    #                 # Image.fromarray(temp_data[0]).show()
    #                 predict_list.append(int(predict_value))

    #             print(img_path)
    #             per_img = Image.open(img_path)
    #             print(' ------------------预测值：{}--------------------------'.format(predict_list))
    #             if QT:
    #                 QT.predict_show('识别图片---%s---, 识别结果为 : %s \n'%(img_path, predict_list))
    #             # plt.imshow(per_img)
    #             # plt.axis('off')
    #             # plt.show()

    #             print('\n')
    #             c += 1

    #         spend_time = time.time() - pre_time
    #         if QT:
    #             QT.predict_show('识别结束, 共识别 %s 张图，用时 %s 秒'%(c, spend_time))
    #         # print('正确率：%.2f%%(%d/%d)' % (r_c*100/len(img_list), r_c, len(img_list)))
M_P = model_predict()
# M_P.run_predict()
# try:
    
# except Exception as e:
#     print(e)
# input('press any end')
# if __name__ == '__main__':
#     M_P.run_predict()