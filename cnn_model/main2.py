from PIL import Image
import numpy as np
import tensorflow as tf
import sys 
import os
import h5py

# from uitl import np_extend
# os.chdir(os.path.dirname(sys.argv[0]))
sys.path.append(r'../')

abs_path = os.path.dirname(__file__) or os.path.abspath('.')
save_path = os.path.join(abs_path, 'model')



class image_transf:

    def __init__(self, file_name=None):
        self.img_list = None
        self.file_name =  file_name or 'test1_5.h5'
        self.f = h5py.File(self.file_name)

    def get_dir(self, path=None):
        path = path if path else abs_path

        tep_list = os.listdir(path)
        tep_list = list(map(lambda x:os.path.join(path, x), tep_list))
        train_list = []
        test_list = []
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

    def get_img_name(self, path=None):
        file_list = os.listdir(path) if path else os.listdir('.')
        img_list = list(map(lambda x:os.path.join(path,x), [x for x in file_list if 'jpg' in x]))
        return img_list

    def img_to_num(self, img_list):
        data = np.array(0)
        for i in img_list:
            im = Image.open(i)
            data_single = np.array(im)
            for j in range(5):
                temp_single = data_single[:, j*32: (j+1)*32,:]
                if data.any():
                    data = np.r_[data,[temp_single]]
                else:
                    data = np.array([temp_single])
            print(data.shape)
        return data

    def insert_h5py(self, name, data, string=False):
        dt = h5py.special_dtype(vlen=str) if string else None
        self.f.create_dataset(name, data=data, dtype=dt)

    def get_array(self, name):
        data = self.f[name]
        array = np.array(data)
        return array

    def get_y(self, path):
        with open(path, 'r') as f:
            y_str = f.read()
            set_y = y_str.split()
            data = np.array(0)
            for i in set_y:
                data_single = np.fromstring(i, dtype=np.uint8) - 48

                for j in data_single:

                    if data.any() or data.shape:
                        data = np.r_[data, [[j]]]
                    else:
                        data = np.array([[j]])
        print(data.shape)
        return data 

    def y_2_one_hot(self, set_y):
        data = np.array(0)
        print(set_y)
        for i in set_y:
          
            c = 0
            temp_one_hot = np.zeros([1,10])
            temp_one_hot[:,i[0]] =1
            # for j in i:
            #     temp_one_hot[:,c*10+j] = 1
            #     c += 1
            if data.any():
                data = np.r_[data,temp_one_hot]
            else:
                data = np.array(temp_one_hot)
        return data 

    def make_train_h5py(self, key='train'):

        t_list = self.get_dir()[key]
        pre_set_y = np.zeros(0)
        data = np.zeros(0)
        for i in t_list:
            
            img_list = self.get_img_name(path=i)
            data_per_dir = self.img_to_num(img_list)
            if data.any():
                data = np.concatenate((data,data_per_dir))
            else:
                data = data_per_dir

            y_path = os.path.join(i,'y.txt')
            y_per_dir = self.get_y(y_path)
            if pre_set_y.any():
                pre_set_y = np.concatenate((pre_set_y,y_per_dir))
            else:
                pre_set_y = y_per_dir
        print(pre_set_y.shape)
        y_one_hot = self.y_2_one_hot(pre_set_y)

        
        print(y_one_hot.shape)
        self.insert_h5py('%s_set_x'%key,data)
        self.insert_h5py('%s_set_y'%key,pre_set_y)
        self.insert_h5py('%s_set_y_one_hot'%key,y_one_hot)

    def get_all_test(self):
        img_list = []
        test_dir_list = self.get_dir()['test']
        for i in test_dir_list:
            temp_list = self.get_img_name(i)
            img_list.extend(temp_list)
        return img_list

    def get_all_y(self):
        
        test_dir_list = self.get_dir()['test']
        test_list = list(map(lambda x: os.path.join(x, 'y.txt'), test_dir_list))

        y_list = []
        for i in test_list:
            temp_y = self.get_y(i)
            y_list.extend(temp_y)
        return y_list

class cnn_model:

    def __init__(self):
        # self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        pass

    def initialize_parameters(self):
        w1 = tf.get_variable(name='w1', dtype=tf.float32, shape=(3, 3, 3, 16))
        w2 = tf.get_variable(name='w2', dtype=tf.float32, shape=(3, 3, 16, 64))
        return w1,w2
    #初始化偏置    

    def weight_variable(self, shape, name='weight'):
        init = tf.truncated_normal(shape, stddev=0.1)
        var = tf.Variable(initial_value=init, name=name)
        return var

    def bias_variable(self, shape, name='bias'):
        init = tf.constant(0.1, shape=shape)
        var = tf.Variable(init, name=name)
        return var
    #卷积    
    def conv2d(self, x, W, name='conv2d'):
        print('+++++++++++++++++x.shape++++++++++++++++++++++')
        print(x.shape)
        print('+++++++++++++++++w.shape++++++++++++++++++++++')
        print(W.shape)
        print('+++++++++++++++++++++++++++++++++++++++')
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name=name)
    #池化 
    def max_pool_2X2(self, x, name='maxpool'):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)


    def cul_loss(self, output, Y):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output))
        return loss

    def get_set(self, index=1, batch=5, test=False):
        a = image_transf()

        if test:
            set_x = a.get_array('test_set_x')
            set_y = a.get_array('test_set_y_one_hot')
            return set_x,set_y

        pre_set_x = a.get_array('train_set_x')
        pre_set_y = a.get_array('train_set_y_one_hot')
        n = pre_set_x.shape[0]
        index = index%n
        if index + batch <= n:
            set_x = pre_set_x[index : index+batch]
            set_y = pre_set_y[index : index+batch]
        else:
            remain = index + batch - n
            set_x = np.r_[pre_set_x[index : index+batch], pre_set_x[:remain]]+
            set_y = np.r_[pre_set_y[index : index+batch], pre_set_y[:remain]]

        return set_x,set_y


    def get_random_set(self, batch=5):
        a = image_transf()
        pre_set_x = a.get_array('train_set_x')
        pre_set_y = a.get_array('train_set_y_one_hot')
        num = len(pre_set_x)
        rand_index = np.random.choice(num, size=batch)
        rand_x = pre_set_x[rand_index]
        rand_y = pre_set_y[rand_index]

        return rand_x, rand_y

    def trans_one_to_five(self, data, y=False):
        print('=============transfor==================')
        print(data.shape)
        if y:
            n,c = data.shape
            pro_data = data.reshape(n*5, int(c/5))
        else:
            n,w,h,c = data.shape
            # print(n*5,w,int(h/5),c)
            pro_data = data.reshape(n*5 ,w, int(h/5), c)
            print(pro_data.shape)
            t_d = pro_data[0]
            t_d2 = data[0,:,0:32,:]
            print(t_d2.shape)
            Image.fromarray(t_d2).show()
        return pro_data 


    def run_model2(self, random_set=False):
        X = tf.placeholder(name='X', shape=(None, 120, 32, 3), dtype=tf.float32)
        Y = tf.placeholder(name='Y', shape=(None, 10), dtype=tf.float32)
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        W1,W2 = self.initialize_parameters()
        Z1 = self.conv2d(X, W1)
        A1 = tf.nn.relu(Z1)
        P1 = self.max_pool_2X2(A1)

        Z2 = self.conv2d(P1, W2)
        A2 = tf.nn.relu(Z2)
        P2 = self.max_pool_2X2(A2)

        W_fc1 = self.weight_variable([30*8*64, 256], 'W_fc1')
        B_fc1 = self.bias_variable([256], 'B_fc1')

        fc1 = tf.reshape(P2, [-1, 30*8*64])
        fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, W_fc1), B_fc1))
        fc1 = tf.nn.dropout(fc1, keep_prob)

        W_fc2 = self.weight_variable([256, 10], 'W_fc2')
        B_fc2 = self.bias_variable([10], 'B_fc2')
        output = tf.add(tf.matmul(fc1, W_fc2), B_fc2, 'output')

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output))
        optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

        predict = tf.reshape(output, [-1, 1, 10], name='predict')
        labels = tf.reshape(Y, [-1, 1, 10], name='labels')

        predict_max_idx = tf.argmax(predict, axis=2, name='predict_max_idx')
        labels_max_idx = tf.argmax(labels, axis=2, name='labels_max_idx')

        predict_correct_vec = tf.equal(predict_max_idx, labels_max_idx)
        accuracy = tf.reduce_mean(tf.cast(predict_correct_vec, tf.float32))

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            steps = 0
            index = 0
            batch = 128
            test_x,test_y = self.get_set(test=True)
            first_86 = True
            first_88 = True
            set_x,set_y = self.get_random_set(batch) if random_set else self.get_set(index=index)
            # set_x = self.trans_one_to_five(set_x)
            # set_y = self.trans_one_to_five(set_y, y=True)
            # print(set_x[10].shape)
            # print(set_y[10].shape)
            # test_img = set_x[10]
            # test_y = set_y[10]
            # im = Image.fromarray(test_img)
            # print('==========num==========',test_y)
            # im.show()

            for epoch in range(99999):
                index += batch
                set_x,set_y = self.get_random_set(batch) if random_set else self.get_set(index=index)
                set_x = set_x/255
                sess.run(optimizer, feed_dict={X : set_x, Y : set_y, keep_prob:0.75})
                if steps % 10 == 0:
                    acc_train = sess.run(accuracy, feed_dict={X :set_x, Y :set_y, keep_prob:1.0})
                    acc_test = sess.run(accuracy, feed_dict={X :test_x, Y :test_y, keep_prob:1.0})
                    print("steps=%d, accuracy_train=%f, accuracy_test=%f" % (steps, acc_train,acc_test))
                    
                    if acc_test > 0.985 and acc_train ==1 and first_86 and steps>2000:
                        print('90___ok!!!!!!!!!!!!!!')
                        saver.save(sess, os.path.join(save_path,"2019_4_24_86.model"), global_step=steps)
                        first_86 = False
                    if acc_test > 0.986and acc_train ==1 and first_88 and steps>2000:
                        print('95___ok!!!!!!!!!!!!!!')
                        saver.save(sess, os.path.join(save_path,"2019_4_24_88.model"), global_step=steps)
                        first_88 = False
                    if acc_test > 0.987 and acc_train ==1 and steps>1500:
                        print('98___ok!!!!!!!!!!!!!!')
                        saver.save(sess, os.path.join(save_path,"2019_4_24_98.model"), global_step=steps)
                        break
                # if steps % 50 == 0:
                #     im = Image.fromarray(set_x[3])
                #     y = set_y[3]
                #     # print('===================y=================', y)
                #     if y == np.array([0,0,0,0,0,0,0,0,0,1]):

                        
                #         im.show()
                steps += 1

Img = image_transf()
def run_auto():

    Img.make_train_h5py()
    Img.make_train_h5py(key='test')

def test_h5(ckeck_num, key='train',show_img=False):
    print('----------------x-shape----------------',Img.get_array('train_set_x').shape)
    print('----------------y-shape----------------',Img.get_array('train_set_y').shape)
    print('----------------t-x-shape----------------',Img.get_array('test_set_x').shape)
    print('----------------t-y-shape----------------',Img.get_array('test_set_y').shape)
    x_set = Img.get_array('%s_set_x'%key)
    x_set = Img.get_array('%s_set_x'%key)
    print(x_set.shape)
    y_set = Img.get_array('%s_set_y'%key)
    print(y_set.shape)
    one_hot = Img.get_array('%s_set_y_one_hot'%key)
    print(one_hot.shape)
    c = 0
    # print(y_set[1][0])
    l = {}
    for i in range(10):
        l[i] = 0
    
    for i in x_set:

        if y_set[c][0] == ckeck_num and show_img:
            print(one_hot[c])
            print('--------c------------',c,1+c//25,1+(c%25)//5,(c%25)%5)
            Image.fromarray(i).show()
        l[y_set[c][0]] += 1
        c += 1
    print(l)

def test_img_y(num=5, key='train'):
    x,w,h,c = Img.get_array('%s_set_x'%key).shape
    random_list = np.random.randint(0, x, num)
    t_img_data = Img.get_array('%s_set_x'%key)
    test_img_list = []
    for i in random_list:
        test_img_list.append(t_img_data[i])

    for j in range(len(test_img_list)):
        im = Image.fromarray(test_img_list[j])

        print('------------------number-----------------', Img.get_array('%s_set_y'%key)[random_list[j]])
        im.show()

    im = Image.fromarray(t_img_data[0])

    print('------------------number-----------------', Img.get_array('%s_set_y'%key)[0])
    # print(Img.get_array('%s_set_y'%key))
    im.show()

    
if __name__ == '__main__':


    ss = cnn_model()
    ss.run_model2(random_set=True)



    # run_auto()
    # test_h5(4)
    # test_img_y()
    # test_img_y(5, key='test')
