import tensorflow as tf, numpy as np, imageio, matplotlib.pyplot as plt, cv2, traceback, os

ckpt = 'model1.ckpt'
"""
Convolutional Layer with Max Pooling and Local Response Normalization
"""
def conv_layer(in_layer,out_chan,size,sigma=0.01,b=0.01, cstrd = [1,2,2,1], kstrd=[1,1,1,1],pool=True):
    in_chan = in_layer.shape.as_list()[3]
    w = tf.Variable(tf.truncated_normal([size,size,in_chan,out_chan],stddev=sigma))
    b = tf.Variable(tf.constant(b, shape=[out_chan]))
    h = tf.nn.relu(tf.nn.conv2d(in_layer, w, strides= cstrd,padding='VALID')+b)
    p = tf.nn.max_pool(h,ksize = [1,2,2,1], strides = kstrd, padding='VALID')
    n = tf.nn.local_response_normalization(p, depth_radius=min(4,out_chan-2))
    n1 = tf.nn.local_response_normalization(h,depth_radius=min(4,out_chan-2))
    if pool:
        return w,b,h,p,n
    return w,b,h,n1


"""
Fully Connected Layer
"""
def conn_layer(in_layer,out_nodes,op_layer=False,sigma=0.01,b=0.01):
    i_s = in_layer.shape.as_list()
    #print(i_s)
    in_layer2 = in_layer
    if len(i_s) > 2:
        in_layer2 = tf.reshape(in_layer,[-1,i_s[1]*i_s[2]*i_s[3]])
        w = tf.Variable(tf.truncated_normal([i_s[1]*i_s[2]*i_s[3],out_nodes],stddev=sigma))
    else:
        w = tf.Variable(tf.truncated_normal([i_s[-1],out_nodes],stddev=sigma))
    b = tf.Variable(tf.constant(b, shape=[out_nodes]))
    h = tf.matmul(in_layer2,w)+b
    if not op_layer:
        h = tf.nn.relu(h)
    r = tf.nn.l2_loss(w)
    return w,b,h,r


"""
The architecture: 3 conv layers and  2 fc layers with dropout
"""
x = tf.placeholder(tf.float32, shape=[None,32*32*1])
y = tf.placeholder(tf.float32, shape=[None,101])
learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)
x_img = tf.reshape(x,[-1,32,32,1])
w1,b1,h1,p1,n1 = conv_layer(x_img,256,8,cstrd=[1,1,1,1])
w2,b2,h2,p2,n2 = conv_layer(p1,64,8,cstrd=[1,1,1,1])
w3,b3,h3,p3,n3 = conv_layer(p2,16,4,cstrd=[1,1,1,1])
w4,b4,h4,p4,n4 = conv_layer(p3,4,2,cstrd=[1,1,1,1])
w5,b5,h5,p5,n5 = conv_layer(p4,16,4,cstrd=[1,1,1,1])
w6,b6,h6,r6 = conn_layer(p5,1024)
h6_drop = tf.nn.dropout(h6,keep_prob)
w7,b7,h7,r7 = conn_layer(h6_drop,512)
h7_drop = tf.nn.dropout(h7,keep_prob)
w8,b8,y_,r8 = conn_layer(h7_drop,101,op_layer=True)


"""
Loss function: Softmax Cross Entropy
"""
loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
reg = r8+r7+r6
loss = loss0 + 0.01*reg

"""
Adaptive moments for training
"""
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)


"""
Compare predicted classes vs actual classes
"""
correct_prediction = tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(y_,1)),tf.float32)

"""
Saver object to save and restore variables
"""
saver = tf.train.Saver({'w1':w1,'b1':b1,'w2':w2,'b2':b2,'w3':w3,'b3':b3,'w4':w4,'b4':b4,'w5':w5,'b5':b5,'w6':w6,'b6':b6})

"""
Visualize output of a convolutional layer
"""
def visualize_layer(layer,sess):
    img = imageio.imread('./New Data/Test/1/umaschd1.pgm')
    ch = 1
    if len(img.shape) > 2:
        ch = min(3,img.shape[2])
        img = img[:,:,:ch]
    ip = cv2.resize(img,(128,128),interpolation=cv2.INTER_AREA).reshape(128*128*ch)
    unit = sess.run(layer,feed_dict = {x:[ip]})
##    m = unit[0][0][0][0]
##    for i in range(unit.shape[0]):
##        for j in range(unit.shape[1]):
##            for k in range(unit.shape[2]):
##                for l in range(unit.shape[3]):
##                    m = max(m,unit[i][j][k][l])
##    unit = unit*255/m
    cv2.imshow('frame',unit[0,:,:,:3])
    cv2.waitKey(1)

def visualize_layer_h5(layer,sess, img):
    ch = 1
##    if len(img.shape) > 2:
##        ch = min(3,img.shape[2])
##        img = img[:,:,:ch]
    img = np.reshape(img, [32*32*1])
    unit = sess.run(layer,feed_dict = {x:[img]})
    m = unit[0][0][0][0]
    for i in range(unit.shape[0]):
        for j in range(unit.shape[1]):
            for k in range(unit.shape[2]):
                for l in range(unit.shape[3]):
                    m = max(m,unit[i][j][k][l])
    unit = unit*255/m
    cv2.imshow('frame',unit[0,:,:,:3])

    cv2.waitKey(1)

"""
check validation accuracy
"""
def validate(net_loader,sess,test=False):
    acc = 0
    ls2 = 0
    acc_t = 0
    ls_t = 0
    test_data  = net_loader.test_data
    step = 1
    out_str = 'validation loss:'
    if test == True:
        step = 4
        out_str = 'test loss:'
    try:
        if net_loader.h5 == 'False':
            for i in range(0,len(test_data),step):
                #print(file, lab)
                ip = net_loader.get_single_img(test_data[i][0])
                lab = test_data[i][1]
                #print('predicted: ',np.argmax(sess.run(y_,feed_dict={x:[ip],keep_prob:1.0})))
                #print('actual: ',np.argmax(lab), ' ',lab)
                acc += correct_prediction.eval(feed_dict={x:[ip],y:[lab],keep_prob:1.0})
                ls2 += loss.eval(feed_dict={x:[ip], y:[lab], keep_prob:1.0})
            acc /= len(test_data)/step
            ls2 /= len(test_data)/step
            print(out_str,ls2, '; test acc: ',acc)
            return acc,ls2
        else:
            for i in range(0,test_data['images'].shape[0],step):
                #print(file, lab)
                lab = [0 for k in range(101)]
                ip = net_loader.get_single_img_h5(i)
                lab[i//1000] = 1
                #print('predicted: ',np.argmax(sess.run(y_,feed_dict={x:[ip],keep_prob:1.0})))
                #print('actual: ',np.argmax(lab), ' ',lab)
                acc += correct_prediction.eval(feed_dict={x:[ip],y:[lab],keep_prob:1.0})
                ls2 += loss.eval(feed_dict={x:[ip], y:[lab], keep_prob:1.0})
                visualize_layer_h5(h1,sess, net_loader.train_data['images'][0])
            acc /= (test_data['images'].shape[0])
            acc *= step
            ls2 /= (test_data['images'].shape[0])
            ls2 *= step
            print(out_str,ls2, '; test acc: ',acc)
            return acc,ls2
    except:
        traceback.print_exc()

def run_validate(net_loader):
    with tf.Session() as sess:
        try:
            saver.restore(sess, net_loader.model_dir + ckpt)
            print("Model reloaded successfully.")
            validate(net_loader, sess)
        except tf.errors.NotFoundError:
            print("Model " + ckpt + " not found, will create new file")

""" 
Train the model. Inputs: number of epochs, learning rate, train and test data, and whether to continue training model or start afresh
"""
def train(epochs,batch_sz,epsilon,net_loader,reload):
    print('epochs:',epochs,' learning rate:',epsilon,' batch size:', batch_sz,' reload:',reload)
    ls = []
    ls2 = []
    acc = []
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        acc_file = []
        prev_acc = -1
        prev_ls = 999999999
        if reload == 'True':
            try:
                saver.restore(sess, net_loader.model_dir+ckpt)
                print("Model reloaded successfully.")
                try:
                    acc_file = open(net_loader.model_dir+'prev_acc.txt','r')
                    prev_acc = np.float32(acc_file.readline().strip())
                    prev_ls = np.float32(acc_file.readline().strip())
                    acc_file.close()
                    print('previous test loss: ',prev_ls)
                    print('previous test accuracy: ',prev_acc)
                except OSError:
                    pass
            except tf.errors.NotFoundError:
                print("Model "+ckpt+" not found, will create new file")
        elif reload == 'False':
            print("'Reload' set to 'False', starting afresh")

        for e in range(epochs):
            print(e+1)
            l = 0
            a = 0
            for b in range(0,net_loader.train_size,batch_sz):
                print(b)
                if net_loader.h5 == 'True':
                    ip = net_loader.get_batch_random_h5(batch_sz)
                else:
                     ip = net_loader.get_batch_random(batch_sz)
                train_step.run(feed_dict={x:ip[0],y:ip[1],learning_rate:epsilon,keep_prob:0.5})
                l += loss.eval(feed_dict={x:ip[0],y:ip[1],keep_prob:1.0})
                a += np.mean(correct_prediction.eval(feed_dict={x:ip[0],y:ip[1],keep_prob:1.0}))
                #visualize_layer_h5(h1,sess, net_loader.train_data['images'][0])
            l /= net_loader.train_size/batch_sz
            a /= net_loader.train_size/batch_sz
            print("Train loss: ",l)
            print("Train acc: ",a)
            ls.append(l)
            if ((e+1)%(epochs/10) == 0) or epochs <= 50:
                a,l = validate(net_loader,sess)
                if len(acc)<=1:
                    if a>=prev_acc and l<prev_ls:
                        save_path = saver.save(sess, net_loader.model_dir+ckpt)
                        print('Model saved at ', save_path)                      
                elif a>=np.amax(acc) and l<np.amin(ls2):
                    save_path = saver.save(sess, net_loader.model_dir+ckpt)
                    print('Model saved at ', save_path)
                    acc_file = open(net_loader.model_dir+'prev_acc.txt','w')
                    acc_file.write(str(a[0])+'\n')
                    acc_file.write(str(l)+'\n')
                    acc_file.close()
                acc.append(a)
                ls2.append(l)
            
        a,l = validate(net_loader,sess,True)
##        save_path = saver.save(sess, net_loader.model_dir+ckpt)
##        print('Model saved at ', save_path)
        x1 = [i for i in range(len(ls))]
        x2 = [i for i in range(len(acc))]
        x3 = [i for i in range(len(ls2))]
        plt.figure('train loss')
        plt.plot(x1,ls)
        plt.figure('test acc')
        plt.plot(x2,acc)
        plt.figure('test loss')
        plt.plot(x3,ls2)
        plt.show()

"""
Test the model without training.
"""
def test(net_loader):
    with tf.Session() as sess:
        ckpt = 'model1.ckpt'
        saver.restore(sess, net_loader.model_dir+ckpt)
        acc = 0
        if net_loader.h5 == 'True':
            for file, lab in net_loader.test_data:
                img = net_loader.get_single_img_h5(file)
                cv2.imshow('frame',sess.run(p1,feed_dict={x:[img]})[0,:,:,:3])
                cv2.waitKey(1)
                acc += correct_prediction.eval(feed_dict={x:[img], y:[lab],keep_prob:1.0})
            acc/=net_loader.test_size
        else:
            for i in range(net_loader.test_data['images'].shape[0]):
                img = net_loader.get_single_img_h5(i)
                cv2.imshow('frame', sess.run(p1, feed_dict={x: [img]})[0, :, :, :3])
                cv2.waitKey(1)
                acc += correct_prediction.eval(feed_dict={x:[img], y:[lab],keep_prob:1.0})
            acc/=net_loader.test_size                
    print(acc)
