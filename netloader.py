import os, traceback,random, cv2, imageio, h5py

class NetLoader:
    def __init__(self, model_dir, train_dir, test_dir, h5 = 'False'):
        self.model_dir = model_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train_size = 0
        self.test_size = 0
        self.class_nums = {}
        self.nums_class = {}
        self.num_classes = 0
        self.labels = []
        self.files_read = {}
        self.train_data = []
        self.test_data = []
        self.h5 = h5
        self.img_dim = 0
        i = 0
        for dirname,dirnames,filenames in os.walk(self.train_dir):
            if dirname != self.train_dir:
                self.class_nums[dirname]=i
                i += 1
        for j in range(i):
            l = []
            for k in range(i):
                if k==j:
                    l.append(1)
                else:
                    l.append(0)
            self.labels.append(l)
        i=0
        self.num_classes = len(self.class_nums)
        for dirname,dirnames,filenames in os.walk(self.test_dir):
            if dirname != self.test_dir:
                self.class_nums[dirname]=i
                i += 1
        for key in self.class_nums:
            self.nums_class[self.class_nums[key]] = key
        if h5 != 'True':
            self.create_data()
        else:
            self.create_data_h5()

    def create_data_h5(self):
        # for dirname, dirnames, filenames in os.walk(self.train_dir):
        #     i = 0
        #     ch = 1
        #     print("Select train set")
        #     for filename in filenames:
        #         print(i, ". ", filename)
        #         i += 1
            #input(ch)
        self.train_data = h5py.File(self.train_dir + '/food_c101_n10099_r64x64x1.h5', 'r')
        self.train_size = self.train_data['images'].shape[0]
        # for dirname, dirnames, filenames in os.walk(self.test_dir):
        #     i = 0
        #     ch = 2
        #     print("Select test set")
        #     for filename in filenames:
        #         print(i, ". ", filename)
        #         i += 1
            #input(ch)
        self.test_data = h5py.File(self.test_dir + '/food_test_c101_n1000_r64x64x1.h5', 'r')
        self.test_size = self.test_data['images'].shape[0]
        print('train data imgs: ', self.train_size)
        print('test data imgs: ', self.test_size)

    """
    Create the train and test data. Inputs are stored as paths to the images, and expected outputs as one-hot vectors.
    """
    def create_data(self):
        files = []
        train_data = []
        test_data = []
        for dirname, dirnames, filenames in os.walk(self.train_dir):
            for filename in filenames:
                if filename.endswith('jpg') or filename.endswith('pgm') or filename.endswith('png'):
                        train_data.append([os.path.join(dirname,filename),self.labels[self.class_nums[dirname]]])
                        self.train_size += 1
        for dirname, dirnames, filenames in os.walk(self.test_dir):
            for filename in filenames:
                if filename.endswith('jpg') or filename.endswith('pgm') or filename.endswith('png'):
                    test_data.append([os.path.join(dirname,filename),self.labels[self.class_nums[dirname]]])
                    self.test_size += 1
        self.train_data = train_data
        self.test_data = test_data
        print('train data imgs: ',self.train_size)
        print('test data imgs: ',self.test_size)


    """
    Get training data in a sequential manner
    """
    def get_batch(self,start,end):
        train_array = []
        train_labels = []
        for file,lab in self.train_data[start:end]:
            img = imageio.imread(file)
            ch = 1
            if len(img.shape) > 2:
                ch = min(3,img.shape[2])
                img = img[:,:,:ch]
            train_array.extend([cv2.resize(img,interpolation=cv2.INTER_AREA).reshape(128*128*ch)])
            train_labels.extend([lab])
        #print(train_array)
        #print(train_labels)
        return [train_array, train_labels]


    """
    Get training data randomly
    """
    def get_batch_random(self,batch_sz):
        train_array = []
        train_labels = []
        for i in range(batch_sz):
            j = random.randint(0,self.train_size-1)
            while j in self.files_read:
                j = random.randint(0,self.train_size-1)
            img = imageio.imread(self.train_data[j][0])
            ch = 1
            if len(img.shape) > 2:
                ch = min(3,img.shape[2])
            img = img[:,:,:ch]
            train_array.append(img.reshape(64*64*ch))
            train_labels.append(self.train_data[j][1])
            if len(self.files_read) == self.train_size:
                self.files_read = {}
        return [train_array, train_labels]

    def get_batch_random_h5(self,batch_sz):
        train_array = []
        train_labels = []
        for i in range(batch_sz):
            lab = [0 for i in range(101)]
            j = random.randint(0,self.train_size-1)
            while j in self.files_read:
                if len(self.files_read) == self.train_size:
                    break
                j = random.randint(0,self.train_size-1)
            img = self.train_data['images'][j]
            ch = 1
            if len(img.shape) > 2:
                ch = min(3,img.shape[2])
                img = img[:,:,:ch]
            train_array.append(img.reshape(64*64*ch))
            lab[j//101] = 1
            #print(j//100)
            train_labels.append(lab)
            self.files_read[j] = 1
        if len(self.files_read) == self.train_size:
            self.files_read = {}
        #print(",", len(self.files_read), " ", self.train_size)
        return [train_array, train_labels]

    def get_single_img(self,file):
        img = imageio.imread(file)
        ch = 1
        if len(img.shape) > 2:
            ch = min(3,img.shape[2])
            img = img[:,:,:ch]
        ip = img.reshape(64*64*ch)
        return ip
            
    def get_single_img_h5(self,file):
        img = self.train_data['images'][file]
        ch = 1
        if len(img.shape) > 2:
            if img.shape[2] > 1:
                ch = min(3,img.shape[2])
                img = img[:,:,:ch]
        ip = img.reshape(64*64*ch)
        return ip        

    




                          
                
    
