import os
import math
import numpy as np
import tensorflow as tf
from scipy import misc
import src.facenet as facenet
import time
import pickle
import threading
from PIL import Image
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import EarlyStopping

gpu_memory_fraction = 0.5
debug = False
lock = threading.RLock


class Register:

    def __init__(self, dataset_path,
                 model_facenet_path,
                 model_classifier_path):
        print(datetime.now(), " Register初始化开始\n")
        self.encoder = Encoder(model_facenet_path)

        self.train_data_path = os.path.join(dataset_path, 'train/')
        self.test_data_path = os.path.join(dataset_path, 'test/')

        self.train_pkl_path = os.path.join(dataset_path, 'train_emb.pkl')
        self.test_pkl_path = os.path.join(dataset_path, 'test_emb.pkl')
        self.model_classifier_path = model_classifier_path
        '''
        因计算face特征码比较耗时，所以在初始化时就计算已有face图片的特征码以备后面用于新用户注册时训练分类器。
        train_pkl_path,test_pkl_path已有文件则直接读入到内存（省时，推荐）
        '''
        if os.path.exists(self.train_pkl_path):
            with open(self.train_pkl_path, 'rb') as infile:
                self.emb_train_array, self.train_labels, self.class_names = pickle.load(infile)
        else:
            self.emb_train_array, self.train_labels, self.class_names = self.encoder.get_embs_and_labels(
                self.train_data_path)
            with open(self.train_pkl_path, 'wb') as outfile:
                pickle.dump((self.emb_train_array, self.train_labels, self.class_names), outfile)

        if os.path.exists(self.test_pkl_path):
            with open(self.test_pkl_path, 'rb') as infile:
                self.emb_test_array, self.test_labels, _ = pickle.load(infile)
        else:
            self.emb_test_array, self.test_labels, _ = self.encoder.get_embs_and_labels(self.test_data_path)
            with open(self.test_pkl_path, 'wb') as outfile:
                pickle.dump((self.emb_test_array, self.test_labels, self.class_names), outfile)
        print(datetime.now(), ' 训练集测试集pkl文件加载完成\n')
        # 如果分类模型不存在，利用emb_array和labels训练模型,然后保存模型到model_classifier_path.
        if not os.path.exists(self.model_classifier_path):
            self.train_model(self.emb_train_array, self.train_labels).save(self.model_classifier_path)
        print(datetime.now(), ' 分类器模型加载完成\n')
        self.classifier = Classifier(model_classifier_path)
        print(datetime.now(), ' Register初始化完成\n')

    @staticmethod
    def train_model(emb_array, labels):
        x_train = emb_array
        print('x_train.size: ', x_train.shape, '\n', len(labels))
        y_train = np_utils.to_categorical(labels, num_classes=200)
        model = Sequential([
            Dense(200, input_dim=512),
            Activation('softmax'), ])
        rmsprop = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=rmsprop,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='loss', patience=3, verbose=0, min_delta=0.001)

        model.fit(x_train, y_train, epochs=100, batch_size=128, shuffle=True, callbacks=[early_stopping])
        return model

    def image2face(self, numpy_image):
        face_crop_size = 160
        face = Face()

        face.container_image = numpy_image
        face.bounding_box = np.zeros(4, dtype=np.int32)

        img_size = np.asarray(numpy_image.shape)[0:2]

        face.bounding_box[2] = img_size[1]
        face.bounding_box[3] = img_size[0]
        face.image = misc.imresize(numpy_image, (face_crop_size, face_crop_size), interp='bilinear')
        face.embedding = self.encoder.generate_embedding(face.image)

        return face

    def validate(self, newcomer_train_embeddings, newcomer_test_embeddings, newcomer_class_name):
        # 合并多个注册图片的特征码到历史特征码，所有合并均为临时数据不可更改self.属性
        start = time.time()

        newcomer_label = len(self.class_names)
        merged_class_names = self.class_names.copy()
        merged_class_names.append(newcomer_class_name)
        # class_name_num = len(merged_class_names)
        print('validate dur_1', time.time() - start)

        # 合并多个注册图片对应的标签到历史标签
        # 新图片的标签为新编号
        merged_train_emb_array = np.concatenate((self.emb_train_array, newcomer_train_embeddings))
        newcomer_train_labels = np.linspace(newcomer_label, newcomer_label, newcomer_train_embeddings.shape[0],
                                            dtype='int')
        merged_train_labels = np.concatenate((self.train_labels, newcomer_train_labels))
        print('validate dur_2', time.time() - start)

        merged_test_emb_array = np.concatenate((self.emb_test_array, newcomer_test_embeddings))
        newcomer_test_labels = np.linspace(newcomer_label, newcomer_label, newcomer_test_embeddings.shape[0],
                                           dtype='int')
        merged_test_labels = np.concatenate((self.test_labels, newcomer_test_labels))
        print('validate dur_3', time.time() - start)
        # 使用合并后的训练集训练一个新的分类器模型

        model_classifier = self.training_model(newcomer_train_embeddings, newcomer_train_labels)
        # 使用合并后的测试集评价新分类器模型的准确度
        print('validate dur_4', time.time() - start)

        accuracy, wrong1, wrong2 = self.evaluate_classifier(model_classifier, merged_test_emb_array, merged_test_labels,
                                                            merged_class_names,
                                                            newcomer_label)

        print('validate dur_5', time.time() - start)

        return (accuracy, wrong1, wrong2), (model_classifier, merged_class_names), (
            merged_train_emb_array, merged_train_labels), (
                   merged_test_emb_array, merged_test_labels)

    @staticmethod
    def evaluate_classifier(model, emb_array, labels, class_names, new_label):
        """
        依据测试集评价分类器准确度，以及图片的准确度
        """
        start = time.time()
        predictions = model.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        print('labels\n', labels)
        print('class_names\n', class_names)
        print('new_label\n', new_label)
        print('classify dur_1', time.time() - start)

        for i in range(len(best_class_indices)):
            print('%4d  %s VS %s: %.3f' % (
                i,
                class_names[labels[i]],
                class_names[best_class_indices[i]],
                best_class_probabilities[i]))
        print('classify dur_2', time.time() - start)

        new_indices_in_pred = np.where(best_class_indices == new_label)[0]
        print("预测组中新分类的位置：", new_indices_in_pred)
        new_indices_in_label = np.where(np.array(labels) == new_label)[0]
        print("标签组中新分类的位置：", new_indices_in_label)
        print('classify dur_3', time.time() - start)

        # 求差集，在new_indices_in_lable中但不在new_indices_in_pred中，是new_label的认为非new_label了，是为漏诊
        wrong1 = list(set(new_indices_in_label).difference(set(new_indices_in_pred)))
        # 求差集，在new_indices_in_pred中但不在new_indices_in_lable中，非new_label的认为new_label了，是为误诊
        wrong2 = list(set(new_indices_in_pred).difference(set(new_indices_in_label)))
        print('wrong1: ', wrong1, 'wrong2: ', wrong2)
        print(np.array(labels)[new_indices_in_label], best_class_indices[new_indices_in_label])
        accuracy = np.mean(np.equal(best_class_indices, labels))
        print('Accuracy: %.3f' % accuracy)
        print('classify dur_3', time.time() - start)

        return accuracy, len(wrong1) / len(new_indices_in_label), len(wrong2) / len(new_indices_in_label)

    def training_model(self, embeddings, labels):
        print('training_model:', embeddings.shape, "\n", labels)
        model = self.classifier.model
        x_train = embeddings
        y_train = np_utils.to_categorical(labels, num_classes=200)

        early_stopping = EarlyStopping(monitor='loss', patience=2, verbose=1, min_delta=0.05)
        model.fit(x_train, y_train, epochs=50, batch_size=2, callbacks=[early_stopping])

        return model

    def save_model(self, model, class_names):
        self.class_names = class_names
        self.classifier.model = model
        model.save(self.model_classifier_path)
        print(time.time(), 'Saved classifier model to file "%s"' % self.model_classifier_path)

    def save_embeddings(self, emb_train_array, train_labels, emb_test_array, test_labels, class_names):
        """
        保存现有特征码给未来训练用
        """
        print('save_embeddings')

        self.emb_train_array, self.train_labels = emb_train_array, train_labels
        self.emb_test_array, self.test_labels = emb_test_array, test_labels
        with open(self.train_pkl_path, 'wb') as outfile:
            pickle.dump((emb_train_array, train_labels, class_names), outfile)

        with open(self.test_pkl_path, 'wb') as outfile:
            pickle.dump((emb_test_array, test_labels, class_names), outfile)

    def save_faces(self, train_faces, test_faces, class_name):
        """
        保存新增face图片集，给未来调试、（如必要）生成特征码使用
        """
        self.save_to_dir(self.train_data_path, train_faces, class_name)
        self.save_to_dir(self.test_data_path, test_faces, class_name)

    @staticmethod
    def save_to_dir(m_path, faces, employee_id):
        print('save_faces')
        dir_path = os.path.join(m_path, employee_id)
        # 如果工号已存在则证明
        if os.path.exists(dir_path):
            print('exists:', dir_path)
        else:
            os.mkdir(dir_path)
        for face in faces:
            img = Image.fromarray(face.image)
            filename = employee_id + "_" + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + ".jpg"
            img_path = os.path.join(dir_path, filename)
            img.save(img_path)
            print(filename, ' has been saved')

    def classify(self, np_image):
        print(datetime.now(), ' 开始预测\n')
        face = self.image2face(np_image)
        print(datetime.now(), ' image2face\n')
        face.name, face.prob = self.classifier.classify(face.embedding, self.class_names)
        print(datetime.now(), ' 预测完成\n')
        if face.name != 'Unknown':
            # 分类器返回预测工号，判断训练集中已有的最新图片是否过期，如果过期，则添加此图片到训练集中。
            # TODO 测试集也需要更新图片（不能只更新训练集图片）未实现
            latest_image = max(os.listdir(os.path.join(self.train_data_path, face.name)))
            try:
                outdated_time = 300      # 设置过期时间，（300表示3个月过期）
                if int(datetime.now().strftime('%Y%m%d')) - int(latest_image.split('_')[1]) >= outdated_time:
                    print('开启线程添加图片')
                    update_image_thread = threading.Thread(target=self.update_image, args=(face,))
                    update_image_thread.start()
            except TypeError:
                print('Error typeError')
            except InterruptedError:
                print('Error InterruptedError')
            except IndexError:
                print('Error IndexError')

        return face

    def update_image(self, face):
        images = os.listdir(os.path.join(self.train_data_path, face.name))
        if len(images) >= 10:
            oldest_image = min(images)
            try:
                os.remove(os.path.join(os.path.join(self.train_data_path, face.name), oldest_image))
                print('remove the image ', oldest_image)
            except FileNotFoundError:
                print(oldest_image, ' not found')
        faces = [face]
        self.save_to_dir(self.train_data_path, faces, face.name)
        print(face.name, '的新图')


class Classifier:
    def __init__(self, classifier_model):
        self.model = load_model(classifier_model)
        print(datetime.now(), ' init Classifier success\n')

    def classify(self, embedding, class_names):
        if embedding is not None:
            predictions = self.model.predict_proba(embedding.reshape(1, 512))

            best_class_indices = np.argmax(predictions, axis=1)

            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            print("预测结果： ", best_class_probabilities, class_names[best_class_indices[0]])
            stranger_threshold = np.float32(0.8)
            if best_class_probabilities[0] < stranger_threshold:
                return "Unknown", stranger_threshold.item()
            return class_names[best_class_indices[0]], best_class_probabilities[0].item()


class Encoder:

    def __init__(self, facenet_model_checkpoint):
        # self.sess = tf.Session()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint)
        print(datetime.now(), ' init Encoder success\n')

    def generate_embedding(self, image):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(image)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]

    def get_embs_and_labels(self, data_dir, image_size=160, batch_size=100):
        # 从已对其的数据集获取 embeddings 和lables
        dataset = facenet.get_dataset(data_dir)
        paths, labels = facenet.get_image_paths_and_labels(dataset)
        class_names = [cls.name.replace('_', ' ') for cls in dataset]
        print("labels: ", labels)
        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(paths))
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        # Run forward pass to calculate embeddings
        print('Calculating features for images')
        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
        emb_array = np.zeros((nrof_images, embedding_size))

        with self.sess.as_default() as sess:
            for i in range(nrof_batches_per_epoch):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
        return emb_array, labels, class_names


class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.prob = None
        self.embedding = None
