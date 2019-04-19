import src.server.Register as Register

from io import BytesIO
import base64
from PIL import Image
import numpy as np
import json
from datetime import *
from twisted.internet import reactor
import shutil
import txaio
import os

from autobahn.twisted.websocket import WebSocketServerFactory, WebSocketServerProtocol, listenWS

file_path = '/home/tedev0/FaceNet_Background'

# 图片和嵌入向量路径
face_data_path = file_path + '/datasets/mxic_dataset'
# Pre_trained model(FaceNet)路径
model_facenet_path = file_path + '/models/facenet/20180408-102900.pb'
# 分类器模型路径
model_classifier_path = file_path + '/models/keras_classifier.h5'

register = Register.Register(face_data_path, model_facenet_path, model_classifier_path)

prev_embedding = np.zeros((512,), dtype='float32')


# 当注册工号已存在于数据集时，需要删除原有数据集中的该工号图片，并删除所有embeddings文件和分类器模型文件，
# 之后重新初始化register
def reinitialize(newcomer_class_name):
    try:
        shutil.rmtree(face_data_path + os.path.join('/train', newcomer_class_name))
        shutil.rmtree(face_data_path + os.path.join('/test', newcomer_class_name))
        print(newcomer_class_name, ' images has been deleted')
    except FileNotFoundError:
        print(newcomer_class_name, ' not found')
    try:
        os.remove(os.path.join(face_data_path, 'train_emb.pkl'))
        os.remove(os.path.join(face_data_path, 'test_emb.pkl'))
        os.remove(model_classifier_path)
        print('train_emb.pkl... has been deleted')
    except FileNotFoundError:
        print('train_emb.pkl not found')
    global register
    register = Register.Register(face_data_path, model_facenet_path, model_classifier_path)
    print('重新初始化完成')


def get_overlays(faces):
    global prev_embedding
    response = []
    if faces is not None:
        for face in faces:
            h, w = face.container_image.shape[0], face.container_image.shape[1]
            face_bb = face.bounding_box
            print("face.container_image.shape:", face.container_image.shape)
            print("face.bounding_box:", face.bounding_box)

            face_bb = [face_bb[0] / w, face_bb[1] / h, face_bb[2] / w, face_bb[3] / h]

            distance = np.sqrt(np.sum(np.square(np.subtract(face.embedding, prev_embedding))))
            prev_embedding = face.embedding
            face_data = {'box': face_bb, 'name': face.name, 'prob': face.prob, 'distance': distance.tolist(),
                         'bounding_box': face.bounding_box.tolist()}

            response.append(face_data)
    return response


# def np_image_to_base64(np_img):
#     pil_img = Image.fromarray(np_img)
#     output_buffer = BytesIO()
#     pil_img.save(output_buffer, format='JPEG')
#     byte_data = output_buffer.getvalue()
#     base64_str = base64.b64encode(byte_data)
#     return base64_str


def base64_to_np_image(base64_img):
    base64de = base64.b64decode(base64_img)
    bytes_img = BytesIO(base64de)
    pil_image = Image.open(bytes_img)
    pil_image = pil_image.convert('RGB')
    np_img = np.array(pil_image)
    return np_img


# def base64_to_pil_image(base64_img):
#     base64de = base64.b64decode(base64_img)
#     bytes_img = BytesIO(base64de)
#     pil_img = Image.open(bytes_img)
#     pil_img = pil_img.convert('RGB')
#     return pil_img


def process_req(req, peer):
    dt_start = datetime.now()
    action_type = req.get('action_type')
    print('action_type: ', action_type)
    req_id = req.get('req_id')
    req_data = req.get('data')
    data = {}

    if action_type == '/identify_no_mtcnn':
        if type(req_data) == str:
            req_data = [req_data]
        images = req_data
        faces = []
        for image in images:
            np_img = base64_to_np_image(image)
            face = register.classify(np_img)
            faces.append(face)

        data = get_overlays(faces)

    elif action_type == '/validate':
        step = 4
        newcomer_class_name = req.get('name')
        if newcomer_class_name in register.class_names:
            reinitialize(newcomer_class_name)
            print('重新初始化register')
        images = req_data
        faces = []
        for image in images:
            np_img = base64_to_np_image(image)
            face = register.image2face(np_img)
            faces.append(face)

        # faces = cached_by_peer[peer]['face']
        if len(faces) > 8:
            newcomer_train_embeddings, newcomer_test_embeddings = [], []
            newcomer_train_faces, newcomer_test_faces = [], []
            for i, face in enumerate(faces):
                print(i, face, face.embedding.shape)
                embedding = face.embedding.reshape(1, 512)
                if i % step == 0:
                    print('test')
                    newcomer_test_embeddings.append(embedding)
                    newcomer_test_faces.append(face)
                else:
                    print('train')
                    newcomer_train_embeddings.append(embedding)
                    newcomer_train_faces.append(face)
            print("newcomer_test_embeddings.length: ", len(newcomer_test_embeddings))
            (accuracy, wrong1, wrong2), (model, class_names), (
                train_embeddings, train_labels), (test_embeddings, test_labels) = register.validate(
                np.concatenate(newcomer_train_embeddings),
                np.concatenate(newcomer_test_embeddings),
                newcomer_class_name
            )
            is_successful = False
            if wrong1 < 0.2:
                is_successful = True
                register.save_model(model, class_names)
                register.save_embeddings(
                    train_embeddings, train_labels,
                    test_embeddings, test_labels,
                    class_names
                )
                register.save_faces(newcomer_train_faces, newcomer_test_faces, newcomer_class_name)
            data = {'succ': is_successful, 'accuracy': accuracy, 'wrong': [wrong1, wrong2]}
    elif action_type == '/clear':
        del cached_by_peer[peer]
    else:
        data = {'succ': False}
    dur = datetime.now() - dt_start
    response = {'action_type': action_type, 'req_id': req_id, 'data': data, 'dur': dur.microseconds}
    return response


cached_by_peer = {}


class MyServerProtocol(WebSocketServerProtocol):

    def onConnect(self, request):
        # self.peer = request.peer
        print("=== Client connecting: {0}".format(request.peer))

    def onOpen(self):
        print("=== WebSocket connection open.")
        cached_by_peer[self.peer] = {'face': []}

    def onMessage(self, payload, is_binary):
        print('=== onMessage type peer', type(self.peer), self.peer)
        # print('http_request_path:',self.http_request_path)
        # prn_obj(self)
        if is_binary:
            print("Binary message received: {0} bytes".format(len(payload)))
        else:
            # print("Text message received: {0}".format(payload.decode('utf8')))
            # print("==== Text message received: ...")
            req = json.loads(payload.decode('utf8'))
            x = process_req(req, self.peer)
            # print('=======x',x)
            # with open("x_"+str(datetime.now())+".pkl", 'wb') as outfile:
            #    pickle.dump(x, outfile)

            face_data = json.dumps(x)
            print(datetime.now(), face_data, '开始发送给客户端')

            self.sendMessage(face_data.encode('utf8'), False)
            print(datetime.now(), '结束发送给客户端')

    def onClose(self, was_clean, code, reason):
        print("=== WebSocket connection closed: {0}".format(reason), self.peer)
        try:
            del cached_by_peer[self.peer]
        except FileNotFoundError:
            print('no cache found', self.peer)

        print('===cached_by_peer:', cached_by_peer)


if __name__ == '__main__':
    txaio.start_logging(level='debug')

    factory = WebSocketServerFactory(u"ws://192.168.164.196:8011")
    # by default, allowedOrigins is "*" and will work fine out of the
    # box, but we can do better and be more-explicit about what we
    # allow. We are serving the Web content on 8080, but our WebSocket
    # listener is on 9000 so the Origin sent by the browser will be
    # from port 8080...

    factory.setProtocolOptions(
        allowedOrigins=["*"]
    )
    factory.protocol = MyServerProtocol
    listenWS(factory)

    reactor.run()
