#!/usr/bin/env bash

echo "是否需要删除特定工号人员图片(yes or no)"

read if_need_del
if [[ ${if_need_del} == "yes" ]]
then
cd ~/FaceNet_Background/datasets/mxic_dataset/train
echo "从以下工号中选择删除(ctrl+c退出)"
ls
if_ensure_del="no"
while [[ ${if_ensure_del} == "no" ]]
do
    echo "输入待删除的工号"
    read employee_id
    echo "确认要删除${employee_id}吗？(yes or no)"
    read result1
    if_ensure_del=${result1}
done
echo "正在删除${employee_id}"
cd ..
rm -r train/${employee_id} test/${employee_id}
echo "${employee_id}删除成功"
echo "正在自动删除相关特征编码文件和分类模型文件"
echo "启动服务会自动重新生成上述文件"
rm test_emb.pkl train_emb.pkl ../../models/keras_classifier.h5
else
echo "是否需要删除分类模型(yes or no)"
read if_need_del_classifier_model
if [[ ${if_need_del} == "yes" ]]
then
rm ~/FaceNet_Background/models/keras_classifier.h5
fi
fi

echo "是否需要启动服务(yes or no)"
read if_need_start_server
if [[ ${if_need_start_server} == "yes" ]]
then
source ~/tensorflow/bin/activate
export PYTHONPATH=/home/tedev0/FaceNet_Background/facenet
echo "server starting..."
python ~/FaceNet_Background/facenet/src/server/socket_server.py
fi