import os
import sys
import glob
import json
import shutil
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import yaml

class Manager:
    def __init__(self,args):
        self.args = args
        self.annotations = []
        self.id = 1
        self.ann_id = 0
        self.images = []
        self.img_id = 0
        self.classname_to_id = {args['class_name']: 1}
        self.categories = []

    def _init_categories(self):
        """
        初始化 COCO 的 标注类别

        例如：
        "categories": [
            {
                "supercategory": "hand",
                "id": 1,
                "name": "hand",
                "keypoints": [
                    "wrist",
                    "thumb1",
                    "thumb2",
                    ...,
                ],
                "skeleton": [
                ]
            }
        ]
        """

        for name, id in self.classname_to_id.items():
            category = {}

            category['supercategory'] = name
            category['id'] = id
            category['name'] = name
            # 21 个关键点数据
            category['keypoint'] = self.args['keypoints']
            # category['keypoint'] = ["wrist",
            #                         "thumb1",
            #                         "thumb2",
            #                         "thumb3",
            #                         "thumb4",
            #                         "forefinger1",
            #                         "forefinger2",
            #                         "forefinger3",
            #                         "forefinger4",
            #                         "middle_finger1",
            #                         "middle_finger2",
            #                         "middle_finger3",
            #                         "middle_finger4",
            #                         "ring_finger1",
            #                         "ring_finger2",
            #                         "ring_finger3",
            #                         "ring_finger4",
            #                         "pinky_finger1",
            #                         "pinky_finger2",
            #                         "pinky_finger3",
            #                         "pinky_finger4"]
            # category['keypoint'] = [str(i + 1) for i in range(args.join_num)]

            self.categories.append(category)

    def _get_keypoints(self, points, keypoints, num_keypoints):
        """
        解析 labelme 的原始数据， 生成 coco 标注的 关键点对象

        例如：
            "keypoints": [
                67.06149888292556,  # x 的值
                122.5043507571318,  # y 的值
                1,                  # 相当于 Z 值，如果是2D关键点 0：不可见 1：表示可见。
                82.42582269256718,
                109.95672933232304,
                1,
                ...,
            ],

        """

        if points[0] == 0 and points[1] == 0:
            visable = 0
        else:
            visable = 1
            num_keypoints += 1
        keypoints.extend([points[0], points[1], visable])
        return keypoints, num_keypoints


    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    def _image(self, obj, path):
        """
        解析 labelme 的 obj 对象，生成 coco 的 image 对象

        生成包括：id，file_name，height，width 4个属性

        示例：
             {
                "file_name": "training/rgb/00031426.jpg",
                "height": 224,
                "width": 224,
                "id": 31426
            }

        """

        image = {}
        #img_x = labelme.utils.img_b64_to_arr(obj.imageData)  # 获得原始 labelme 标签的 imageData 属性，并通过 labelme 的工具方法转成 array
        #image['height'], image['width'] = img_x.shape[:-1]  # 获得图片的宽高
        image['height'] = obj['imageHeight']
        image['width'] = obj['imageWidth']
        # self.img_id = int(os.path.basename(path).split(".json")[0])
        self.img_id = self.img_id + 1
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path).replace("json", self.args['pic_format'])

        return image



    def _annotation(self, bboxes_list, keypoints_list, json_path):
        """
        对于指定的json文件生成rectangle的ann，并填充coco 下的annotations键

        Args：
            bboxes_list： 矩形标注框
            keypoints_list： 关键点
            json_path：json文件路径

        """

        if len(keypoints_list) != self.args['joint_num'] * len(bboxes_list):
            print('you loss {} keypoint(s) with file {}'.format(self.args['joint_num'] * len(bboxes_list) - len(keypoints_list),
                                                                json_path))
            print('Please check ！！！')
            sys.exit()

        i = 0
        for object in bboxes_list:
            annotation = {}
            keypoints = []
            num_keypoints = 0

            label = object['label']
            bbox = object['points']
            annotation['id'] = self.ann_id
            annotation['image_id'] = self.img_id
            annotation['category_id'] = int(self.classname_to_id[label])
            annotation['iscrowd'] = 0
            annotation['area'] = 1.0
            annotation['segmentation'] = [np.asarray(bbox).flatten().tolist()]
            annotation['bbox'] = self._get_box(bbox) # [x1,y1,x2,y2]-->[xmin,ymin,w,h]


            if self.args['label_style'] == 'single':
                for keypoint in keypoints_list[ i * self.args['joint_num']: (i + 1) * self.args['joint_num'] ]:
                    point = keypoint['points']
                    annotation['keypoints'], num_keypoints = self._get_keypoints(point[0], keypoints, num_keypoints)
                annotation['num_keypoints'] = num_keypoints

            elif self.args['label_style'] == 'group':
                kb_lists = [] #  corresponding keypoints for each rectangle # list of dicts
                rnum = len(bboxes_list)  # num of rectangles in each json
                for j in range(self.args['joint_num']):
                    kb_lists.append(keypoints_list[j*rnum + i])
                for keypoint in kb_lists:
                    point = keypoint['points'] # point: [ [x,y] ]
                    annotation['keypoints'], num_keypoints = self._get_keypoints(point[0], keypoints, num_keypoints)
                annotation['num_keypoints'] = num_keypoints

            i += 1
            self.ann_id += 1
            self.annotations.append(annotation)

    def _to_coco(self, json_path_list):
        """
        Labelme 原始标签转换成 coco 数据集格式，生成的包括标签和图像

        Args：
            json_path_list：原始数据集的目录

        """

        self._init_categories()

        for json_path in tqdm(json_path_list):
            #obj = labelme.LabelFile(filename=json_path)  # 解析一个标注文件
            obj = load_json(json_path)
            self.images.append(self._image(obj, json_path))  # 解析图片
            shapes = obj['shapes']  # 读取 labelme shape 标注

            bboxes_list, keypoints_list = [], []
            for shape in shapes:
                if shape['shape_type'] == 'rectangle':  # bboxs
                    bboxes_list.append(shape)  # keypoints
                elif shape['shape_type'] == 'point':
                    keypoints_list.append(shape)

            self._annotation(bboxes_list, keypoints_list, json_path)

        keypoints = {}
        keypoints['info'] = {'description': 'Lableme Dataset', 'version': 1.0, 'year': 2021}
        keypoints['license'] = ['BUAA']
        keypoints['images'] = self.images
        keypoints['annotations'] = self.annotations
        keypoints['categories'] = self.categories
        return keypoints

    def _save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    def _make_trainset(self, train_paths):
        if self.args['convert_type'] == 'labelme2coco':
            train_keypoints = self._to_coco(train_paths)  # train_keypoints: coco dict
            self._save_coco_json(train_keypoints,
                                os.path.join(self.args['output'], "coco", "annotations", "keypoints_train.json"))
            for file in train_paths:
                shutil.copy(file.replace("json", self.args['pic_format']),
                            os.path.join(self.args['output'], "coco", "train"))

        elif self.args['convert_type'] == 'labelme2yolo':
            self._to_yolo(json_paths=train_paths,flag='train')

    def _to_yolo(self,json_paths, flag):
        for json_path in tqdm(json_paths):
            obj = load_json(json_path)
            shapes = obj['shapes']  ## list of dict
            rect_num = int(len(shapes) / (self.args['joint_num'] + 1))
            width = obj['imageWidth']
            height = obj['imageHeight']
            txt_path,img_path = None, None
            if flag == "train":
                txt_path = os.path.join(self.args['output'], 'labels', 'train', str(self.id) + '.txt')
                img_path = os.path.join(self.args['output'], 'images', 'train',
                                        str(self.id) + '.' + self.args['pic_format'])
            elif flag == "val":
                txt_path = os.path.join(self.args['output'], 'images', 'val', str(self.id) + '.txt')
                img_path = os.path.join(self.args['output'], 'images', 'val',
                                        str(self.id) + '.' + self.args['pic_format'])

            with open(txt_path, 'w',encoding='utf-8') as file:
                if self.args['label_style'] == 'group':
                    for i in range(rect_num):
                        x_center = (shapes[i]['points'][0][0] + shapes[i]['points'][1][0]) / (2 * width)
                        y_center = (shapes[i]['points'][0][1] + shapes[i]['points'][1][1]) / (2 * height)
                        w = (shapes[i]['points'][1][0] - shapes[i]['points'][0][0]) / (1 * width)
                        h = (shapes[i]['points'][1][1] - shapes[i]['points'][0][1]) / (1 * height)
                        txt_write(file, self.classname_to_id[self.args['class_name']])
                        txt_write(file, x_center)
                        txt_write(file, y_center)
                        txt_write(file, w)
                        txt_write(file, h)
                        for j in range(1, self.args['joint_num']+1):
                            txt_write(file, shapes[rect_num * j + i]['points'][0][0] / width)
                            txt_write(file, shapes[rect_num * j + i]['points'][0][1] / height)
                        file.write('\n')

                elif self.args['label_style'] == 'single':
                    for i in range(rect_num):  ## cor to each row in txt file
                        x_center = (shapes[(self.args['joint_num'] + 1) * i]['points'][0][0] +
                                    shapes[(self.args['joint_num'] + 1) * i]['points'][1][0]) / (2 * width)

                        y_center = (shapes[(self.args['joint_num'] + 1) * i]['points'][0][1] +
                                    shapes[(self.args['joint_num'] + 1) * i]['points'][1][1]) / (2 * height)

                        w = (shapes[(self.args['joint_num'] + 1) * i]['points'][1][0] -
                             shapes[(self.args['joint_num'] + 1) * i]['points'][0][0]) / (1 * width)

                        h = (shapes[(self.args['joint_num'] + 1) * i]['points'][1][1] -
                             shapes[(self.args['joint_num'] + 1) * i]['points'][0][1]) / (1 * height)

                        txt_write(file, self.classname_to_id[self.args['class_name']])
                        txt_write(file, x_center)
                        txt_write(file, y_center)
                        txt_write(file, w)
                        txt_write(file, h)
                        for j in range(1, self.args['joint_num'] + 1):
                            txt_write(file, shapes[rect_num * i + j]['points'][0][0] / width)
                            txt_write(file, shapes[rect_num * i + j]['points'][0][1] / height)
                        file.write('\n')

            ## save img file
            img_dir = json_path.replace('json', self.args['pic_format'])
            shutil.copyfile(img_dir, img_path)
            self.id += 1

    def _make_valset(self,val_paths):
        if self.args['convert_type'] == 'labelme2coco':
            val_instance = self._to_coco(val_paths)
            self._save_coco_json(val_instance,
                                   os.path.join(self.args['output'], "coco", "annotations", "keypoints_val.json"))
            for file in val_paths:
                shutil.copy(file.replace("json", self.args['pic_format']),
                            os.path.join(self.args['output'], "coco", "val"))

        elif self.args['convert_type'] == 'labelme2yolo':
            self._to_yolo(json_paths=val_paths, flag='val')

    def _clear_props(self):
        self.__init__(self.args)
        pass

    def make_dataset(self):
        json_list_path = glob.glob(self.args['input'] + "/*.json")
        train_paths, val_paths = train_test_split(json_list_path, test_size=self.args['ratio'])
        self._make_trainset(train_paths=train_paths)
        if self.args['convert_type'] == 'labelme2coco':
            self._clear_props()
        self._make_valset(val_paths=val_paths)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        jd = json.load(file)
        return jd

def txt_write(file, data):
    file.write(str(round(data, 3)))
    file.write(' ')
    return file

def read_yaml(cpath):
    with open(cpath, encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    return config

def init_dir(cfg):
    """
    初始化COCO数据集的文件夹结构；
    coco - annotations  #标注文件路径
         - train        #训练数据集
         - val          #验证数据集
    Args：
        base_path：数据集放置的根路径
    """
    if cfg['convert_type'] == 'labelme2coco':
        if not os.path.exists(os.path.join(cfg['output'], "coco", "annotations")):
            os.makedirs(os.path.join(cfg['output'], "coco", "annotations"))
        if not os.path.exists(os.path.join(cfg['output'], "coco", "train")):
            os.makedirs(os.path.join(cfg['output'], "coco", "train"))
        if not os.path.exists(os.path.join(cfg['output'], "coco", "val")):
            os.makedirs(os.path.join(cfg['output'], "coco", "val"))
    elif cfg['convert_type'] == 'labelme2yolo':
        if not os.path.exists(os.path.join(cfg['output'], "images", "train")):
            os.makedirs(os.path.join(cfg['output'], "images", "train"))
        if not os.path.exists(os.path.join(cfg['output'], "images", "val")):
            os.makedirs(os.path.join(cfg['output'], "images", "val"))
        if not os.path.exists(os.path.join(cfg['output'], "labels", "train")):
            os.makedirs(os.path.join(cfg['output'], "labels", "train"))
        if not os.path.exists(os.path.join(cfg['output'], "labels", "val")):
            os.makedirs(os.path.join(cfg['output'], "labels", "val"))

    else:
        print("convert type is wrong, please check.")


if __name__ == '__main__':
    cfg = read_yaml("D:/labelme2coco/MyCoco.yaml")
    ### 创建文件夹
    init_dir(cfg)
    ### 生成数据
    mag_train = Manager(cfg)  # 构造数据集生成类
    mag = Manager(cfg)
    mag.make_dataset()



