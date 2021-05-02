import xml.etree.ElementTree as ET
import pathlib
import tqdm


def generate_target_csv(
        xml_dir, img_dir, save_to='./target.csv', class_names=None):
    with open(save_to, 'a') as f:
        if xml_dir is None:
            for img_path in tqdm.tqdm(pathlib.Path(img_dir).iterdir()):
                if img_path.suffix.lower() not in [".png", ".jpeg", ".jpg"]:
                    continue
                target = [img_path, '', '', '', '', '']
                f.write(','.join([str(t) for t in target]) + '\n')
        else:
            for xml_path in tqdm.tqdm(pathlib.Path(xml_dir).iterdir()):
                if xml_path.suffix != ".xml":
                    if xml_path.suffix == ".csv":
                        with open(xml_path, 'r') as fr:
                            f.write(fr.read() + '\n')
                    continue
                tree = ET.parse(str(xml_path))
                root = tree.getroot()
                img_path = pathlib.Path(img_dir) / pathlib.Path(
                    root.find('filename').text
                ).name
                assert img_path.exists(), f"{img_path} does not exists."
                object_exits = False
                for object_tree in root.iter('object'):
                    name = object_tree.find('name').text
                    if len(name) == 0:
                        continue
                    elif class_names is not None and name not in class_names:
                        continue
                    else:
                        object_exits = True
                    for bndbox_tree in object_tree.iter('bndbox'):
                        xmin = bndbox_tree.find('xmin').text
                        ymin = bndbox_tree.find('ymin').text
                        xmax = bndbox_tree.find('xmax').text
                        ymax = bndbox_tree.find('ymax').text
                        target = [img_path, xmin, ymin, xmax, ymax, name]
                        f.write(','.join([str(t) for t in target]) + '\n')
                if not object_exits:
                    target = [img_path, '', '', '', '', '']
                    f.write(','.join([str(t) for t in target]) + '\n')


def get_classname(xml_files):
    # get all class names from pascal voc xml files
    classes = list()
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            class_name = obj.find('name').text
            if class_name in classes or int(difficult) == 1:
                continue
            classes.append(class_name)
    return classes


def convert_annotation(xml_file, classes):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotation = str()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        class_name = obj.find('name').text
        if class_name not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(class_name)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
             int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        annotation += " " + ",".join([str(a)for a in b]) + ',' + str(cls_id)

    return annotation
