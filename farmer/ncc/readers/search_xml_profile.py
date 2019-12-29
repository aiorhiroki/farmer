import xml.etree.ElementTree as ET


def get_classname(xml_files):
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
