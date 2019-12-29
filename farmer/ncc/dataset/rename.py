import os
import glob
import xml.etree.ElementTree as ET


def get_zero_padding(filelist):
    return '_%0'+str(len(str(len(filelist))))+'d'


def rename(dirpath, prefix, start=1):
    """ Rename files for easy to understand.
    # Arguments
        dirpath: Path to the directory.
        prefix: Prefix of new file name.

    # Returns
        New name files, 'aaa_000b.ext'. (aaa=prefix, b=serial number, .ext=extension)
    """

    names = [x for x in sorted(os.listdir(dirpath)) if os.path.isfile(os.path.join(dirpath, x))]
    ext = list(set([os.path.splitext(x)[1] for x in names]))
    assert len(ext) == 1, 'It contains multiple extensions.'

    new_name = str(prefix) + get_zero_padding(names) + str.lower(ext[0])
    for idx, name in enumerate(names, start=start):
        os.rename(os.path.join(dirpath, name), os.path.join(dirpath, new_name) % idx)


def rename_with_xml(xml_dir, img_dir, prefix, start=1):
    """ Rename image files and xml files.
    # Arguments
        xml_dir: Path to the directory that contains xml.
        img_dir: Path to the directory that contains image.
        prefix: Prefix of new file name. (ex. 'pos', 'neg')

    # Returns
        New name files, 'aaa_000b.ext'. (aaa=prefix, b=serial number, .ext=extension)
    """
    xml_names = [x for x in sorted(os.listdir(xml_dir)) if x.endswith(('xml',))]
    img_names = [x for x in sorted(os.listdir(img_dir)) if x.endswith(('png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG'))]
    xml_ext = list(set([os.path.splitext(x)[1] for x in xml_names]))
    img_ext = list(set([os.path.splitext(x)[1] for x in img_names]))
    assert len(xml_ext) == 1, 'Xml directory contains multiple extensions.'
    assert len(img_ext) == 1, 'Image directory contains multiple extensions.'
    assert len(xml_names) == len(img_names), 'Mismatch length, %d xmls with %d images.' % (len(xml_names), len(img_names))

    new_img = str(prefix) + get_zero_padding(img_names) + str.lower(img_ext[0])
    new_xml = str(prefix) + get_zero_padding(xml_names) + str.lower(xml_ext[0])
 
    for idx, xml in enumerate(xml_names, start=start):
        tree = ET.parse(os.path.join(xml_dir, xml))
        root = tree.getroot()
        img = root.find('filename').text
        # Rename image file
        os.rename(os.path.join(img_dir, img), os.path.join(img_dir, new_img) % idx)
        # Rename xml tag
        root.find('filename').text = new_img % idx
        tree.write(os.path.join(xml_dir, xml))
        # Rename xml file
        os.rename(os.path.join(xml_dir, xml), os.path.join(xml_dir, new_xml) % idx)


def change_filename(xml_dir):
    """ Change path tag in the xml.
    """
    xmls = sorted([x for x in os.listdir(xml_dir) if x.endswith(('xml',))])

    for xml in xmls:
        tree = ET.parse(os.path.join(xml_dir, xml))
        root = tree.getroot()
        filename = root.find('filename').text
        filename = filename + '.jpg'
        root.find('filename').text = filename
        tree.write(os.path.join(xml_dir, xml))


def change_classname(xml_dir):
    """ Change path tag in the xml.
    """
    xmls = sorted([x for x in os.listdir(xml_dir) if x.endswith(('xml',))])

    for xml in xmls:
        tree = ET.parse(os.path.join(xml_dir, xml))
        root = tree.getroot()
        for object_tree in root.findall('object'):
            object_tree.find('name').text = 'scc'
        tree.write(os.path.join(xml_dir, xml))