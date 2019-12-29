from glob import glob
import os
import shutil


def separate_folder(target_dir, new_target_dir, folder_name, remove_remained=False):
    """
    同一フォルダ内にファイルがすべてあって`target_dir/*.png`、データセットをファイル名で分けているときに、
    データセットごとにフォルダを作成して、画像を'new_target_dir/datasest/images/*.png`に移動させる。
    画像ファイル名は、(動画の属性)_(フレーム番号).pngになっている必要がある。(動画の賊属性)の部分がフォルダー名になる。
    :parames folder_name: 元画像やラベル画像を入れるフォルダ名(images, labels, palettesなど)
    """
    if not os.path.exists(new_target_dir):
        os.makedirs(new_target_dir)
    image_files = glob(target_dir + '/*.png')
    for image_file in image_files:
        file_name = image_file.split('/')[-1]
        case_number = '_'.join(file_name.split('_')[:-1])
        case_dir = os.path.join(new_target_dir, case_number)
        new_folder = os.path.join(case_dir, folder_name)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        new_file_path = os.path.join(new_folder, file_name)
        shutil.copy(image_file, new_file_path)
        if remove_remained and os.path.exists(new_file_path):
            os.remove(image_file)


def move_folder(target_dir, new_target_dir, folder_name):
    """
    別々のtarget_dirにあるときに、移動させる。/target_dir/data_set/folder_name/image_file
    => /new_targert_dir/data_set/folder_name/image_file
    """
    case_dirs = os.listdir(target_dir)
    for case_dir in case_dirs:
        old_image_dir = os.path.join(target_dir, case_dir, folder_name)
        new_image_dir = os.path.join(new_target_dir, case_dir, folder_name)
        os.makedirs(new_image_dir)
        image_files = glob(old_image_dir + '/*.png')
        for image_file in image_files:
            new_image_file = os.path.join(new_image_dir, image_file.split('/')[-1])
            shutil.copy(image_file, new_image_file)
            if os.path.exists(new_image_file):
                os.remove(image_file)
