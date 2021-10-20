import numpy as np


def cross_val_split(dirs, counts, k=5, n_iter=15, mix_step=5):
    data = [dict(case=case, count=count)
            for case, count in zip(dirs, counts)]
    # sort raughly
    data = sorted(data, reverse=True, key=lambda x: x["count"])
    cross = [[v] for v in data[:k]]
    for v in data[k:]:
        sum_list = [sum([case_dic["count"] for case_dic in c]) for c in cross]
        min_sum_id = np.argmin(sum_list)
        cross[min_sum_id].append(v)

    # n swap sort
    swapping = False
    for step in range(n_iter):
        for i in range(k-1):
            for j in range(k-i):
                n = cross[-j-1]
                m = cross[i]
                n_sum = sum([case_dic["count"] for case_dic in n])
                m_sum = sum([case_dic["count"] for case_dic in m])
                if n_sum > m_sum:
                    large = n
                    small = m
                    diff = n_sum - m_sum
                else:
                    small = n
                    large = m
                    diff = m_sum - n_sum

                for i_l, v_l in enumerate(large):
                    for i_s, v_s in enumerate(small):
                        no_change_swap = int((step + 1) % mix_step == 0)
                        l_s_diff = v_l["count"] - v_s["count"]
                        if 0 < l_s_diff < diff + no_change_swap:
                            large[i_l] = v_s
                            small[i_s] = v_l
                            swapping = True
                            break
                    if swapping:
                        swapping = False
                        break

    cross_val_dirs = list()
    for case_list in cross:
        val_dirs = [d["case"] for d in case_list]
        cross_val_dirs.append(val_dirs)

    return cross_val_dirs


def representive_sequence(annotation_set, mask_dir):
    """
    step1: find image folder that has most images in all folders.
    step2: get the mode using envery 10000 frames

    args:
    mask_dir: mask dir name
    annoatton_set: [[image_path, mask_path], [image_path, mask_path], ...]
        - image_path: ~/root_folder/image_dir/mask/frame_000111.png

    returns:
    video_name: image_folder which has most images
    start_frame: 
    end_frame: 
    """

    img_dirs = [
        Path(img_path).parent.parent.name for img_path, _ in annotation_set]
    img_dirs = list(set(img_dirs))
    nb_img_per_dir = [len(glob(f"{img_dir}/{mask_dir}/*.png")) for img_dir in img_dirs]

    pass