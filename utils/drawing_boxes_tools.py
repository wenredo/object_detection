from PIL import ImageDraw
import numpy as np

def letter_box_pos_to_original_pos(letter_pos, current_size, ori_image_size)-> np.ndarray:
    """
    Parameters should have same shape and dimension space. (Width, Height) or (Height, Width)
    :param letter_pos: The current position within letterbox image including fill value area.
    :param current_size: The size of whole image including fill value area.
    :param ori_image_size: The size of image before being letter boxed.
    :return:
    """
    letter_pos = np.asarray(letter_pos, dtype=np.float)
    current_size = np.asarray(current_size, dtype=np.float)
    ori_image_size = np.asarray(ori_image_size, dtype=np.float)
    final_ratio = min(current_size[0]/ori_image_size[0], current_size[1]/ori_image_size[1])
    pad = 0.5 * (current_size - final_ratio * ori_image_size)
    pad = pad.astype(np.int32)
    to_return_pos = (letter_pos - pad) / final_ratio
    return to_return_pos

def convert_to_original_size(box, size, original_size, is_letter_box_image):
    if is_letter_box_image:
        box = box.reshape(2, 2)
        box[0, :] = letter_box_pos_to_original_pos(box[0, :], size, original_size)
        box[1, :] = letter_box_pos_to_original_pos(box[1, :], size, original_size)
    else:
        ratio = original_size / size
        box = box.reshape(2, 2) * ratio
    return list(box.reshape(-1))

def draw_boxes(boxes, img, cls_names, detection_size, is_letter_box_image,separate_cls_score=False):
    draw = ImageDraw.Draw(img)
    w,h = img.size
    box_width = w // 400

    for cls, bboxs in boxes.items():
        color = tuple(np.random.randint(0, 256, 3))
        for box, score in bboxs:
            box = convert_to_original_size(box, np.array(detection_size),
                                           np.array(img.size),
                                           is_letter_box_image)
            draw.rectangle(box, outline=color, width=box_width)
            if separate_cls_score is False:
                draw.text(box[:2], '{} {:.2f}%'.format(
                    cls_names[cls], score * 100), fill=color)
            else:
                text_score_color = (1,1,1)
                # Class
                class_box = box[:2]
                font = draw.getfont()
                fontsize = font.getsize('{}'.format(cls_names[cls]))
                new_class_box = [class_box[0],
                                 class_box[1] - fontsize[1]]
                draw.text(new_class_box, '{}'.format(cls_names[cls]), fill=text_score_color)

                # Score
                score_box = [box[2],box[1]]
                fontsize = font.getsize('{:.0f}%'.format(score * 100))
                new_score_box = [score_box[0]-fontsize[0],
                                 score_box[1]-fontsize[1]]
                draw.text(new_score_box, '{:.0f}%'.format(score * 100), fill=text_score_color)
