import cv2


def from_normalized(img, bb):
    '''
    This method changes percentage to pixel value regarding
    the width and the height of the image accordingly.
    '''
    h, w, c = img.shape
    # [0.48473296, 0.17299813, 0.5436626 , 0.2382411 ]
    ymin, xmin, ymax, xmax = bb
    xmin , xmax = int(xmin * w), int(xmax * w)
    ymin , ymax = int(ymin * h), int(ymax * h)

    return ((xmin, ymin), (xmax, ymax))



def draw_bbs_on_detection(img,
    dt_boxes,
    dt_classes,
    dt_scores,
    category_index,
    min_score_thresh=0.8):
    img = img.copy()
    num_vis = (dt_scores >= min_score_thresh).sum()
    dt_scores = dt_scores[:num_vis]
    dt_boxes = dt_boxes[:num_vis]
    dt_classes = dt_classes[:num_vis]
    
    # add bounding
    for i in range(num_vis):
        bb = from_normalized(img, dt_boxes[i])
        # bound the object
        cv2.rectangle(img, *bb, (0, 255, 0), 2)
        # add label
        bb_w = bb[1][0] - bb[0][0]
        xmin, ymin, xmax, ymax = bb[0][0]-1, bb[0][1]-20, bb[0][0]+bb_w+2, bb[0][1]
        # fill the background of text
        img[ymin:ymax, xmin:xmax] = (0, 255, 0)
        # add label
        score = format(dt_scores[i], '.2f')
        cv2.putText(img, f'{category_index[dt_classes[i]]}:{score}', (xmin-2, bb[0][1]-2), 4, .5, (0, 0, 0), 1)

    return img