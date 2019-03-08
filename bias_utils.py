import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import glob
from collections import defaultdict

# Read original BDD val labels and only select labels with pedestrians
def format_labels(images):
    format_labels = []
    for img in images:
        
        for l in img['labels']:
            if l['category'] != 'person':
                continue
            else:
                item = {}
                item['name'] = img['name']
                item['timeofday'] = img['attributes']['timeofday']
                item['occluded']  = l['attributes']['occluded']
                item['truncated'] = l['attributes']['truncated']
                item['bbox'] = [l['box2d']['x1'], l['box2d']['y1'], l['box2d']['x2'], l['box2d']['y2']]
                
                format_labels.append(item)
    return format_labels

def group_by_key(detections, key):
    groups = defaultdict(list)
    for d in detections:
        groups[d[key]].append(d)
    return groups

def load_frozen_graph(frozen_graph_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph
            
def load_image_into_numpy_array(image):
    (width, height) = image.size
    return np.array(image.getdata()).reshape(height, width, 3).astype(np.uint8)

def run_inference_for_single_image(image, graph):
    """ Mainly based off of tensorboard's run_inference_for_single_image code"""
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                  'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
    return output_dict

def show_bbox(img, bbox, outputpath):
    for i, b in enumerate(bbox):
        ymin, xmin, ymax, xmax = b
        (height, width, channel) = img.shape
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)
    cv2.imwrite(outputpath, img)

def rescale_bbox(height, width, bounding_boxes):
    
    scaled_ymin = np.asarray(bounding_boxes[:,0] * height)
    scaled_xmin = np.asarray(bounding_boxes[:,1] * width)
    scaled_ymax = np.asarray(bounding_boxes[:,2] * height)
    scaled_xmax = np.asarray(bounding_boxes[:,3] * width)
    scaled_bbox = np.vstack((scaled_ymin, scaled_xmin, scaled_ymax, scaled_xmax)).T
    return scaled_bbox


def run_inference_show_bbox(frozen_graph_path, inputdir, outputdir, model_prefix):
    """ Run inference on all images in inputdata using provided frozen graph. copy images and draw bounding boxes into outputdir
    Args
       frozen_graph_path - frozen model
       inputdata - list of paths to input images
       model_prefix - prefix of the frozen model
    Returns:
       detection_result - detected class and bounding box
    """
    # load the frozen model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        
    detection_result = []
    inputdata = glob.glob(inputdir + '/*jpg')
    
    for input_image_path in inputdata:
        filename = input_image_path.replace(inputdir + '/', '')

        # open image
        image = Image.open(input_image_path)

        # prepare images to feed into the model 
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        output_dict['name'] = filename

        # rescale pixels to full range of width and height
        img = cv2.imread(input_image_path)
        height, width, channel = img.shape
        output_dict['scaled_detection_boxes'] = rescale_bbox(height, width, output_dict['detection_boxes'])

        # Draw detected bounding boxes and store in outputdir
        det_image_path = outputdir + '/det_' + model_prefix + '_' + filename
        show_bbox(img, output_dict['scaled_detection_boxes'], det_image_path)
        detection_result.append(output_dict)
        print(filename, "Number of detected pedestrians", output_dict['num_detections'])
    return detection_result

# format the prediction results for later use
def format_pred(predictions):
    format_pred = []
    for img in predictions:
        
        for idx in range(int(img['num_detections'])):
            item = {}
            item['name'] = img['name']
            item['bbox']  = [img['scaled_detection_boxes'][idx][1], img['scaled_detection_boxes'][idx][0],
                            img['scaled_detection_boxes'][idx][3], img['scaled_detection_boxes'][idx][2]]
            
            item['score'] = img['detection_scores'][idx]

            format_pred.append(item)
    return format_pred

def cat_pc(gt, predictions, thresholds):
    """
    Mainly based off of https://github.com/ucbdrive/bdd-data/blob/master/bdd_data/evaluate.py
    """
    num_gts = len(gt)
    image_gts = group_by_key(gt, 'name')
    image_gt_boxes = {k: np.array([[float(z) for z in b['bbox']]
                                   for b in boxes])
                      for k, boxes in image_gts.items()}
    image_gt_checked = {k: np.zeros((len(boxes), len(thresholds)))
                        for k, boxes in image_gts.items()}
    image_gt_occluded = {k: np.array([b['occluded']
                                   for b in boxes])
                      for k, boxes in image_gts.items()}
    
    num_gts_occluded = reduce(lambda x, y: int(x) + int(y), [i['occluded'] for i in gt])
    
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)

    # go down dets and mark TPs and FPs
    nd = len(predictions)
    tp = np.zeros((nd, len(thresholds)))
    fp = np.zeros((nd, len(thresholds)))
    
    nd_occ = len(predictions)
    tp_occ = np.zeros((nd, len(thresholds)))
    fp_occ = np.zeros((nd, len(thresholds)))
    
    nd_nocc = len(predictions)
    tp_nocc = np.zeros((nd, len(thresholds)))
    fp_nocc = np.zeros((nd, len(thresholds)))
    
    for i, p in enumerate(predictions):
        box = p['bbox']
        ovmax = -np.inf
        jmax = -1
        try:
            gt_boxes = image_gt_boxes[p['name']]
            gt_checked = image_gt_checked[p['name']]
            gt_occluded = image_gt_occluded[p['name']]
        except KeyError:
            gt_boxes = []
            gt_checked = None
            gt_occluded = None
            # todo: remove later
            continue
            
        if len(gt_boxes) > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(gt_boxes[:, 0], box[0])
            iymin = np.maximum(gt_boxes[:, 1], box[1])
            ixmax = np.minimum(gt_boxes[:, 2], box[2])
            iymax = np.minimum(gt_boxes[:, 3], box[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((box[2] - box[0] + 1.) * (box[3] - box[1] + 1.) +
                   (gt_boxes[:, 2] - gt_boxes[:, 0] + 1.) *
                   (gt_boxes[:, 3] - gt_boxes[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
            
        for t, threshold in enumerate(thresholds):
            if ovmax > threshold:
                if gt_checked[jmax, t] == 0:
                    if gt_occluded[jmax] == True:
                        tp_occ[i, t] = 1.
                    else:
                        tp_nocc[i, t] = 1.
                    tp[i, t] = 1
                    gt_checked[jmax, t] = 1
                else:
                    if gt_occluded[jmax] == True:
                        fp_occ[i, t] = 1.
                    else:
                        fp_nocc[i, t] = 1.
                    fp[i, t] = 1.
            else:
                if gt_occluded[jmax] == True:
                    fp_occ[i, t] = 1.
                else:
                    fp_nocc[i, t] = 1.
                fp[i, t] = 1.

    # compute precision recall
    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)
    recalls = tp / float(num_gts)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = np.zeros(len(thresholds))
    
    # occluded
    fp_occ = np.cumsum(fp_occ, axis=0)
    tp_occ = np.cumsum(tp_occ, axis=0)
    recalls_occ = tp_occ / float(num_gts_occluded) # probaly should divide by number of occluded
    precisions_occ = tp_occ / np.maximum(tp_occ + fp_occ, np.finfo(np.float64).eps)
    ap_occ = np.zeros(len(thresholds))
    
    # not occluded
    fp_nocc = np.cumsum(fp_nocc, axis=0)
    tp_nocc = np.cumsum(tp_nocc, axis=0)
    recalls_nocc = tp_nocc / float((num_gts - num_gts_occluded)) # probaly should divide by number of occluded
    precisions_nocc = tp_nocc / np.maximum(tp_nocc + fp_nocc, np.finfo(np.float64).eps)
    ap_nocc = np.zeros(len(thresholds))
    
    result = {}
    result['recall'] = np.squeeze(recalls[-1])
    result['recall_oc'] = np.squeeze(recalls_occ[-1])
    result['recall_noc'] = np.squeeze(recalls_nocc[-1])
    result['num_gts'] = num_gts
    result['num_gts_oc'] = num_gts_occluded
    
    return result

def evaluate_detection(gt, pred):
    thresholds = [0.75]
    cat_gt = group_by_key(gt, 'timeofday')
    results = {}
    
    for cat in cat_gt.keys():
        results[cat] = cat_pc(cat_gt[cat], pred, thresholds)
        #print(cat, 'Recall', results[cat]['recall'])
    return results

# calculate conditional difference
def calc_unexp_diff(day_result, night_result):
    mean_diff = day_result['recall'] - night_result['recall']
    occ_accp = (day_result['recall_oc'] + night_result['recall_oc'])/2
    nocc_accp = (day_result['recall_noc'] + night_result['recall_noc'])/2
    prob_occ_night = night_result['num_gts_oc'] / night_result['num_gts']
    prob_nocc_night = (night_result['num_gts'] - night_result['num_gts_oc']) / night_result['num_gts']
    prob_occ_day = day_result['num_gts_oc'] / day_result['num_gts']
    prob_nocc_day = (day_result['num_gts'] - day_result['num_gts_oc']) / day_result['num_gts']

    exp_diff = occ_accp * (prob_occ_day - prob_occ_night) + nocc_accp * (prob_nocc_day - prob_nocc_night)
    
    unexp_diff = mean_diff - exp_diff
    print('mean difference', mean_diff, ', exp diff', exp_diff, ', unexp_diff', unexp_diff)
    
    return mean_diff, exp_diff
