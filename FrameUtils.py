import numpy as np
import imagehash
import cv2
from PIL import Image
import copy


def milliseconds_to_time_str(timestamp):
    total_seconds = int(timestamp / 1000)
    milliseconds = int(timestamp % 1000)
    seconds = total_seconds % 60
    minutes = (total_seconds // 60) % 60
    hours = (total_seconds // 3600)
    time_str = "{:02d}:{:02d}:{:02d}:{:03d}".format(hours, minutes, seconds, milliseconds)
    return time_str


def seconds_to_time_str(timestamp):
    milliseconds = timestamp * 1000
    return milliseconds_to_time_str(milliseconds)


def get_ranges(stich_group, height, width):
    xmin = min(d['x'] for d in stich_group) / width
    ymin = min(d['y'] for d in stich_group) / height
    xmax = max(d['x'] for d in stich_group) / width
    ymax = max(d['y'] for d in stich_group) / height
    x_range = abs(xmax - xmin)
    y_range = abs(ymax - ymin)
    return x_range, y_range


def filter_by_stdv(arr, stds):
    mean = np.mean(arr)
    std_dev = np.std(arr)
    lower_bound = mean - stds * std_dev
    upper_bound = mean + stds * std_dev
    bool_arr = (arr >= lower_bound) & (arr <= upper_bound)
    return bool_arr


def fixbounds(x1, y1, x2, y2, h, w):
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = max(0, x2)
    y2 = max(0, y2)
    x1 = min(x1, w)
    y1 = min(y1, h)
    x2 = min(x2, w)
    y2 = min(y2, h)
    return x1, y1, x2, y2


def check_bounds(x1, y1, x2, y2):
    if y1 >= y2 or x1 >= x2:
        return False
    if y2 <= 0 or x2 <= 0:
        return False
    return True


def get_intersection_similarity(shift_x, shift_y, imgA, imgB):
    h, w, c = imgA.shape
    x1_a, y1_a, x2_a, y2_a = fixbounds(shift_x, shift_y, w + shift_x, h + shift_y, h, w)
    x1_b, y1_b, x2_b, y2_b = fixbounds(-1 * shift_x, -1 * shift_y, w - shift_x, h - shift_y, h, w)
    if not check_bounds(x1_a, y1_a, x2_a, y2_a):
        return None
    if not check_bounds(x1_b, y1_b, x2_b, y2_b):
        return None

    A = imgB[y1_a:y2_a, x1_a:x2_a]
    B = imgA[y1_b:y2_b, x1_b:x2_b]

    hashA = imagehash.phash(Image.fromarray(A[:, :, ::-1]))
    hashB = imagehash.phash(Image.fromarray(B[:, :, ::-1]))
    difference = abs(hashA - hashB)

    """print(difference)
    while True:
        cv2.imshow('window', A)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    while True:
        cv2.imshow('window', B)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break"""

    return difference


def intersection_check(shift_x, shift_y, imgA, imgB):
    h, w, c = imgA.shape
    x1_a, y1_a, x2_a, y2_a = fixbounds(shift_x, shift_y, w + shift_x, h + shift_y, h, w)
    x1_b, y1_b, x2_b, y2_b = fixbounds(-1 * shift_x, -1 * shift_y, w - shift_x, h - shift_y, h, w)
    if not check_bounds(x1_a, y1_a, x2_a, y2_a):
        return False
    if not check_bounds(x1_b, y1_b, x2_b, y2_b):
        return False
    return True


class FrameAccumulator:
    def __init__(self, initial_pic, flowestimator, frame_count, div_amount, stdv=1, angle_tolerance=.33):
        self.angle_tolerance = angle_tolerance
        self.stdv = stdv
        self.old_frame = {'frame_counter': 0, 'img': initial_pic}
        self.counter = 0
        self.history = np.zeros(frame_count, dtype=bool)
        self.frame_list = []
        self.div_amount = div_amount
        self.flowestimator = flowestimator
        self.state = {'op': "searching", 'counter': 0}
        self.collect = []
        self.moving_x = 0
        self.moving_y = 0
        self.collecting_counter_threshold = 2
        self.history = np.zeros(frame_count, dtype=bool)
        self.ranges = np.zeros(frame_count, dtype=bool)
        # use to hold indexes of us
        self.intermediary = np.ones(frame_count, dtype=int) * -1
        self.frame_list = []
        self.index = 0
        self.internal_filter = np.zeros(frame_count, dtype=bool)
        self.inference_height = flowestimator.get_height()
        self.inference_width = flowestimator.get_width()
        self.height_threshold = int(self.inference_height * .05)
        self.width_threshold = int(self.inference_width * .05)
        self.state = 'searching'
        self.holding = []
        self.stitch_candidate_groups = []
        self.reference_frame = {'img': initial_pic, 'x': 0, 'y': 0, 'frame_counter': 0}
        self.candidate_frame = None

    def panning_analysis(self, old_img, new_img):
        flow = self.flowestimator(old_img, new_img)
        intersects_passed = False
        similarity_passed = False
        # Analyze the optical flow to determine if it's a panning shot
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitude_filter = filter_by_stdv(magnitude.flatten(), self.stdv)
        angle_filter = filter_by_stdv(angle.flatten(), self.stdv)
        full_filter = np.logical_and(magnitude_filter, angle_filter)
        a_ratio = np.sum(full_filter) / len(full_filter)
        # distance_x = int(np.round(np.mean(flow[..., 0].flatten()[full_filter]), 0))
        # distance_y = int(np.round(np.mean(flow[..., 1].flatten()[full_filter]), 0))
        # mean_magnitude = np.mean(magnitude)
        mean_angle = np.mean(angle)
        tolerance = self.angle_tolerance
        angles_passed = np.all(np.isclose(angle, mean_angle, atol=tolerance))
        # angles_passed = True
        if angles_passed:
            # get the distances by filtering out extreme values
            # and filter for values within nth number of standard deviations
            magnitude_filter = filter_by_stdv(magnitude.flatten(), self.stdv)
            angle_filter = filter_by_stdv(angle.flatten(), self.stdv)
            full_filter = np.logical_and(magnitude_filter, angle_filter)
            distance_x = int(np.round(np.mean(flow[..., 0].flatten()[full_filter]), 0))
            distance_y = int(np.round(np.mean(flow[..., 1].flatten()[full_filter]), 0))



            # get the cropped overlap as determined by optical flow
            intersects_passed = intersection_check(distance_x, distance_y, old_img, new_img)
            if intersects_passed:
                # filter out false positive by making use when croped it looks like the same image
                try:
                    diff = get_intersection_similarity(distance_x, distance_y, old_img, new_img)
                    # filter out false positives by making sure when cropped along both overlaps it is a similar image
                    if diff < 15:
                        return True, distance_x, distance_y
                    else:
                        return False, None, None
                except Exception as ex:
                    print(ex)
                    similarity_passed = False
                    return False, None, None
            else:
                return False, None, None
        else:
            return False, None, None
    def add(self, img):
        # three frames to keep track of
        # reference frame the frame to get to use to get the distance
        # candidate frame is a frame that could be read if not passed a threshold
        if self.counter >= self.div_amount:
            # check to see if duplicate frame, still shot, or very slow panning show
            hashA = imagehash.phash(Image.fromarray(img['img'][:, :, ::-1]))
            hashB = imagehash.phash(Image.fromarray(self.reference_frame['img'][:, :, ::-1]))
            difference = abs(hashA - hashB)
            self.counter = 0
            if difference <= 6:
                # suspected same frame return
                return
            has_panned, distance_x, distance_y = self.panning_analysis(self.reference_frame['img'], img['img'])

            if has_panned:
                if distance_y<0:
                    a=abs(distance_y)
                    p=0
                height_passed = self.height_threshold < abs(distance_y)
                width_passed = self.width_threshold < abs(distance_x)
                passed = height_passed or width_passed
                reference_x = self.reference_frame['x']
                reference_y = self.reference_frame['y']
                reference_frame_counter = self.reference_frame['frame_counter']
                if passed:
                    self.holding += [{'x': reference_x, 'y': reference_y, 'frame_counter': reference_frame_counter}]
                    reference_x = reference_x + distance_x
                    reference_y = reference_y + distance_y
                    self.reference_frame = {'img': img['img'], 'frame_counter': img['frame_counter'], 'x': reference_x,
                                            'y': reference_y}
                    self.candidate_frame = None
                else:
                    self.candidate_frame = {'img': img['img'], 'frame_counter': img['frame_counter'],
                                            'x': reference_x + distance_x,
                                            'y': reference_y + distance_y}
                self.state = 'collecting'
            elif self.state == 'collecting':
                if self.candidate_frame is None:
                    reference_x = self.reference_frame['x']
                    reference_y = self.reference_frame['y']
                    reference_frame_counter = self.reference_frame['frame_counter']
                    self.holding += [{'x': reference_x, 'y': reference_y, 'frame_counter': reference_frame_counter}]
                    self.reference_frame = {'img': img['img'], 'frame_counter': img['frame_counter'], 'x': 0, 'y': 0}
                else:
                    reference_x = self.reference_frame['x']
                    reference_y = self.reference_frame['y']
                    reference_frame_counter = self.reference_frame['frame_counter']
                    self.holding += [{'x': reference_x, 'y': reference_y, 'frame_counter': reference_frame_counter}]
                    candidate_x = self.candidate_frame['x']
                    candidate_y = self.candidate_frame['y']

                    candidate_frame_counter = self.reference_frame['frame_counter']
                    self.holding += [{'x': candidate_x, 'y': candidate_y, 'frame_counter': candidate_frame_counter}]
                    self.candidate_frame = None
                    self.reference_frame = {'img': img['img'], 'frame_counter': img['frame_counter'], 'x': 0, 'y': 0}
                self.stitch_candidate_groups += [copy.copy(self.holding)]
                self.holding = []
                self.state = "searching"
            else:
                self.moving_x=0
                self.moving_y=0
                self.reference_frame = {'img': img['img'], 'frame_counter': img['frame_counter'], 'x': 0, 'y': 0}
                self.state = "searching"

        else:
            self.counter += 1


def get_first_bounds(shift_x, shift_y, w, h):
    return shift_x, shift_y, w, h


def outofdvsums(imgA, imgB, flowestimator, w, h):
    flow = flowestimator(imgA, imgA)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    magnitude_filter = filter_by_stdv(magnitude.flatten(), 1.5)
    angle_filter = filter_by_stdv(angle.flatten(), 1.5)
    full_filter = np.logical_and(magnitude_filter, 1.5)
    distance_x = int(np.round(np.mean(flow[..., 0].flatten()[full_filter]), 0))
    distance_y = int(np.round(np.mean(flow[..., 1].flatten()[full_filter]), 0))
    x1, y1, x2, y2 = get_first_bounds(distance_x, distance_y, w, h)
    magnitude_filter = filter_by_stdv(magnitude[y1:y2, x1:x2].flatten(), 1.5)
    angle_filter = filter_by_stdv(angle[y1:y2, x1:x2].flatten(), 1.5)
    full_filter = np.logical_and(magnitude_filter, 1.5)
    full_filter_inv = np.logical_not(full_filter)
    return np.sum(full_filter_inv)


# prunes stitich candidates
def prune_stitch_movement(stich_group, flow_estimator, threshold=.93, prefered_ratio=.3, height=480, width=640):
    group = copy.copy(stich_group)
    # first determine range
    xmin = min(d['moving_x'] for d in stich_group)
    ymin = min(d['moving_y'] for d in stich_group)
    xmax = max(d['moving_x'] for d in stich_group)
    ymax = max(d['moving_y'] for d in stich_group)

    x_range = int(xmax - xmin) + 1
    y_range = int(ymax - ymin) + 1

    base_area = height * width
    bool_mask = np.zeros((y_range + height, x_range + width), dtype=bool)
    # set so that all value at at origin
    for d in group:
        d['moving_x'] -= xmin
        d['moving_y'] -= ymin

        x1 = int(d['moving_x'])
        y1 = int(d['moving_y'])
        x2 = int(d['moving_x']) + width
        y2 = int(d['moving_y']) + height
        d['bounds'] = (x1, y1, x2, y2)

    first = group.pop(0)
    first['arearatio'] = 1.0
    stich_collect = [first]
    x1, y1, x2, y2 = first['bounds']
    bool_mask[y1:y2, x1:x2] = True
    group = np.array(group)

    other_bool_mask = np.zeros((y_range + height, x_range + width), dtype=bool)

    last_picked = None

    for grp in group:
        x1, y1, x2, y2 = grp['bounds']
        other_bool_mask[y1:y2, x1:x2] = True
    while True:
        area_ratios = []
        for grp in group:
            x1, y1, x2, y2 = grp['bounds']
            arearatio = np.sum(bool_mask[y1:y2, x1:x2]) / base_area
            # print(arearatio)
            area_ratios += [arearatio]
        area_ratios = np.array(area_ratios)
        # pre pruning
        prune_filter = area_ratios <= threshold
        group = group[prune_filter]
        area_ratios = area_ratios[prune_filter]
        if len(group) <= 0:
            break

        anyfnd = False
        if not last_picked is None:
            prune_indexes = []
            fsms = []

            for i, pr in enumerate(prune_filter):
                if prune_filter[i]:
                    prune_indexes += [i]
                    xdiff = abs(last_picked['x'] - group['x'])
                    ydiff = abs(last_picked['y'] - group['y'])
                    if xdiff > int(width * .9) or ydiff > int(height * .9):
                        fsms += [float('inf')]
                        continue
                    anyfnd = True
                    flow = flow_estimator(last_picked['img'], group[i]['img'])
                    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    magnitude_filter = filter_by_stdv(magnitude.flatten(), 1.5)
                    angle_filter = filter_by_stdv(angle.flatten(), 1.5)
                    full_filter = np.logical_and(magnitude_filter, 1.5)
                    sum_ = np.sum(np.logical_not(full_filter))
                    fsms += [sum_]
            fsms = np.array(fsms)
            sendlyerindx = np.argmin(fsms)
            closest_index = prune_indexes[sendlyerindx]
        if not anyfnd:
            absolute_diff = np.abs(area_ratios - .5)  # Calculate the absolute differences
            closest_index = np.argmin(absolute_diff)
            picked = group[closest_index]

        picked['arearatio'] = area_ratios[closest_index]
        x1, y1, x2, y2 = picked['bounds']
        bool_mask[y1:y2, x1:x2] = True
        stich_collect += [picked]
        # post pruning
        prune_filter = area_ratios <= threshold
        prune_filter[closest_index] = False
        group = group[prune_filter]
        if len(group) <= 0:
            break

    bool_mask = np.uint8(bool_mask) * 255
    other_bool_mask = np.uint8(other_bool_mask) * 255
    stich_length = len(stich_collect)
    # cv2.imwrite('withpruning.png',bool_mask)
    # cv2.imwrite('withoutpruning.png', other_bool_mask)
    return stich_collect


# prunes stitich candidates
def prune_stitch_candidates(stich_group, threshold=.93, prefered_ratio=.3, height=480, width=640):
    group = copy.copy(stich_group)
    # first determine range
    xmin = min(d['x'] for d in stich_group)
    ymin = min(d['y'] for d in stich_group)
    xmax = max(d['x'] for d in stich_group)
    ymax = max(d['y'] for d in stich_group)

    x_range = int(xmax - xmin) + 1
    y_range = int(ymax - ymin) + 1

    base_area = height * width
    bool_mask = np.zeros((y_range + height, x_range + width), dtype=bool)
    # set so that all value at at origin
    for d in group:
        d['x'] -= xmin
        d['y'] -= ymin

        x1 = int(d['x'])
        y1 = int(d['y'])
        x2 = int(d['x']) + width
        y2 = int(d['y']) + height
        d['bounds'] = (x1, y1, x2, y2)

    first = group.pop(0)
    first['arearatio'] = 1.0
    stich_collect = [first]
    x1, y1, x2, y2 = first['bounds']
    bool_mask[y1:y2, x1:x2] = True
    group = np.array(group)

    other_bool_mask = np.zeros((y_range + height, x_range + width), dtype=bool)
    for grp in group:
        x1, y1, x2, y2 = grp['bounds']
        other_bool_mask[y1:y2, x1:x2] = True
    while True:
        area_ratios = []
        for grp in group:
            x1, y1, x2, y2 = grp['bounds']
            arearatio = np.sum(bool_mask[y1:y2, x1:x2]) / base_area
            # print(arearatio)
            area_ratios += [arearatio]
        area_ratios = np.array(area_ratios)
        # pre pruning
        prune_filter = area_ratios <= threshold
        group = group[prune_filter]
        area_ratios = area_ratios[prune_filter]
        if len(group) <= 0:
            break

        absolute_diff = np.abs(area_ratios - .5)  # Calculate the absolute differences
        closest_index = np.argmin(absolute_diff)
        picked = group[closest_index]
        picked['arearatio'] = area_ratios[closest_index]
        x1, y1, x2, y2 = picked['bounds']
        bool_mask[y1:y2, x1:x2] = True
        stich_collect += [picked]
        # post pruning
        prune_filter = area_ratios <= threshold
        prune_filter[closest_index] = False
        group = group[prune_filter]
        if len(group) <= 0:
            break

    bool_mask = np.uint8(bool_mask) * 255
    other_bool_mask = np.uint8(other_bool_mask) * 255
    stich_length = len(stich_collect)
    # cv2.imwrite('withpruning.png',bool_mask)
    # cv2.imwrite('withoutpruning.png', other_bool_mask)
    return stich_collect