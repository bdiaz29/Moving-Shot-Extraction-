import argparse
import json
from FrameUtils import FrameAccumulator
import os
import cv2
from decord import VideoReader
from decord import cpu, gpu
import numpy as np
import time
from tqdm import tqdm
from FrameUtils import filter_by_stdv, get_ranges, prune_stitch_candidates, seconds_to_time_str
from PIL.PngImagePlugin import PngInfo
from PIL import Image
import random


def batch_generator(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def panning_process(args):
    file = args.videodir
    outdir = args.outdir
    starting_timestamp = args.starting_timestamp
    ending_timestamp = args.ending_timestamp
    ratio = float(args.ratio)
    min_range = float(args.min_range)
    wrt_skipped = open(outdir + '/skipped.txt', 'a')
    wrt_failed = open(outdir + '/failed.txt', 'a')
    stich_group_candidates = []

    if args.use_trt:
        message=''
        import_libs = False
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            import_libs=True
            from MLUtils import Raft_TRT
            flowestimator = Raft_TRT(args.model_path)
        except Exception as ex:
            print(ex)
            print("resoring to onnx models")
    else:
        from MLUtils import RaftOnnx
        flowestimator=RaftOnnx(args.model_path)

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    cap = cv2.VideoCapture(file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Get the frame rate of the video
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    WIDTH = int(width * ratio)
    HEIGHT = int(height * ratio)
    cap.release()
    vr = VideoReader(file, ctx=cpu(0), width=flowestimator.get_width(), height=flowestimator.get_height())
    frame_count = len(vr)
    frame = vr.next().asnumpy()[:, :, ::-1]
    frame_rate = vr.get_avg_fps()
    print("collecting stich group candidates")
    time.sleep(.1)
    start_frame = 1
    if not starting_timestamp is None:
        start_frame = int((float(starting_timestamp) / 1000) * frame_rate)

    if not ending_timestamp is None:
        frame_count = min(int((float(ending_timestamp) / 1000) * frame_rate), frame_count)
    print("starting collection of stage ", start_frame)
    time.sleep(.2)
    pbar = tqdm(total=frame_count - start_frame)
    im = vr[start_frame - 1].asnumpy()[:, :, ::-1]

    acc = FrameAccumulator(im, flowestimator, frame_count, args.frame_skip, stdv=1.5, angle_tolerance=.4)
    for f in range(start_frame, int(frame_count / 1)):
        pbar.update(1)
        newframe = {'img': vr.next().asnumpy()[:, :, ::-1], 'frame_counter': f}
        acc.add(newframe)
    stitch_candidate_groups = acc.stitch_candidate_groups
    pruned_stitch_candidate_groups = []

    import pickle
    with open('filename2.pickle', 'wb') as handle:
        pickle.dump(stitch_candidate_groups, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmmp=[]
    for stich_group in stitch_candidate_groups:
        x_range, y_range = get_ranges(stich_group, flowestimator.get_height(), flowestimator.get_width())
        mx = max(x_range, y_range)
        if mx < args.min_range:
            frame_indexes = [d['frame_counter'] for d in stich_group]

            starting_frame = min(d['frame_counter'] for d in stich_group)
            ending_frame = max(d['frame_counter'] for d in stich_group)
            b_times = starting_frame * frame_rate
            e_times = ending_frame * frame_rate
            b_timestr = seconds_to_time_str(b_times)
            e_timestr = seconds_to_time_str(e_times)
            info_dict = {'file': file, 'frame_indexes': frame_indexes, 'b_timestr': b_timestr, 'e_timestr': e_timestr,
                         'starting_frame': starting_frame, 'ending_frame': ending_frame, 'x_range': x_range,
                         'y_range': y_range}
            info_json = json.dumps(info_dict)
            wrt = open(f'{outdir}/skipped.txt', 'a')
            message = f'{time.time()} {info_json}' + '\n'
            wrt.write(message)
            wrt.close()
            continue
        pruned_stitch_candidate_groups += [
            prune_stitch_candidates(stich_group, threshold=.93, prefered_ratio=.3, height=flowestimator.get_height(),
                                    width=flowestimator.get_width())]

    vr = VideoReader(file, ctx=cpu(0), width=WIDTH, height=HEIGHT)
    stitcher=stitcher = cv2.Stitcher.create(1)
    for stich_group in tqdm(pruned_stitch_candidate_groups):
        frame_indexes = [d['frame_counter'] for d in stich_group]
        x_range, y_range = get_ranges(stich_group, flowestimator.get_height(), flowestimator.get_width())

        img_list = [vr[d['frame_counter'] - 1].asnumpy()[:, :, ::-1] for d in stich_group]
        stiched_image=None
        print(f"attempting stich on {len(img_list)} sized image list")
        try:
            status_, stiched_image = stitcher.stitch(img_list)
        except:
            stiched_image = None
        starting_frame = min(d['frame_counter'] for d in stich_group)
        ending_frame = max(d['frame_counter'] for d in stich_group)
        b_times = starting_frame * frame_rate
        e_times = ending_frame * frame_rate
        b_timestr = seconds_to_time_str(b_times)
        e_timestr = seconds_to_time_str(e_times)
        info_dict = {'file': file, 'frame_indexes': frame_indexes, 'b_timestr': b_timestr, 'e_timestr': e_timestr,
                     'starting_frame': starting_frame, 'ending_frame': ending_frame, 'x_range': x_range,
                     'y_range': y_range}
        info_json=json.dumps(info_dict)
        if stiched_image is None:
            print("could not stich image information of failed stitch saved in failed.txt")
            wrt = open(f'{outdir}/failed.txt', 'a')
            message = f'{time.time()} {info_json}' + '\n'
            wrt.write(message)
            wrt.close()
        else:
            save_file = f'{outdir}/stiched_{random.randint(1000, 9999)}_{time.time()}.png'
            metadata = PngInfo()
            metadata.add_text("parameters", info_json)
            imgPIL = Image.fromarray(stiched_image[:, :, ::-1])
            imgPIL.save(save_file, pnginfo=metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video files.')
    parser.add_argument('--videodir', help='Video File Directory', required=True)
    parser.add_argument('--outdir', help='Output directory for processed video files', required=True)
    parser.add_argument("--starting_timestamp", default=None, help="when the start the video reading")
    parser.add_argument("--ending_timestamp", default=None, help="when to end the video reading")
    parser.add_argument("--ratio", default=1.0,
                        help="the ratio of what output size from original video size 2 is twice and .5 is haf the size")
    parser.add_argument("--min_range", default=.1, help="minimum range to allow stitching")
    parser.add_argument('--use_trt', action='store_true', help='Enable TensorRT')
    parser.add_argument('--frame_skip', default=18, help='How many frames to skip before checking if panning shot')
    parser.add_argument('--model_path', default='models/raft_things_iter20_240x320.onnx', help='path of onnx or tensor rt engine')

    args = parser.parse_args()
    panning_process(args)
