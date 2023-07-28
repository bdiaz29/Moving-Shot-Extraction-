import tkinter as tk
from tkinter import filedialog, messagebox
import os
video_dir=None
output_dir=None
model_dir=None
use_tensorrt=False
root = tk.Tk()

# Padding for all widgets
padx, pady = 5, 5

# Video Path Button and Label

# Output Directory Button and Label
def select_output_dir():
    global output_dir
    output_dir = filedialog.askdirectory()
    outdir_label.config(text=output_dir)

# Output Directory Button and Label

def select_video():
    global video_dir
    video_dir = filedialog.askopenfilename(filetypes=(("Video files", "*.mp4 *.avi *.mkv *.webm *.mpg"), ("All files", "*.*")))
    video_label.config(text=video_dir)


def select_model_ONNX():
    global output_dir, use_tensorrt,model_dir
    use_tensorrt = False
    model_dir = filedialog.askdirectory()
    TFmodel_label.config(text=model_dir)
    TRTmodel_label.config(text="")

def select_model_TRT():
    global output_dir,use_tensorrt,model_dir
    use_tensorrt=True
    model_dir = filedialog.askopenfilename(filetypes=(("Video files", "*.trt *.engine"), ("All files", "*.*")))
    TRTmodel_label.config(text=model_dir)
    TFmodel_label.config(text="")


def start_extraction():
    global video_dir,output_dir,model_dir,use_tensorrt
    if video_dir is None:
        messagebox.showinfo("error","please set video directory" )
        return
    if output_dir is None:
        messagebox.showinfo("error","please set output directory" )
        return
    if model_dir is None:
        messagebox.showinfo("error","please set model directory" )
        return

    frame_skip=frame_skip_entry.get()
    min_range=min_range_entry.get()
    command=f'python extract_panning_shots.py --videodir "{video_dir}" --outdir "{output_dir}" --model_path "{model_dir}" --frame_skip {frame_skip} --min_range {min_range} '
    if use_tensorrt:
        command+='--use_trt'
    print(command)
    os.system(command)
    messagebox.showinfo("done", "finished extracting panning shots to " +output_dir)





outdir_button = tk.Button(root, text="output directory", command=select_output_dir)
outdir_button.grid(row=0, column=0, padx=padx, pady=pady)
outdir_label=tk.Label(root)
outdir_label.grid(row=0, column=1, padx=padx, pady=pady)


TFmodel_button = tk.Button(root, text="ONNX model path", command=select_model_ONNX)
TFmodel_button.grid(row=1, column=0, padx=padx, pady=pady)
TFmodel_label=tk.Label(root)
TFmodel_label.grid(row=1, column=1, padx=padx, pady=pady)

TRTmodel_button = tk.Button(root, text="TRT model path", command=select_model_TRT)
TRTmodel_button.grid(row=2, column=0, padx=padx, pady=pady)
TRTmodel_label=tk.Label(root)
TRTmodel_label.grid(row=2, column=1, padx=padx, pady=pady)

video_button = tk.Button(root, text="video path", command=select_video)
video_button.grid(row=3, column=0, padx=padx, pady=pady)
video_label=tk.Label(root)
video_label.grid(row=3, column=1, padx=padx, pady=pady)


# Min_range Entry
min_range_label = tk.Label(root, text="min_range (the amount the picture has to move to consider for a stich .1=10%)")
min_range_label.grid(row=4, column=0, padx=padx, pady=pady)

min_range_entry = tk.Entry(root)
min_range_entry.insert(0, ".1")  # default value
min_range_entry.grid(row=4, column=1, padx=padx, pady=pady)


# Frame_skip Entry
frame_skip_label = tk.Label(root, text="frame_skip (how many frames to skip before comparing frames,larger numbers means more frames skipped)")
frame_skip_label.grid(row=5, column=0, padx=padx, pady=pady)

frame_skip_entry = tk.Entry(root)
frame_skip_entry.insert(0, "16")  # default value
frame_skip_entry.grid(row=5, column=1, padx=padx, pady=pady)


start_button = tk.Button(root, text="start", command=start_extraction)
start_button.grid(row=6, column=0, padx=padx, pady=pady)



root.geometry("1800x600")
root.title("Panning Shot Extractor")
root.mainloop()
