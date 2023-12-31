<h1> Extraction And  Stiching of Panning Shots from Video Files  </h1>

<p>This repository provides a solution for automatically detecting and extracting panning shots in video files. Panning shots are a common technique in animation where the camera or background moves horizontally, creating an illusion of motion and depth. This technique not only adds visual interest but also serves as a cost-saving measure by minimizing the need to animate individual elements separately. </p>

<p> However a lot of high quality shots are left out for use in datasets as a result of this animation technique. This repos seeks to recover those shots by analyzing a video identifying the parts of the video that are likely panning shots and stitching together in an attempt to recover the original full picutre used for it. 
This is achiveced through a combination of Machine learning/computer vision techniques. The approach is to use a ML optical flow model (Raft) to identify the likely parts of the video that 
are likely to be a panninshot. Then double check it by taking the crop of the intersection between the two frames and checking if its the same picture using an 
image hashing algorithm. the rationale is that if those two frames are part of a panning shot than the intersectio that optical flow model says they both should share should be the 
same picture. This is done for the frames of the video and the frames that are likely panning shots are put into groups and a stitch is attempted using opencvs stitching class. 
<br></br>
A visual representation of the concept is shown in the animation below 
<br></br>
</p>
<img src="docs/animated.gif" alt="Your Image">


<p>
    To use this script you first need to download the optical flow models from <a href="https://github.com/PINTO0309/PINTO_model_zoo/tree/main/252_RAFT"> PINTO model zoo</a> and place it in the models folder. The script supports running onnx files or tensortRT engines if you convert those files. 

</p>

Explantion of script parameters <br></br>
<b> --videodir </b> The filepath of the video file to analyze 
</br>
<b> --outdir </b> The directory to save the image stitches and log files
</br>
<b> --starting_timestamp </b> The time to start analysing the video 
</br>
<b> --ending_timestamp </b>The time to stop analysing the video 
</br>
<b> --ratio </b> The ratio of the output .5 resizes the image to half their dimensions before stitching for example 
</br>
<b> --min_range </b> the minumum range of detected optical movement in order to attempt stitching
</br>
<b> --use_trt </b> Flag to use tensorRT
</br>
<b> --frame_skip </b> how many frames to skip before running optical flow inference 
</br>
 <b> --model_path </b> the path of the optical flow model
</br>

a gui was also added for ease of use 
<br>

![image](https://github.com/bdiaz29/Moving-Shot-Extraction-/assets/16212103/312a8b86-7916-492b-960d-1cbf44d57fc8)

<h1> Installation  </h1>

git clone https://github.com/bdiaz29/Moving-Shot-Extraction-
<br>
pip install -r requirements.txt

<h1> Caveat  </h1>

The Stitching process isn't always perfect and so you should always do a manual inspection of the stiches before including them in a dataset. 
I'm still trying to figure out the best approach to dealing with frames that are panning where there is character movement. 



