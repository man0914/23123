import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from subprocess import call
import youtube_dl


def download_youtube_video(youtube_link,start_time,end_time):
    if os.path.exists('output/vod.mp4'):
        os.remove('output/vod.mp4')
    ydl_opts = {'outtmpl': 'output/vod_full.%(ext)s', 'format': '137'}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_link])
    # TODO test this on other systems.
    print("Calling ffmpeg to cut up video")
    call(['ffmpeg', '-i', 'output/vod_full.mp4', '-ss', start_time, '-to', end_time, '-c', 'copy', 'output/vod.mp4'],shell=True)
    os.remove('output/vod_full.mp4')

def process_mp4(test_mp4_vod_path):
    video = cv2.VideoCapture(test_mp4_vod_path)
    print("Opened ", test_mp4_vod_path)
    print("Processing MP4 frame by frame")

    # forward over to the frames you want to start reading from.
    # manually set this, fps * time in seconds you wanna start from
    video.set(1, 0);
    success, frame = video.read()
    count = 0
    file_count = 0
    success = True
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Loading video %d seconds long with FPS %d and total frame count %d " % (total_frame_count/fps, fps, total_frame_count))

    while success:
        success, frame = video.read()
        if not success:
            break
        if count % 1000 == 0:
            print("Currently at frame ", count)

        # i save once every fps, which comes out to 1 frames per second.
        # i think anymore than 2 FPS leads to to much repeat data.
        if count %  fps == 0:
            im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(im).crop((1625, 785, 1920, 1080))
            im.save(os.path.join('output/', str(file_count) + '.jpg'), quality=90)
            file_count += 1
        count += 1

def get_ward_bounding_boxes(frame_obj,teamplate_obj):
    img = cv2.imread(frame_obj,0)
    img2 = img.copy()
    template = cv2.imread(teamplate_obj,0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img,template,cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    return np.array(['blue ward', top_left[0], top_left[1], top_left[0]+ w, top_left[1] + h])    

def check_boxes_for_ward_in_dict(frame_obj,num_of_ward):    
    boxes_in_frame = []
    for i in range(1, num_of_ward+1):
        template_index = str(i)
        box = get_ward_bounding_boxes(frame_obj,'input/'+template_index+'.png')
        boxes_in_frame.append(box)
    print(boxes_in_frame)
        
def get_bounding_boxes_and_images():
    #sources = np.load('blueWardData.npz')
    all_images = []
    all_boxes = []
    im = Image.open('input/test.jpg')
    im = np.array(im, dtype = np.uint8)
    all_images.append(im)
    boxes = check_boxes_for_ward_in_dict('input/test.jpg',1)
    all_boxes.append(np.array(boxes))
    np.savez('blueWardData.npz', images=all_images,boxes=all_boxes)
    

def _main():
    download_youtube_video('https://www.youtube.com/watch?v=dGzJUTzdecM','0:22:52','0:48:23')

if __name__ == '__main__':
    _main()
