1. Get source video place it into `data/original_src_vid/<your_vid>`
2. Run `ffmpeg -i data/original_src_vid/<your_vid> data/original_src_vid/<your_vid_frames>/frame_%04d.png` to get the raw frames
3. Run `python images2landmarkcsv.py -p shape_predictor_68_face_landmarks.dat -d data/original_src_vid/<your_vid_frames> -o data/original_src_vid/<your_vid.csv>`
4. Run `python save_tight_cropped_images.py -p shape_predictor_68_face_landmarks.dat -d data/original_src_vid/<your_vid_frames> -o data/original_src_vid/<your_vid.csv>` to get the face landmarks and bounding boxes dumped as csv
   1. Dont forget to create teh dir `data/original_src_vid/<your_vid>`
   2. This will also save the face cropped in the dir `data/original_src_vid/<your_vid>` same dir name as the csv
5. Send the cropped frames as zipped dir to kamil and ask him to do the magic
6. He sent you a video! put it into `data/crops/<your_vid_dir>`
7. Extract the frames from this video using ffmpeg see point 2
8. Extract the audio using ffmpeg `ffmpeg -i <your_vid_from_kamil>.mp4 -q:a 0 -map a <your_vid_from_kamil>.mp3`
9. Run `python transform_src_and_paste.py`
   1. Don't forget to change the paths inside - `source_landmarks` - Path to the csv created in step 4.
   2. `dest_landmarks` path to the csv with landmarks of the face in the original video create in step 3.
   3. `output_path` a valid directory where blended frames will be put
10. run `[dir_to_mp4.sh](dir_to_mp4.sh)` to rename and dump a video from these blended frames create in step 9
    1. Sometimes the `ffmpeg` command fails at the bottom of the bash then manually run the ffmpeg command
    2. The ffmpeg manual command `ffmpeg -framerate 25 -i <output_path_in_step_9>/frame_%04d.png -i "<your_vid_from_kamil>.mp3" -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p <FinalVideo_path_and_name>.mp4`