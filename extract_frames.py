import csv
import glob
import os
import os.path
import subprocess as sp
from subprocess import call
#from subprocess import Popen
from joblib import Parallel,delayed

data_file = []
def subprocess_cmd(command):
    process = sp.Popen(command,stdout=sp.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    print (proc_stdout)
def subprocess_debug(ffmpeg_command):
    p = sp.Popen(ffmpeg_command, stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = p.communicate()
#    return stdout, stderr
def core_func(video_path):
    global data_file
    video_parts = get_video_parts(video_path)
    filename_no_ext, filename = video_parts

    # Only extract if we haven't done it yet. Otherwise, just get
    # the info.
    if not check_already_extracted(video_parts):
        # Now extract it.
        src = os.path.join(video_dir,filename)
        dest = os.path.join(output_dir,filename_no_ext + '-%04d.jpg')
        #print('in', src, dest)
#        call(["/usr/local/bin/ffmpeg", "-i", src,"-r", "4", dest])
        call(["bash", "./get_frames.sh", src, dest, "30"])
    # Now get how many frames it is.
    nb_frames = get_nb_frames_for_video(video_parts)
    #print('written: ',nb_frames)
    data_file.append([filename_no_ext, nb_frames])



def get_nb_frames_for_video(video_parts):
    """Given video parts of an (assumed) already extracted video, return
    the number of frames that were extracted."""
    filename_no_ext, _ = video_parts
    generated_files = glob.glob(os.path.join(output_dir, filename_no_ext + '*.jpg'))
    return len(generated_files)

def get_video_parts(video_path):
    """Given a full path to a video, return its parts."""
    parts = video_path.split(os.path.sep)
    #print(parts)
    filename = parts[-1]
    filename_no_ext = filename.split('.')[0]
    return filename_no_ext, filename#('video6514', 'video6514.mp4')

def check_already_extracted(video_parts):
    """Check to see if we created the -0001 frame of this file."""
    filename_no_ext, _ = video_parts
    return bool(os.path.exists(os.path.join(output_dir,
                               filename_no_ext + '-0030.jpg')))
#%%
#os.chdir('/Users/DennisLin/Desktop/')
ROOT = os.getcwd()+'/'
video_dir = "./videos"
output_dir = "./frames_2"
features_out = './features'
jobs = 1

vfiles = sorted(glob.glob(os.path.join(video_dir, '*.mp4')))
results = Parallel(n_jobs=jobs)(delayed(core_func)(video_path) for video_path in vfiles)               
#results = ((core_func(video_path)) for video_path in vfiles)
with open('data_file_2.csv', 'w') as fout:
    writer = csv.writer(fout)
    writer.writerows(data_file)

print("Extracted and wrote %d video files." % (len(data_file)))
# data_file above did not work in parallel processing need to write a code to count them manually:
#from tqdm import tqdm
#pbar = tqdm(total=len(vfiles))
#
#data_file = []
#for video_path in vfiles:
#    video_parts = get_video_parts(video_path)
#    filename_no_ext, filename = video_parts
#    generated_files = glob.glob(os.path.join(output_dir, filename_no_ext + '*.jpg'))
#    data_file.append([filename_no_ext, len(generated_files)])
#    pbar.update(1)
#pbar.close()
#with open('data_file.csv', 'w') as fout:
#    writer = csv.writer(fout)
#    writer.writerows(data_file)
#
#print("Extracted and wrote %d video files." % (len(data_file)))
'''
Note: Need to specify the ffmpeg path when running.
'''
