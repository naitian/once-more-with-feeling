import random
import sys
from collections import Counter

from src.utils import DATA_DIR
from src.video.meta import VideoMetadata

random.seed(1)

minConf = 0.18

names = {}
active_frames = {}


def read_active_speaker(filename):
    with open(filename) as file:
        for line in file:
            cols = line.rstrip().split("\t")
            frame = int(cols[1])
            faceno = int(cols[2])
            score = float(cols[3])
            bbox = cols[4:]
            if frame not in active_frames:
                active_frames[frame] = []
            if score > 0:
                active_frames[frame].append((faceno, score, bbox))


def read_names(filename):
    with open(filename) as file:
        for line in file:
            cols = line.rstrip().split("\t")
            aid = cols[0]
            name = cols[1]
            names[aid] = name


def read_recog(filename):
    recog = {}
    with open(filename) as file:
        for line in file:
            cols = line.rstrip().split("\t")
            trackno = cols[0]

            actor, score = cols[3].split(" ")[0].split(":")
            recog[trackno] = actor, float(score)

    return recog


def read_tracks(filename):
    tracks = {}
    faceno2track = {}
    with open(filename) as file:
        for line in file:
            cols = line.rstrip().split("\t")
            trackno = cols[0]
            fno = int(cols[1])
            faceno = int(cols[2])
            if trackno not in tracks:
                tracks[trackno] = []
            tracks[trackno].append(fno)
            faceno2track[fno, faceno] = trackno
    return tracks, faceno2track


def get_actors(actors_onscreen, start, end, fps, min_duration_secs=0.3, window=4):
    start -= window
    end += window

    if start < 0:
        start = 0

    start_frame = int(start * fps)
    end_frame = int(end * fps)

    counts = Counter()
    for fno in range(start_frame, end_frame):
        if fno in actors_onscreen:
            for actor in actors_onscreen[fno]:
                counts[actor] += 1 / fps

    vals = []
    for k, v in counts.most_common():
        name = k
        if k in names:
            name = names[k]
        if v > min_duration_secs:
            vals.append((name, "%.3f" % v))

    return counts


def read_fps(filename):
    with open(filename) as file:
        cols = file.readline().split("\t")

        frames = int(cols[1])
        fps = float(cols[4])
        return frames, fps


def read_film_output(path, idd):
    fpsFile = "%s/movies_fps/%s.fps.txt" % (path, idd)
    tracksFile = "%s/tracks/%s.tracks.txt" % (path, idd)
    recogFile = "%s/recog/%s.recog.txt" % (path, idd)

    frames, fps = read_fps(fpsFile)
    tracks, faceno2track = read_tracks(tracksFile)
    recog = read_recog(recogFile)

    actors_onscreen = {}

    for trackno in tracks:
        actor_id, conf = recog[trackno]
        if conf > minConf:
            for frame in tracks[trackno]:
                if frame not in actors_onscreen:
                    actors_onscreen[frame] = {}
                actors_onscreen[frame][actor_id] = 1
    return actors_onscreen, fps, faceno2track, recog


def get_active(start, end, fps, faceno2track, recog):
    if start < 0:
        start = 0

    start_frame = int(start * fps)
    end_frame = int(end * fps)

    counts = Counter()
    # print(start_frame, end_frame, "FRAMES")
    for fno in range(start_frame, end_frame):
        if fno in active_frames:
            for faceno, score, bbox in active_frames[fno]:
                if (fno, faceno) in faceno2track:
                    trackno = faceno2track[fno, faceno]

                    actor, score = recog[trackno]
                    if score >= minConf:
                        counts[actor] += 1

    return counts.most_common()


def proc(filename, folder, idd, outname):
    actors_onscreen, fps, faceno2track, recog = read_film_output(folder, idd)

    outfile = open(outname, "w")
    with open(filename) as file:
        file.readline()
        for line in file:
            cols = line.rstrip().split("\t")
            if len(cols) < 4:
                continue
            start = float(cols[0])
            end = float(cols[1])
            active_speakers = get_active(start, end, fps, faceno2track, recog)

            text = cols[3]

            y_name = None

            if len(active_speakers) > 0:
                y_name = active_speakers[0][0]
                if y_name in names:
                    y_name = names[y_name]
                y_name = "%s" % y_name

            # print(f"%s\t%s\t%s\t\033[{color_code}m%s\033[30m\tVALS" % (start, end, text, y_name ))
            outfile.write("%s\t%s\t%s\t%s\n" % (start, end, text, y_name))


# Usage:
#
# python <imdb> name.basics.tsv

# name.basics is from https://datasets.imdbws.com/title.basics.tsv.gz

# assumes existtence of *.fps.txt, *.recog.txt and *.tracks.txt (see read_film_output above) and *.asd.txt
# change location of those folders as necessary

imdb = sys.argv[1]
nameFile = sys.argv[2]
read_names(nameFile)

folder = "/global/secure0/groups/isch-bamman-fa/computed_data/movies/"
filename = DATA_DIR / "asr_extracts" / "split_data" / f"{imdb}.tsv"
outname = DATA_DIR / "asr_extracts" / "asd_data" / f"{imdb}.tsv"
video = VideoMetadata.from_imdb(imdb)
asdFile = "%s/asd/%s.asd.txt" % (folder, video.path.stem)
read_active_speaker(asdFile)
proc(filename, folder, video.path.stem, outname)
