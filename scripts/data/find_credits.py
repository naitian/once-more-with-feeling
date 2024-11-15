"""
Find the start timestamp of the credits.
"""

import argparse
from bisect import bisect_left
from pathlib import Path

import numpy as np
import torch
from cv2 import VideoCapture, CAP_PROP_FPS, CAP_PROP_POS_FRAMES, CAP_PROP_FRAME_COUNT
from torchvision.io import write_jpeg, write_video
import pytesseract
from tqdm import tqdm

from src.utils import DATA_DIR
from src.video.meta import VideoMetadata

test_folder = DATA_DIR / "test_shots"
test_folder.mkdir(exist_ok=True, parents=True)


def find_credits(
    meta: VideoMetadata,
    save_video=False,
    video_dir=None,
    debug=False,
    min_credit_length=30,
    skip_start_minutes=60,
):
    if save_video and video_dir is None:
        raise ValueError("video_dir must be provided if save_video is True.")
    if not meta.path.exists():
        raise ValueError(f"Video file {meta.path} does not exist.")
    reader = VideoCapture(str(meta.path))

    def _read_frame(frame):
        reader.set(CAP_PROP_POS_FRAMES, frame)
        ret, frame = reader.read()
        if not ret:
            return None
        return torch.from_numpy(frame)

    fps = reader.get(CAP_PROP_FPS)
    num_frames = int(reader.get(CAP_PROP_FRAME_COUNT))
    print(num_frames / fps / 60)
    # skip the first 60 minutes
    first_shot = bisect_left(
        meta.shots, (skip_start_minutes * 60 * fps), key=lambda x: x[0]
    )
    shots = []
    for i, shot in enumerate(tqdm(meta.shots[first_shot:])):
        # skip shot if less than 3 seconds
        if shot[1] - shot[0] < 3 * fps:
            continue
        # get first, middle, and last frame of each shot
        # get middle frame of each shot
        frame_nums = [shot[0] + 1, (shot[0] + shot[1]) // 2, shot[1] - 1]

        frames = [_read_frame(frame) for frame in frame_nums]
        text = pytesseract.image_to_string(frames[1].numpy())
        if debug:
            write_jpeg(
                frames[1].permute(2, 0, 1)[[2, 1, 0], :, :],
                str(test_folder / f"{meta.path.stem}_{i}.jpg"),
            )
            print(
                f"===at: {shot[0] / fps / 60 / 60}|duration: {(shot[1] - shot[0]) / fps / 60 / 60} minutes==="
            )
            print(text)

        if len(text) < 50:
            continue

        shots.append(shot)

    # group shots within 5 seconds of each other
    starts, ends = list(zip(*shots))
    starts = np.array(starts)
    ends = np.array(ends)

    groups = ((starts[1:] - ends[:-1]) > 5 * fps).cumsum()
    groups = np.concatenate([[0], groups])

    durations = np.bincount(groups, weights=ends - starts)
    # first group of shots longer than 30 seconds
    group = np.argmax(durations > min_credit_length * fps)
    first_shot = shots[np.argmax(groups == group)]
    last_shot = shots[np.where(groups == group)[0][-1]]

    if save_video:
        frames = []
        reader.set(CAP_PROP_POS_FRAMES, first_shot[0])
        while reader.get(CAP_PROP_POS_FRAMES) < last_shot[1]:
            ret, frame = reader.read()
            if not ret:
                break
            frames.append(torch.from_numpy(frame))
        frames = torch.stack(frames)
        write_video(
            str(video_dir / f"{meta.path.stem}_credits.mp4"),
            frames,
            fps=reader.get(CAP_PROP_FPS),
        )

        write_jpeg(
            _read_frame(first_shot[0]).permute(2, 0, 1)[[2, 1, 0], :, :],
            str(test_folder / f"{meta.path.stem}_final.jpg"),
        )
    return first_shot[0], fps


def main(args):
    meta = VideoMetadata.from_video(args.video_path)
    args.video_dir.mkdir(exist_ok=True, parents=True)
    frame, fps = find_credits(
        meta,
        save_video=args.save_video,
        video_dir=args.video_dir,
        debug=args.debug,
        min_credit_length=args.min_credit_length,
        skip_start_minutes=args.skip_start_minutes,
    )
    args.output_dir.mkdir(exist_ok=True, parents=True)
    with open(args.output_dir / f"{meta.imdb}_credits.txt", "w") as f:
        f.write(f"{meta.imdb}\t{frame / fps}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find the start timestamp of the credits."
    )
    parser.add_argument("video_path", type=Path, help="Path to the video file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "credits",
        help="Directory to save the credits timestamp.",
    )
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=DATA_DIR / "credit_videos",
        help="Directory to save the credits video.",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save the credits video and a frame from the start of the credits",
        default=False,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save a frame from each text-heavy shot to debug the OCR",
        default=False,
    )
    parser.add_argument(
        "--min_credit_length",
        type=float,
        default=30,
        help="Minimum length of the credits in seconds.",
    )
    parser.add_argument(
        "--skip_start_minutes",
        type=float,
        default=60,
        help="Skip the first n minutes of the video.",
    )
    main(parser.parse_args())
