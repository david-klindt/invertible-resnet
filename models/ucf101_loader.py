import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


MEAN = (87.7006, 94.8281, 98.9483)
STD = (70.7814, 70.0517, 71.8607)

class UCF101(Dataset):
    """Dataset Class for Loading Video"""

    def __init__(
        self,
        root_dir,
        channels=3,
        time_depth=16,
        step_size=1,
        x_size=32,
        y_size=32,
        mean=None,
        transform=None,
        center_crop=True,
    ):
        """
        Args:
            clips_list_file (string): Path to the clipsList file with labels.
            root_dir (string): Directory with all the videoes.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            channels: Number of channels of frames
            time_depth: Number of frames to be loaded in a sample
            x_size, y_size: Dimensions of the frames
            mean: Mean value of the training set videos over each channel
        """

        self.clips_list = [f for f in os.listdir(root_dir) if f.endswith('.avi')]
        self.root_dir = root_dir
        self.channels = channels
        self.time_depth = time_depth
        self.x_size = x_size
        self.y_size = y_size
        self.mean = mean
        self.transform = transform
        self.step_size = step_size
        self.center_crop = center_crop

    def __len__(self):
        return len(self.clips_list)

    def read_video(self, video_file):
        # Open the video file
        cap = cv2.VideoCapture(video_file)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frameCount < self.time_depth * self.step_size:
            return None, True
        frames = torch.FloatTensor(
            self.time_depth, self.channels, self.x_size, self.y_size)
        # pick random frame and following
        maximum = frameCount - self.time_depth * self.step_size
        rand_frame = np.random.uniform(0, maximum)
        cap.set(1, rand_frame)
        ind = 0
        f = 0
        while True:
            ret, frame = cap.read()
            if not(f % self.step_size):
                if ret:
                    if self.center_crop:
                        frame = frame[40:-40, 80:-80]  # center crop
                    frame = cv2.resize(frame, (self.x_size, self.y_size))
                    frame = torch.from_numpy(frame)
                    # HWC2CHW
                    frame = frame.permute(2, 0, 1)
                    frames[ind] = frame
                else:
                    return None, True
                ind += 1
                if ind == self.time_depth:
                    break
            f += 1
        cap.release()

        for c in range(self.channels):
            frames[:, c] -= MEAN[c]
            frames[:, c] /= STD[c]

        return frames, False # clip, flag whether it failed

    def __getitem__(self, idx):

        while True:
            video_file = os.path.join(self.root_dir, self.clips_list[idx])
            clip, failed_clip = self.read_video(video_file)
            if not failed_clip:
                break
            else:
                other_idx = np.random.randint(self.__len__())
                return self.__getitem__(other_idx)
        if self.transform:
            clip = self.transform(clip)
        return clip
