import os
import numpy as np


class DVSSource:
    def __init__(self, path: str,class_num:int):
        self.make_dataset(path,class_num)

    def make_dataset(self, path: str,class_num:int):
        self.data = []
        label = []
        class_folders = [d for d in os.listdir(
            path) if os.path.isdir(os.path.join(path, d))]
        self.class_to_idx = {class_folder: label for label,
                             class_folder in enumerate(class_folders)}

        for i, class_folder in enumerate(class_folders):
            class_folder_path = os.path.join(path, class_folder)
            npz_files = [f for f in os.listdir(
                class_folder_path) if f.endswith('.npz')]

            for npz_file in npz_files:
                file_path = os.path.join(class_folder_path, npz_file)
                frames = self.load_npz_frames(file_path)
                self.data.append(frames)
                label.append(i)

            print(f'{class_folder} loaded')

        self.label = np.eye(class_num)[label]

    @staticmethod
    def load_npz_frames(file_name: str) -> np.ndarray:
        '''
        :param file_name: path of the npz file that saves the frames
        :type file_name: str
        :return: frames
        :rtype: np.ndarray
        '''
        return np.load(file_name, allow_pickle=True)['frames'].astype(np.float32)

    def __getitem__(self, index):
        return self.data[index], self.label[index][:]

    def __len__(self):
        return len(self.data)
