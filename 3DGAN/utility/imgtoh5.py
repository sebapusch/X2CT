import h5py
import numpy as np
from PIL import Image
import os
import sys

def img_to_h5(img_paths: list, out_dir: str):
    assert len(img_paths) > 0, 'at least a single image necessary'

    images = []

    for img_path in images:
        image = Image.open(img_path).convert('L')
        image = image.resize((256, 256))
        images.append(np.array(image).astype('float32') / 255)

    with h5py.File(os.path.join(out_dir, 'ct_xray_data.h5'), 'w') as f:
        for i, img_data in enumerate(images):
            f.create_dataset(f'xray{i}', data=img_data)

        # If needed, add dummy CT data too
        # ct_dummy = np.random.rand(64, 256, 256).astype('float32')
        # f.create_dataset('ct', data=ct_dummy)


if __name__ == '__main__':
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    in_path = os.path.join(base_path, sys.argv[1])
    out_path = os.path.join(base_path, sys.argv[2])

    print('input path: ', in_path)
    print('output path: ', out_path)

    entries = {}

    for entry in os.listdir(in_path):
        filename, ext = os.path.splitext(entry)

        if ext not in ['.png', '.jpg', '.jpeg']:
            continue

        id_ = filename.split('_', 1)[0]

        if id_ in entries:
            entries[id_].append(os.path.join(in_path, entry))
        else:
            entries[id_] = [os.path.join(in_path, entry)]

    print('found', len(entries), 'unique entries')

    for id_ in entries:
        datapoint_path = os.path.join(out_path, f'datapoint.{id_}')
        os.path.exists(datapoint_path) or os.mkdir(datapoint_path)
        img_to_h5(entries[id_], datapoint_path)
        print(f"HDF5 file created for entry with id: {id_} at location {datapoint_path}")
