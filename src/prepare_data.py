import pyarrow, pyarrow.parquet
import cv2, argparse, base64, io, struct, numpy, pandas, multiprocessing, os, tqdm

from src.utils import crop_and_add_axis
from src.prompt import INSTRUCTION_SYSTEM, INSTRUCTION_USER


def process_image(image_path):
    image_w_axis = crop_and_add_axis(image_path, w_axis=True)
    image_wo_axis = crop_and_add_axis(image_path, w_axis=False)
    # image (axis)
    byte_stream = io.BytesIO()
    image_w_axis.save(byte_stream, format='PNG')
    base64_image = byte_stream.getvalue()
    # image (no axis, binary)
    image_gray = cv2.cvtColor(numpy.array(image_wo_axis), cv2.COLOR_RGB2GRAY)
    _, image_bin = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image_bin = (image_bin == 0).astype(numpy.uint8)
    bits = (image_bin > 0).astype(numpy.uint8).ravel()
    h, w = image_bin.shape
    data = struct.pack('>II', h, w) + numpy.packbits(bits, bitorder='little').tobytes()
    base64_images_no_axis_binary = base64.b64encode(data).decode('ascii')

    return base64_image, base64_images_no_axis_binary


def main(args):
    paths = list()
    for root, _, files in os.walk(args.path_image):
        for file in files:
            if file.lower().split('.')[-1] in ['jpg', 'jpeg', 'png']:
                paths.append(os.path.join(root, file))

    print('Processing images.')
    with multiprocessing.Pool(processes=16) as pool:
        images = list(tqdm.tqdm(pool.imap_unordered(process_image, paths), total=len(paths)))

    print('Saving parquets.')
    data = list()
    for idx, (base64_image, base64_images_no_axis_binary) in enumerate(images):
        data.append({
            'data_source': 'character-reconstruction',
            'prompt': [
                {'role': 'system', 'content': INSTRUCTION_SYSTEM},
                {'role': 'user', 'content': '<image>' + ' ' + INSTRUCTION_USER},
            ],
            'images': [{'bytes': base64_image, 'path': None}],
            'ability': 'character-reconstruction',
            'reward_model': {'style': 'rule', 'ground_truth': ''},
            'extra_info': {'index': idx, 'base64': base64_images_no_axis_binary},
        })

    if os.path.dirname(args.path_train_data) != '':
        os.makedirs(os.path.dirname(args.path_train_data), exist_ok=True)
    pyarrow.parquet.write_table(
        pyarrow.Table.from_pandas(pandas.DataFrame(data)),
        args.path_train_data,
        row_group_size=1024,
    )

    if os.path.dirname(args.path_val_data) != '':
        os.makedirs(os.path.dirname(args.path_val_data), exist_ok=True)
    pyarrow.parquet.write_table(
        pyarrow.Table.from_pandas(pandas.DataFrame(data[:1])),
        args.path_val_data,
        row_group_size=1024,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_image', type=str)
    parser.add_argument('--path_train_data', type=str)
    parser.add_argument('--path_val_data', type=str)

    args = parser.parse_args()

    main(args)
