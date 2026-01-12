import argparse, torch, json, os, re
from transformers import Qwen3VLProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import matplotlib.pyplot as plt

from src.utils import crop_and_add_axis
from src.prompt import INSTRUCTION_SYSTEM, INSTRUCTION_USER


def main(args):
    model = LLM(model=args.model_path, max_model_len=128000, max_num_seqs=1, gpu_memory_utilization=0.9)
    processor = Qwen3VLProcessor.from_pretrained(args.model_path)

    paths, paths_out = list(), list()
    if os.path.isdir(args.path_image):
        for root, _, files in os.walk(args.path_image):
            for file in files:
                if file.lower().split('.')[-1] in ['jpg', 'jpeg', 'png']:
                    paths.append(os.path.join(root, file))
                    paths_out.append(os.path.join(args.path_output_image, os.path.relpath(root, args.path_image), file))
    else:
        paths.append(args.path_image)
        paths_out.append(os.path.join(args.path_output_image, args.path_image.split('/')[-1]))

    if args.visualize:
        os.makedirs(args.path_output_image, exist_ok=True)
    if os.path.dirname(args.path_output) != '':
        os.makedirs(os.path.dirname(args.path_output), exist_ok=True)

    data_out = list()
    for path, path_out in zip(paths, paths_out):
        # make input
        image = crop_and_add_axis(path, w_axis=True)
        image_inputs, _ = process_vision_info([{'content': [{'type': 'image', 'image': image}]}])
        conversation = [
            {'role': 'system', 'content': [{'type': 'text', 'text': INSTRUCTION_SYSTEM}]},
            {'role': 'user', 'content': [{'type': 'image', 'image': image_inputs[0]}, {'type': 'text', 'text': INSTRUCTION_USER}]},
        ]
        # inference
        with torch.no_grad():
            response = model.generate(
                prompts=[{
                    'prompt': processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True),
                    'multi_modal_data': {'image': image_inputs[0]}
                }],
                sampling_params=SamplingParams(top_p=1.0, top_k=1, temperature=0, max_tokens=3000, stop_token_ids=None)
            )
            output = response[0].outputs[0].text
        # parse output
        try:
            strokes = [json.loads(stroke) for stroke in re.findall(r'<stroke>([^<>]*)</stroke>', output, flags=re.DOTALL)]
            assert len(strokes) <= 50
            for stroke in strokes:
                assert isinstance(stroke, list)
                assert len(stroke) == 2
                for point in stroke:
                    assert isinstance(point, list) and len(point) == 2
                    assert isinstance(point[0], float) and isinstance(point[1], float)
                    assert 0 <= point[0] <= 1 and 0 <= point[1] <= 1
        except:
            strokes = list()

        data_out.append({'path_in': path, 'model_out': output})

        # visualize
        if args.visualize:
            plt.figure(figsize=(10.24, 10.24), dpi=100)
            plt.axis([0, 1, 1, 0])
            plt.axis('off')
            for stroke in strokes:
                plt.plot([stroke[0][0], stroke[1][0]], [stroke[0][1], stroke[1][1]], c='blue', linewidth=8, zorder=2)

            os.makedirs(os.path.dirname(path_out), exist_ok=True)
            plt.savefig(path_out, bbox_inches='tight', pad_inches=0)
            plt.close()

            data_out[-1]['path_out'] = path

    with open(args.path_output, 'w') as fp:
        json.dump(data_out, fp, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str)
    parser.add_argument('--path_image', type=str)
    parser.add_argument('--path_output', type=str)
    parser.add_argument('--path_output_image', type=str)
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()

    main(args)
