import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import PIL.Image, PIL.ImageOps, numpy, io


def crop_and_add_axis(image_path, pad=0.15, w_axis=False):
    image = PIL.Image.open(image_path).convert('L').convert('1')

    # crop image
    bbox = image.getbbox()
    if bbox is not None:
        image_cropped = image.crop(bbox)
        w, h = image_cropped.size
        if w / h >= 1 / 1.05 and w / h <= 1.05 / 1:
            pad_w = int(w * pad)
            pad_h = int(h * pad)
        else:
            tot = max(w, h) * (1 + 2 * pad)
            pad_w = int(round((tot - w) / 2))
            pad_h = int(round((tot - h) / 2))
        # pad image
        image = PIL.Image.new('RGB', (w + 2 * pad_w, h + 2 * pad_h), (255, 255, 255))
        image.paste(image_cropped.convert('RGB'), (pad_w, pad_h))

    fig, ax = plt.subplots(figsize=(10.24, 10.24), dpi=100)
    ax.imshow(PIL.ImageOps.flip(image), extent=(0, 1, 0, 1), aspect='auto')
    ax.set_position([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(1, 0))

    # axis and ticks
    if w_axis:
        # position
        ax.set_xticks(numpy.arange(0.1, 1.0, 0.1))
        ax.set_yticks(numpy.arange(0.1, 1.0, 0.1))
        ax.xaxis.set_tick_params(pad=-19)
        ax.yaxis.set_tick_params(pad=-33)
        ax.tick_params(axis='both', direction='in', labelsize=20)
        offset = mtransforms.ScaledTranslation(0.25, 0, fig.dpi_scale_trans)
        for label in ax.get_xticklabels():
            label.set_transform(label.get_transform() + offset)
        offset = mtransforms.ScaledTranslation(0, 0.17, fig.dpi_scale_trans)
        for label in ax.get_yticklabels():
            label.set_transform(label.get_transform() + offset)
        # color
        ax.tick_params(axis='both', colors='#005CFF', labelcolor='#005CFF')
        for spine in ax.spines.values():
            spine.set_color('#005CFF')
        ax.grid(True, linewidth=3, alpha=0.7, color='#005CFF')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpg')
    plt.close()
    buffer.seek(0)

    return PIL.Image.open(buffer)
