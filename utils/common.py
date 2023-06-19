import cv2
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


classes = {}
classes[2] = {} # add face
classes[7] = {2:8, 3:9, 4:13, 5:14, 6:15} # add hair and ears
classes[16] = {} # add facial features
classes[19] = {} # add neck and cloth

# Log images
def _log_input_image(x, nc=3):
    if nc==3:
        return tensor2im(x)
    elif nc==1:
        return tensor2sketch(x)
    elif nc==68:
        return tensor2landmark(x)
    elif nc=='featmap':
        return tensor2featmap(x)
    elif nc==2:
        return tensor2imagenetmap(x)
    else:
        return tensor2map(x, nc)

def log_input_image(x, nc=3, size=256):
    _x = _log_input_image(x, nc)
    if not _x.size[0] == size or not _x.size[1] == size:
        _x = _x.resize((size,size)) 
    return _x


def tensor2im(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))

def tensor2featmap(var):
    feature_image = var.cpu().detach().numpy()
    feature_image -= feature_image.mean()
    feature_image /= feature_image.std ()
    feature_image *=  64
    feature_image += 128
    feature_image = np.clip(feature_image, 0, 255).astype('uint8')
    return Image.fromarray(feature_image)

def tensor2map(var, nc):
    mask = np.argmax(var.data.cpu().numpy(), axis=0)
    if nc == 7:
        for key in classes[nc].keys():
            mask[mask == key] = classes[nc][key]
    colors = get_colors()
    mask_image = np.ones(shape=(mask.shape[0], mask.shape[1], 3))
    for class_idx in np.unique(mask):
        mask_image[mask == class_idx] = colors[class_idx]
    mask_image = mask_image.astype('uint8')
    return Image.fromarray(mask_image)

def tensor2imagenetmap(var):
    mask = np.argmax(var.data.cpu().numpy(), axis=0)
    colors = [[0, 0, 192], [192, 0, 0]]
    mask_image = np.ones(shape=(mask.shape[0], mask.shape[1], 3))
    for class_idx in np.unique(mask):
        mask_image[mask == class_idx] = colors[class_idx]
    mask_image = mask_image.astype('uint8')
    return Image.fromarray(mask_image)


def tensor2sketch(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = cv2.cvtColor(var, cv2.COLOR_GRAY2BGR)
    var = var * 255
    return Image.fromarray(var.astype('uint8'))

def tensor2landmark(heatmap):
    heatmap = heatmap.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    heatmap = heatmap/heatmap.max(axis=(0,1), keepdims=True)
    heatmap = heatmap.sum(2)
    heatmap = heatmap/heatmap.max()*255.0
    heatmap = heatmap.astype('uint8')
    return Image.fromarray(heatmap)

def float2map(mask, nc):
    if nc == 7:
        for key in classes[nc].keys():
            mask[mask == key] = classes[nc][key]
    colors = get_colors()
    mask_image = np.ones(shape=(mask.shape[0], mask.shape[1], 3))
    for class_idx in np.unique(mask):
        mask_image[mask == class_idx] = colors[class_idx]
    mask_image = mask_image.astype('uint8')
    return Image.fromarray(mask_image)

# Visualization utils
def get_colors():
    # currently support up to 19 classes (for the celebs-hq-mask dataset)
    colors = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],
              [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204],
              [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
    return colors


def vis_faces(log_hooks, dataset_type=None):
    display_count = len(log_hooks)
    fig = plt.figure(figsize=(8, 4 * display_count))
    gs = fig.add_gridspec(display_count, 3)        
    for i in range(display_count):
        hooks_dict = log_hooks[i]
        fig.add_subplot(gs[i, 0])
        if 'diff_input' in hooks_dict:
            vis_faces_with_id(hooks_dict, fig, gs, i)
        else: 
            vis_faces_with_rgb(hooks_dict, fig, gs, i)
    plt.tight_layout()
    return fig


def vis_faces_with_id(hooks_dict, fig, gs, i):
    plt.imshow(hooks_dict['input_face'])
    plt.title('Input\nOut Sim={:.2f}'.format(float(hooks_dict['diff_input'])))
    fig.add_subplot(gs[i, 1])
    plt.imshow(hooks_dict['target_face'])
    plt.title('Target\nIn={:.2f}, Out={:.2f}'.format(float(hooks_dict['diff_views']),
                                                     float(hooks_dict['diff_target'])))
    fig.add_subplot(gs[i, 2])
    plt.imshow(hooks_dict['output_face'])
    plt.title('Output\n Target Sim={:.2f}'.format(float(hooks_dict['diff_target'])))


def vis_faces_no_id(hooks_dict, fig, gs, i):
    plt.imshow(hooks_dict['input_face'], cmap="gray")
    plt.title('Input')
    fig.add_subplot(gs[i, 1])
    plt.imshow(hooks_dict['target_face'])
    plt.title('Target')
    fig.add_subplot(gs[i, 2])
    plt.imshow(hooks_dict['output_face'])
    plt.title('Output')
    
def vis_faces_with_seg(hooks_dict, fig, gs, i):
    plt.imshow(hooks_dict['input'], cmap="gray")
    plt.title('Input')
    fig.add_subplot(gs[i, 1])
    plt.imshow(hooks_dict['output_face'])
    plt.title('OutputFace')
    fig.add_subplot(gs[i, 2])
    plt.imshow(hooks_dict['reconstruct'], cmap="gray")
    plt.title('Recon')
    
def vis_faces_with_rgb(hooks_dict, fig, gs, i):
    plt.imshow(hooks_dict['input'])
    plt.title('Input')
    fig.add_subplot(gs[i, 1])
    plt.imshow(hooks_dict['output_face'])
    plt.title('OutputFace')
    fig.add_subplot(gs[i, 2])
    plt.imshow(hooks_dict['reconstruct'])
    plt.title('Recon')


