import math
import cv2
import numpy as np


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)  ####
    

    def release(self):
        for handle in self.handles:
            handle.remove()


def get_cam_weights(grads):
    return np.mean(grads, axis=(2, 3), keepdims=True)


def get_loss(output, target_category):
    loss = 0
    for i in range(len(target_category)):
        loss = loss + output[i, target_category[i]]
    return loss

def get_cam_image(activations, grads):
    weights = get_cam_weights(grads)
    weighted_activations = weights * activations
    cam = weighted_activations.sum(axis=1)

    return cam,weights


def get_target_width_height(input_tensor):
    width, height = input_tensor.size(-1), input_tensor.size(-2)
    return width, height

def compute_cam_per_layer(activations_and_grads, input_tensor):
    activations_list = [a.cpu().data.numpy()
                        for a in activations_and_grads.activations]
    grads_list = [g.cpu().data.numpy()
                    for g in activations_and_grads.gradients]
    
    target_size = get_target_width_height(input_tensor)

    cam_per_target_layer = []
    weights_per_target_layer = []
    # Loop over the saliency image from every layer

    for layer_activations, layer_grads in zip(activations_list, grads_list):
        cam, weights = get_cam_image(layer_activations, layer_grads)
        cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
        scaled = scale_cam_image(cam, target_size)
        cam_per_target_layer.append(scaled[:, None, :])
        weights_per_target_layer.append(weights)

    return cam_per_target_layer,weights_per_target_layer

def aggregate_multi_layers(cam_per_target_layer):
    cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
    cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
    result = np.mean(cam_per_target_layer, axis=1)
    return scale_cam_image(result)

def scale_cam_image(cam, target_size=None):
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        if target_size is not None:
            img = cv2.resize(img, target_size)
        result.append(img)
    result = np.float32(result)

    return result


class ResizeTransform:
    def __init__(self, im_h: int, im_w: int):
        # self.height = self.feature_size(im_h)
        # self.width = self.feature_size(im_w)
        self.height = im_h
        self.width = im_w

    # @staticmethod
    # def feature_size(s):
    #     s = math.ceil(s / 4)  # PatchEmbed
    #     s = math.ceil(s / 2)  # PatchMerging1
    #     s = math.ceil(s / 2)  # PatchMerging2
    #     s = math.ceil(s / 2)  # PatchMerging3
    #     return s

    def __call__(self, x):
        height = int(np.sqrt(x.size(1)))
        width = height
        result = x.reshape(x.size(0),
                           height,
                           width,
                           x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)

        return result