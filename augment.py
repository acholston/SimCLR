import random
from torchvision import transforms as T
import torch
import scipy.ndimage
import numpy as np

class GaussianBlur(object):
    def __init__(self, sigma_range, ratio, prob):
        self.sigma_range = sigma_range
        self.ratio = ratio
        self.prob = prob

    def get_kernel(self, kernel):
        self.gf = torch.nn.Conv2d(3, 3, kernel_size=kernel*2+1, stride=1, padding=kernel, bias=False, groups=1)

    def __call__(self, img):
        if random.random() < self.prob:
            sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
            img = img.unsqueeze(0)
            size = img.shape
            kernel = int(max(size[1], size[2])*self.ratio)
            self.get_kernel(kernel)

            filter = np.zeros([kernel*2+1, kernel*2+1])
            filter[kernel, kernel] = 1.0
            kernel_info = scipy.ndimage.gaussian_filter(filter, sigma=sigma)
            for _, p in self.gf.named_parameters():
                p.data.copy_(torch.from_numpy(kernel_info))
            img = self.gf(img).detach()
            img = img.squeeze()
        return img

class CheckGrayscale(object):
    def __init__(self):
        pass

    def __call__(self, img):
        shape = img.shape
        if len(shape) == 3 and shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif len(shape) == 4 and shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)
        return img



def create_transform(args, is_train, offsets):
    if is_train:
        color_jitter = T.ColorJitter(0.8*args.color_distort, 0.8*args.color_distort, 0.8*args.color_distort, 0.2*args.color_distort)

        transform = [
            T.RandomResizedCrop(args.img_size, scale=args.crop[:2], ratio=args.crop[2:]),
            T.RandomHorizontalFlip(p=args.flip_prob),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=args.grayscale_prob),
            T.ToTensor(),
            GaussianBlur(args.gaussian_vals[:2], args.gaussian_vals[2], args.gaussian_vals[3]),

            #Check general size etc
            CheckGrayscale(),
            T.Normalize(offsets[0], offsets[1])
        ]

    else:
        transform = [
            T.Resize(size=args.img_size),
            T.ToTensor(),
            CheckGrayscale(),
            T.Normalize(offsets[0], offsets[1])
        ]
    return T.Compose(transform)







