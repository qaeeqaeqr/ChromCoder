import torch
import torchvision.transforms as transforms
from einops import rearrange
import PIL.Image as Image
import PIL.ImageOps as PIO
import PIL.ImageEnhance as PIE
import matplotlib.pyplot as plt
import numpy as np
import random
from data.cms_patch.patching import single_patching
import time


class CMS_Trans_for_inference(object):
    """Convert tensor image into grid of patches, where each path overlaps half of its neighbours

    Args:
        grid_size (int): defines the output grid size for the patchification
    """

    def __init__(self, args, eval, aug, patch_size=32):
        self.args = args
        self.eval = eval
        self.aug = aug
        self.patch_size = patch_size

    def __call__(self, x):
        # 这个call方法使得这个类可以作为transforms的一部分。
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be patchified.

        Returns:
            Tensor: Patchified Tensor image of shape (grid_size x grid_size x C x patch_size x patch_size)
        """

        patches = single_patching(np.array(x), self.patch_size, self.args.grid_size)
        patches = patches.reshape(patches.shape[0] * patches.shape[1], patches.shape[2], patches.shape[3])
        patches = patches.astype(np.uint8)  # ToTensor only scale the data with type 'uint8'

        patches = transforms.ToTensor()(patches).float()  # (3, 32*13, 32)
        patches = transforms.Normalize(mean=self.aug['mean'], std=self.aug['std'])(patches)

        patches = rearrange(patches, 'c (g p1) p2 -> g c p1 p2', g=self.args.grid_size, p1=self.patch_size)
        patches = patches.unsqueeze(dim=1)
        patches = patches.unsqueeze(dim=0)

        return patches

    def __repr__(self):
        return self.__class__.__name__ + '(grid_size={0})'.format(self.grid_size)


class SimplePatchify(object):
    """
    Patchify for chromosome_patched dataset
    """
    def __init__(self, grid_size, patch_size=32, patch_num=13):
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.patch_num = patch_num

    def __call__(self, x, *args, **kwargs):
        grid_patches = x.view(self.patch_num, self.patch_size, self.patch_num, self.patch_size, 3)
        grid_patches = grid_patches.permute(0, 2, 1, 3, 4)
        grid_patches = grid_patches.contiguous().view(self.patch_num, self.patch_num, self.patch_size, self.patch_size, 3)

        grid_patches = grid_patches.permute(0, 1, 4, 2, 3).float()

        return grid_patches[:, 0:1, :, :, :]


class CMS_Trans(object):
    def __init__(self, args, eval, aug):
        self.args = args
        self.eval = eval
        self.aug = aug

    def __call__(self, x):
        if self.args.image_resize:
            x1 = transforms.Resize(args.image_resize)(x)
        else:
            x1 = x
        if not self.eval:
            x1 = transforms.RandomHorizontalFlip()(x1)
        if self.args.gray:
            x1 = transforms.Grayscale()(x1)
            x1 = transforms.ToTensor()(x1)
            if self.eval or not self.args.patch_aug:
                x1 = transforms.Normalize(mean=self.aug["bw_mean"], std=self.aug["bw_std"])(x1)
        else:
            # trans.append(transforms.RandomGrayscale(p=0.25)) # As in CPCV2
            x1 = transforms.ToTensor()(x1)
            if self.eval or not self.args.patch_aug:
                x1 = transforms.Normalize(mean=self.aug["mean"], std=self.aug["std"])(x1)
        if not self.args.fully_supervised:
            if not self.eval and self.args.patch_aug:
                x1 = PatchifyAugment(gray=self.args.gray, grid_size=self.args.grid_size)(x1)
            else:
                x1 = SimplePatchify(grid_size=self.args.grid_size)(x1)
        if not self.eval and self.args.patch_aug:
            if self.args.gray:
                x1 = PatchAugNormalize(mean=self.aug["bw_mean"], std=self.aug["bw_std"])(x1)
            else:
                x1 = PatchAugNormalize(mean=self.aug["mean"], std=self.aug["std"])(x1)
        return x1

class PatchifyAugment(object):
    """Convert tensor image into grid of patches, where each path overlaps half of its neighbours. 
    Then applies patch based augmentation.

    Args:
        gray (boolean): defines whether the input tensor is coloured or not
        grid_size (int): defines the output grid size for the patchification
    """

    def __init__(self, gray, grid_size):
        super().__init__(grid_size=grid_size)
        self.gray = gray
        
        # As labeled certain transformations have been written so that they
        # are applied on tensors, this alleviates the need to convert to PIL.Image
        self.transformations = [
            self.ShearX, # PIL
            self.ShearY, # PIL
            self.TranslateX, # Tensor
            self.TranslateY, # Tensor
            self.Rotate, # PIL
            PIO.autocontrast, # PIL 
            self.Invert, # Tensor
            PIO.equalize, # PIL 
            self.Solarize, # Tensor
            self.Posterize, # Tensor 
            self.Contrast, # PIL 
            self.Brightness, # PIL 
            self.Sharpness, # PIL 
            self.Cutout # Tensor
        ]

        if not gray:
            self.transformations.append(self.Color)

    def __call__(self, x):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be patchified and augmented.

        Returns:
            Tensor: Patchified and Augmented Tensor image of shape (grid_size x grid_size x C x patch_size x patch_size)
        """
        # Patchify using parent class
        x = super().__call__(x) 
        
        self.number_of_transforms = 2
        self.patch_dim = (self.patch_size, self.patch_size)
        
        # For each path apply augmentation as in CPC V2
        for patch_row in range(self.grid_size):
            for patch_col in range(self.grid_size):
                patch = x[patch_row][patch_col]

                # Randomly choose two of the 16 (15 if grayscale) transformations from AutoAugment
                for _ in range(self.number_of_transforms):
                    rand = random.randint(0, len(self.transformations)) 

                    # Tensor based functions - TranslateX/Y, Invert, Solarize, Posterize, Cutout
                    if rand == 2 or rand == 3 or rand == 6 or rand == 8 or rand == 9 or rand == 13 : 
                        transform = self.transformations[rand]
                        x[patch_row][patch_col] = transform(patch)

                    # Tensor based function - SamplePairing - requires two inputs
                    elif rand == len(self.transformations):
                        other_patch_row = random.randint(0, self.grid_size - 1)
                        other_patch_col = random.randint(0, self.grid_size - 1)
                        other_patch = x[other_patch_row][other_patch_col]

                        x[patch_row][patch_col] = self.SamplePairing(patch, other_patch)

                    # PIL functions
                    else:   
                        # Convert patch from tensor to PIL image
                        patch_PIL = transforms.ToPILImage()(patch)

                        # Choose transform from transformations array
                        transform = self.transformations[rand]
                        patch_PIL = transform(patch_PIL)

                        # Convert PIL back to tensor
                        x[patch_row][patch_col] = transforms.ToTensor()(patch_PIL)

                # Add other CPCV2 Augmentations...

                # 2. Using the primitives from De Fauw et al. (2018)
                # Randomly apply elastic deformation and shearing with a probability of 0.2. 
                # Randomly apply their colorhistogram automentations with a probability of 0.2.
                
                # 3. Randomly apply the color augmentations from
                # Szegedy et al. (2014) with a probability of 0.8.

                # 4. Greyscale with 25% chance
                if random.random() < 0.25:
                    patch_PIL = transforms.ToPILImage()(patch)
                    patch_PIL = transforms.Grayscale()(patch_PIL)
                    x[patch_row][patch_col] = transforms.ToTensor()(patch_PIL)

        return x

    def __repr__(self):
        return self.__class__.__name__ + f'(grid_size={self.grid_size}, gray={self.gray})'

    # The following transformations either use PIL or are performed directly on Tensors
    def ShearX(self, pil_img):
        level = random.random() * 0.6 - 0.3  # [-0.3,0.3] As in AutoAugment
        return pil_img.transform(self.patch_dim, Image.AFFINE, (1, level, 0, 0, 1, 0))


    def ShearY(self, pil_img):
        level = random.random() * 0.6 - 0.3  # [-0.3,0.3] As in AutoAugment
        return pil_img.transform(self.patch_dim, Image.AFFINE, (1, 0, 0, level, 1, 0))


    def TranslateX(self, patch):
        # Autoaugment does [-150,150] pixels which is eqiuvalent to 45% of 331x331 image
        # 1/3 of patch - 45% seems excessive
        pixels = random.randint(int(-self.patch_size/3), int(self.patch_size/3))
        channels = patch.shape[0]

        # (C, H, W) - columns are dim 2
        if pixels < 0:
            patch = torch.cat((patch[:,:,-pixels:], torch.zeros(channels, self.patch_size, -pixels)), dim=2)
        elif pixels > 0:
            patch = torch.cat((torch.zeros(channels, self.patch_size, pixels), patch[:,:,:self.patch_size-pixels]), dim=2)

        return patch
        #return pil_img.transform(self.patch_dim, Image.AFFINE, (1, 0, pixels, 0, 1, 0))


    def TranslateY(self, patch):
        # Autoaugment does [-150,150] pixels which is eqiuvalent to ~1/2 of 331x331 image
        # 1/3 of patch - 45% seems excessive
        pixels = random.randint(int(-self.patch_size/3), int(self.patch_size/3))
        channels = patch.shape[0]

        # (C, H, W) - rows are dim 1
        if pixels < 0:
            patch = torch.cat((patch[:,-pixels:,:], torch.zeros(channels, -pixels, self.patch_size)), dim=1)
        elif pixels > 0:
            patch = torch.cat((torch.zeros(channels, pixels, self.patch_size), patch[:,:self.patch_size-pixels,:]), dim=1)

        return patch
        #return pil_img.transform(self.patch_dim, Image.AFFINE, (1, 0, pixels, 0, 1, 0))
        

    def Rotate(self, pil_img):
        degrees = random.random() * 60 - 30  # [-30, 30] as in AutoSegment
        return pil_img.rotate(degrees)


    def Invert(self, patch):
        return 1 - patch


    def Solarize(self, patch):
        threshold = random.random() # [0, 1) - equivalent to [0, 256] as in AutoSegment
        cond = patch >= threshold
        patch[cond] = 1 - patch[cond]
        return patch
        #threshold = random.randint(0, 256)  # [0, 256] as in AutoSegment
        #return PIO.solarize(pil_img, threshold)


    def Posterize(self, patch):
        bits = random.randint(4, 8)  # [4,8] as in AutoSegment
        patch = (patch * 255) // (2 ** (8 - bits)) * (2 ** (8-bits)) / 255
        return patch
        #return PIO.posterize(pil_img, bits)


    def Contrast(self, pil_img):
        level = random.random() * 1.8 + 0.1  # [0.1,1.9] As in AutoAugment
        return PIE.Contrast(pil_img).enhance(level)


    def Color(self, pil_img):
        level = random.random() * 1.8 + 0.1  # [0.1,1.9] As in AutoAugment
        return PIE.Color(pil_img).enhance(level)


    def Brightness(self, pil_img):
        level = random.random() * 1.8 + 0.1  # [0.1,1.9] As in AutoAugment
        return PIE.Brightness(pil_img).enhance(level)


    def Sharpness(self, pil_img):
        level = random.random() * 1.8 + 0.1  # [0.1,1.9] As in AutoAugment
        return PIE.Sharpness(pil_img).enhance(level)


    def Cutout(self, patch):
        # Autoaugment does [0, 60] pixels which is eqiuvalent to ~1/5th of 331x331 image
        # 1/3 of patch otherwise they are too small
        size = random.randint(1, int(self.patch_size/3))

        # generate top_left crop coordinate
        x_coord = random.randint(0, self.patch_size - size)
        y_coord = random.randint(0, self.patch_size - size)

        patch[:, x_coord:x_coord+size, y_coord:y_coord+size] = 0.5

        return patch


    def SamplePairing(self, patch, other_patch):
        level = random.random() * 0.4 # [0, 0.4] As in AutoAugment
        return patch + level * other_patch


class PrePatchAugNormalizeReshape(object):
    """
    Converts a tensor (grid_size x grid_size x C x patch_size x patch_size) to (C x  grid_size**2 x patch_size**2)
    """
    def __call__(self, img):
        # Move C to start
        img = img.permute(2, 0, 1, 3, 4)

        # Combine dimnensions
        img = img.view(img.shape[0], img.shape[1] * img.shape[2], img.shape[3] * img.shape[4]) 

        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


class PostPatchAugNormalizeReshape(object):
    """
    Converts a tensor (C x  grid_size**2 x patch_size**2) to (grid_size x grid_size x C x patch_size x patch_size)
    """
    def __call__(self, img):
        # Calcualte grid and patch size
        grid_size = int(img.shape[1] ** 0.5)
        patch_size = int(img.shape[2] ** 0.5)

        # Get rid of grid_size**2 and patch_size**2
        img = img.view(img.shape[0], grid_size, grid_size, patch_size, patch_size) 

        # Move C to dim 2
        img = img.permute(1, 2, 0, 3, 4)

        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


class PatchAugNormalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given the input shape of (grid_size x grid_size x C x patch_size x patch_size) 
    it performs reshaping before and after the normalization

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Args:
            tensor (Tensor): Tensor image of size (grid_size x grid_size x C x patch_size x patch_size) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        img = PrePatchAugNormalizeReshape()(img)
        img = transforms.Normalize(mean=self.mean, std=self.std)(img)
        img = PostPatchAugNormalizeReshape()(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'