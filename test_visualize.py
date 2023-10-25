#!/usr/bin/env python3

"""Unit test running through computations of mae_visualize notebook."""

import os
import shutil
import unittest

import PIL
import matplotlib.pyplot as pyplot
import numpy
import requests
import torch

import models_mae

IMAGENET_MEAN = numpy.array([0.485, 0.456, 0.406])
IMAGENET_STD = numpy.array([0.229, 0.224, 0.225])


def _cache_url(url, path):
    try:
        check_fp = open(path, "rb")
        check_fp.close()
        return
    except FileNotFoundError:
        pass

    try:
        print(f"fetching {url}")

        response = requests.get(url, stream=True)
        with open(path, "wb") as write_fp:
            shutil.copyfileobj(response.raw, write_fp)
    except:
        # clean up failed downloads
        os.unlink(path)
        raise


def _show_image(image, title=""):
    # image is [H, W, 3]
    assert image.shape[2] == 3

    # unnormalize image
    image = image * IMAGENET_STD + IMAGENET_MEAN

    # convert from [0,1] to [0, 255]
    image = torch.clip(image * 255, 0, 255)

    pyplot.imshow(image.int())
    pyplot.title(title, fontsize=16)
    pyplot.axis("off")


class VisualizeTest(unittest.TestCase):
    CHECKPOINT_PATH = "/mae/cache/mae_visualize_vit_large.pth"
    CHECKPOINT_URL = (
        "https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth"
    )

    TEST_IMAGE_PATH = "/mae/cache/fox.jpg"
    TEST_IMAGE_URL = "https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg"

    def setUp(self):
        self.arch = "mae_vit_large_patch16"
        self.model = getattr(models_mae, self.arch)()

        checkpoint = torch.load(self.CHECKPOINT_PATH, map_location="cpu")
        msg = self.model.load_state_dict(checkpoint["model"], strict=False)
        print(msg)

        # make random mask reproducible
        torch.manual_seed(2)

    @classmethod
    def setUpClass(cls):
        _cache_url(cls.CHECKPOINT_URL, cls.CHECKPOINT_PATH)
        _cache_url(cls.TEST_IMAGE_URL, cls.TEST_IMAGE_PATH)

    def test_mask_ratio(self):
        for mask_ratio in [0.25, 0.50, 0.75, 0.80, 0.85, 0.90, 0.95]:
            self.try_model_and_plot(
                f"mask_ratio={mask_ratio:.2f}", mask_ratio=mask_ratio
            )

    def try_model_and_plot(self, test_name, **model_params):
        test_image = PIL.Image.open(self.TEST_IMAGE_PATH)
        test_image = test_image.resize((224, 224))
        test_image = numpy.array(test_image) / 255.0
        self.assertEqual(test_image.shape, (224, 224, 3))

        test_image = test_image - IMAGENET_MEAN
        test_image = test_image / IMAGENET_STD

        x = torch.tensor(test_image)

        # make it a batch-like
        x = x.unsqueeze(dim=0)
        x = torch.einsum("nhwc->nchw", x)

        # run MAE
        loss, y, mask = self.model(x.float(), **model_params)
        y = self.model.unpatchify(y)
        y = torch.einsum("nchw->nhwc", y).detach().cpu()

        # visualize the mask
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(
            1, 1, self.model.patch_embed.patch_size[0] ** 2 * 3
        )  # (N, H*W, p*p*3)
        mask = self.model.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum("nchw->nhwc", mask).detach().cpu()

        x = torch.einsum("nchw->nhwc", x)

        # masked image
        im_masked = x * (1 - mask)

        # MAE reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + y * mask

        # make the pyplot figure larger
        pyplot.rcParams["figure.figsize"] = [24, 24]

        pyplot.subplot(1, 5, 1)
        _show_image(x[0], "original")

        pyplot.subplot(1, 5, 2)
        _show_image(im_masked[0], "masked")

        pyplot.subplot(1, 5, 3)
        _show_image(y[0], "reconstruction")

        pyplot.subplot(1, 5, 4)
        _show_image(im_paste[0], "reconstruction + visible")

        pyplot.show()
        pyplot.savefig(f"/mae/cache/test-{test_name}.png")


############################################################
# startup handling #########################################
############################################################

if __name__ == "__main__":
    unittest.main()
