import unittest
import torch
from torch.nn.functional import pad
from .roi_padding import pad_spatial, spatial_padding, unpad_spatial


class TrackNetPadding(unittest.TestCase):
    def test_in_plane_padding_does_not_expand_channels_or_batch(self):
        tensor = torch.arange(2*3*4*5).reshape(1,2,3,4,5)
        widths = spatial_padding(tensor, (6,7,8))
        d0,d1,h0,h1,w0,w1 = widths
        legacy = pad(tensor, (0,0,0,0,d0,d1,h0,h1,w0,w1))
        self.assertEqual(legacy.shape, (4,5,6,4,5))
        result = pad_spatial(tensor, widths)
        self.assertEqual(result.shape, (1,2,6,7,8))
        self.assertTrue(torch.equal(unpad_spatial(result, widths), tensor))

    def test_valid_official_depth_only_is_bitwise_identical(self):
        tensor = torch.arange(2*3*4*5).reshape(1,2,3,4,5)
        widths = spatial_padding(tensor, (6,4,5))
        d0,d1,*_ = widths
        legacy = pad(tensor, (0,0,0,0,d0,d1,0,0,0,0))
        result = pad_spatial(tensor, widths)
        self.assertTrue(torch.equal(result, legacy))
        self.assertTrue(torch.equal(unpad_spatial(result, widths), tensor))

    def test_patch_larger_than_target_fails(self):
        with self.assertRaises(ValueError):
            spatial_padding(torch.zeros(1,2,4,4,4), (3,4,4))


if __name__ == "__main__":
    unittest.main()
