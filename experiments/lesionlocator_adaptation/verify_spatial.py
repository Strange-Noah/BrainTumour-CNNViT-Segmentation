"""CPU acceptance of frozen geometry using analytic transforms and official operators."""
import json
import unittest

import numpy as np
import torch
from icon_registration.mermaidlite import compute_warped_image_multiNC

from cloud.storage import connect
from experiments.lesionlocator_benchmark.roi_padding import pad_spatial,spatial_padding,unpad_spatial
from .geometry_contract import native_to_work,registration_to_work,work_to_native,work_to_registration,z_threshold
from .intensity_fit import PREFIX,IDENTITY,read_private
from .spatial import config_for
from .source_ct import official


class Coordinates(unittest.TestCase):
    def test_native_identity_and_crop(self):
        xyz=np.array([18,31,8])
        shape=np.array([61,100,97])
        starts=np.array([5,10,14])
        perm=[0,1,2]
        equal=native_to_work(xyz,shape,shape,starts,perm)
        np.testing.assert_array_equal(equal,xyz[::-1]-starts)
        np.testing.assert_array_equal(work_to_native(equal,shape,shape,starts,perm),xyz)

    def test_oblique_and_anisotropic_center_map(self):
        matrix=np.array([[0,-.8,.1,30],[.9,0,0,-3],[0,0,2.7,11],[0,0,0,1]])
        point=np.array([4.7,31.,21.])
        crop=(37,68,82)
        work=(21,61,55)
        origin=(3,5,7)
        perm=(1,0,2)
        mapped=native_to_work(point,crop,work,origin,perm)
        native=work_to_native(mapped,crop,work,origin,perm)
        np.testing.assert_allclose(matrix@np.r_[native,1],matrix@np.r_[point,1],atol=1e-10)

    def test_registration_different_shapes_and_windows(self):
        point=np.array([10.5,31.25,12.75])
        dims=(53,61,97)
        p=work_to_registration(point,dims,7)
        np.testing.assert_allclose(registration_to_work(p,dims,7),point)
        mapped=registration_to_work(p,(74,81,59),2)
        np.testing.assert_allclose(work_to_registration(mapped,(74,81,59),2),p)
        self.assertEqual([z_threshold(i) for i in (3.,2.,1.5,1.)],[20,30,40,60])

    def test_official_warp_identity_translation_and_boundary(self):
        dims=(7,9,11)
        spacing=[1./(n-1) for n in dims]
        phi=torch.stack(torch.meshgrid(*(torch.linspace(0,1,n) for n in dims),indexing="ij"))[None]
        mask=torch.zeros((1,1,*dims))
        mask[0,0,3,4,5]=7
        torch.testing.assert_close(compute_warped_image_multiNC(mask,phi,spacing,0,True),mask)
        reverse=phi.clone()
        reverse[:,1]+=spacing[1]
        warped=compute_warped_image_multiNC(mask,reverse,spacing,0,True)
        self.assertEqual(float(warped[0,0,3,3,5]),7.)
        self.assertEqual(int(warped.count_nonzero()),1)
        reverse[:,1]+=2
        self.assertEqual(int(compute_warped_image_multiNC(mask,reverse,spacing,0,True).count_nonzero()),0)

    def test_padding_channels_and_instance_value(self):
        value=torch.zeros((1,2,7,13,21))
        value[0,1,3,5,8]=11
        widths=spatial_padding(value,(10,16,26))
        padded=pad_spatial(value,widths)
        self.assertEqual(tuple(padded.shape),(1,2,10,16,26))
        torch.testing.assert_close(unpad_spatial(padded,widths),value)


def main():
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Coordinates))
    if not result.wasSuccessful():
        raise RuntimeError("Synthetic spatial acceptance failed")
    c,b=connect(output=True)
    fit=json.loads(read_private(c,b,f"{PREFIX}/intensity/{IDENTITY}/fit.json"))
    folder=f"{PREFIX}/spatial/{fit['plans_sha256']}"
    frozen=json.loads(read_private(c,b,folder+"/selection.json"))
    validated=json.loads(read_private(c,b,folder+"/validation-summary.json"))
    if validated["patients"]!=45:
        raise ValueError("Independent geometry coverage incomplete")
    params,plans,data=official(c,b)
    if plans["image_reader_writer"]!="SimpleITKIO":
        raise ValueError("Explicit transform-chain code requires a confirmed SimpleITK reader")
    payload=read_private(c,b,folder+f"/combined-{frozen['plans_sha256']}.json")
    from hashlib import sha256
    if sha256(payload).hexdigest()!=frozen["plans_sha256"]:
        raise ValueError("Combined plans digest mismatch")
    manager,config=config_for(json.loads(payload),params["configuration"],frozen["spacing"])
    if tuple(config.spacing)!=tuple(frozen["spacing"]):
        raise ValueError("Preprocessing and official exporter configuration mismatch")
    print(json.dumps(dict(event="s3_acceptance",tests=result.testsRun,passed=True,
         spacing=config.spacing,reader=plans["image_reader_writer"],configuration=params["configuration"],
         official_sha256=params["checkpoint_sha256"],dataset_sha256=params["dataset_sha256"],
         plans_sha256=frozen["plans_sha256"],z_vox=z_threshold(config.spacing[0]),
         registration=[175]*3,unet=[96,256,256],validation_patients=45,validation_pairs=114)))


if __name__=="__main__":
    main()
