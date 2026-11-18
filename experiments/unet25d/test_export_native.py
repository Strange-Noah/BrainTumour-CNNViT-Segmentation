import hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
import nibabel as nib
from .export_native import export_native
class NativeExport(unittest.TestCase):
 def setup_case(self,root,partition='validation'):
  split=root/'split.csv';split.write_text('patient_id,partition\np,'+partition+'\n');digest=hashlib.sha256(split.read_bytes()).hexdigest()
  meta=dict(patient_id='p',exam_id='synthetic',shape=[4,4,3],native_shape=[2,2,3],native_affine=np.eye(4).tolist(),padding=[[1,1],[1,1],[0,0]],oblique_resampled=False)
  logits=np.full((4,4,3),-1,dtype=np.float32);logits[1,1,1]=1
  return split,digest,meta,logits
 def test_existing_work_folder_new_export_child_and_frozen_provenance(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);split,digest,meta,logits=self.setup_case(root);folder=root/'visit';folder.mkdir()
   native,ref=export_native(logits,meta,folder/'prediction','a'*40,'private-checkpoint',split,digest)
   self.assertEqual(native.shape,(2,2,3));self.assertEqual(native.sum(),1);image=nib.load(folder/'prediction/seg.nii.gz');self.assertTrue(np.array_equal(np.asarray(image.dataobj),native));self.assertEqual(image.header.get_xyzt_units()[0],'mm')
   record=json.loads((folder/'prediction/prediction.json').read_text());self.assertEqual(record['split_sha256'],digest);self.assertEqual(record['git_sha'],'a'*40)
 def test_wrong_partition_or_split_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);split,digest,meta,logits=self.setup_case(root,'train')
   with self.assertRaises(ValueError):export_native(logits,meta,root/'bad','a'*40,'checkpoint',split,digest)
   with self.assertRaises(ValueError):export_native(logits,meta,root/'bad','a'*40,'checkpoint',split,'0'*64)
   self.assertFalse((root/'bad').exists())
 def test_nonfinite_saved_logits_are_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);split,digest,meta,logits=self.setup_case(root);logits[0,0,0]=np.nan
   with self.assertRaises(FloatingPointError):export_native(logits,meta,root/'bad','a'*40,'checkpoint',split,digest)
if __name__=='__main__':unittest.main()
