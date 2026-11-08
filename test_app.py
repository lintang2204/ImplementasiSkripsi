import io
import unittest
import zipfile
import numpy as np
from PIL import Image
from inference import PLANES,TISSUES,collect_images,summarize,THRESHOLD

def png():
    b=io.BytesIO();Image.fromarray(np.full((224,224),100,dtype=np.uint8)).save(b,format='PNG');return b.getvalue()

class Checks(unittest.TestCase):
    def test_complete(self):
        images,subject=collect_images([(f'SUB_{t}_{p}_slice_80.png',png()) for p in PLANES for t in TISSUES])
        self.assertEqual(len(images),12);self.assertEqual(subject,'SUB')
    def test_duplicate(self):
        with self.assertRaisesRegex(ValueError,'duplikat'): collect_images([('SUB_ANAT_axial.png',png())]*2)
    def test_mixed(self):
        with self.assertRaisesRegex(ValueError,'lebih dari satu'):collect_images([('ONE_ANAT_axial.png',png()),('TWO_CSF_axial.png',png())])
    def test_zip(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:
            for p in PLANES:
                for t in TISSUES:z.writestr(f'folder/SUB_{t}_{p}.png',png())
        self.assertEqual(len(collect_images([('images.zip',b.getvalue())])[0]),12)
    def test_math(self):
        a=np.arange(60,dtype=np.float32).reshape(3,5,4)/60
        r=summarize(a)
        expected=np.mean([float(np.mean(a[i].mean(axis=0).astype(np.float64))) for i in range(3)])
        self.assertAlmostEqual(r['prob_asd'],expected,12)
        self.assertEqual(r['prediction'],'ASD' if expected>=THRESHOLD else 'TC')
    def test_invalid_probability(self):
        with self.assertRaises(ValueError):summarize(np.full((3,5,4),np.nan))

if __name__=='__main__':unittest.main()
