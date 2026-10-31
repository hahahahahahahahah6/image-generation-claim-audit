import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from image_generation_claim_audit.audit import audit, claims, evidence, matches
from image_generation_claim_audit.phrases import IMAGEGENERATION, NOIMAGEGENERATION

FIXTURES = Path(__file__).parent / 'fixtures'

class ClaimsTests(unittest.TestCase):
    def test_fences(self):
        for marker in ('```', '~~~', '   ````'):
            self.assertEqual(claims(marker+'python\nsupports image generation\n'+marker), (False, False))
    def test_inline(self):
        self.assertEqual(claims('`supports image generation`'), (True, False))
    def test_negative_span(self):
        self.assertEqual(claims('does not support image generation'), (False, True))
    def test_hyphen(self):
        self.assertEqual(claims('text-to-image'), (True, False))
    def test_contraction(self):
        self.assertEqual(claims("doesn't support image generation"), (False, True))
    def test_paragraph_join(self):
        self.assertEqual(claims('supports image\ngeneration'), (True, False))
        self.assertEqual(claims('supports image\n\ngeneration'), (False, False))
    def test_longest(self):
        self.assertEqual(matches('supports image generation', ('image generation', 'supports image generation')), [(0,25)])
    def test_boundaries(self):
        self.assertEqual(claims('ximage generationx'), (False, False))
    def test_sorted(self):
        for phrases in (IMAGEGENERATION, NOIMAGEGENERATION):
            self.assertEqual(tuple(sorted(phrases, key=lambda s:(-len(s),s))), phrases)
    def test_all_templates(self):
        for phrase in IMAGEGENERATION:
            self.assertEqual(claims(phrase), (True, False), phrase)
        for phrase in NOIMAGEGENERATION:
            self.assertEqual(claims(phrase), (False, True), phrase)

class EvidenceTests(unittest.TestCase):
    pass

class IntegrationTests(unittest.TestCase):
    def cli(self, *args):
        return subprocess.run([sys.executable, '-m', 'image_generation_claim_audit', *map(str,args)], capture_output=True, text=True)
    def test_fixtures(self):
        for name, severity, rule in [('good',None,None), ('bad','HIGH','documented-image-generation-claim-false'), ('testonly','MEDIUM','documented-image-generation-claim-false'), ('no-image-generation','HIGH','documented-no-image-generation-claim-false'), ('no-claim',None,None)]:
            with self.subTest(name=name):
                result=self.cli(FIXTURES/name,'--json')
                self.assertEqual(result.returncode,2 if severity=='HIGH' else 0)
                data=json.loads(result.stdout)
                self.assertEqual(set(data),{'verdict','findings','files_scanned'})
                self.assertEqual(len(data['findings']), int(severity is not None))
                if severity:
                    self.assertEqual(data['findings'][0]['severity'],severity)
                    self.assertEqual(data['findings'][0]['rule'],rule)
        self.assertEqual(self.cli(FIXTURES/'testonly','--fail-on','MEDIUM').returncode,2)
    def test_ignore(self):
        self.assertEqual(audit(FIXTURES/'good',['app.py'])['findings'][0]['severity'],'HIGH')
        self.assertEqual(audit(FIXTURES/'bad',['README.md'])['findings'],[])
    def test_verbose(self):
        self.assertIn('Files scanned: 2',self.cli(FIXTURES/'good','--verbose').stdout)
    def test_version(self):
        result=self.cli('--version');self.assertEqual(result.returncode,0);self.assertEqual(result.stdout.strip(),'0.1.0')
    def test_invalid_root(self):
        self.assertEqual(self.cli(FIXTURES/'absent').returncode,1)
    def test_usage(self):
        for args in [(), ('--unknown',), (FIXTURES/'good','--fail-on','LOW')]:
            self.assertEqual(self.cli(*args).returncode,1)
    def test_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'README.md').write_text('supports image generation')
            for directory in ('docs','fixtures','examples'):
                p=root/directory;p.mkdir();(p/'x.py').write_text('client.images.generate()')
            (root/'link.py').symlink_to(FIXTURES/'good'/'app.py')
            (root/'linked').symlink_to(FIXTURES/'good',target_is_directory=True)
            self.assertEqual(audit(root)['findings'][0]['severity'],'HIGH')
    def test_docs_and_separate_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'nested'/'docs';p.mkdir(parents=True)
            (p/'x.md').write_text('supports image generation\n\ndoes not support image generation')
            self.assertEqual(len(audit(root)['findings']),1)
    def test_test_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'README.md').write_text('supports image generation')
            (root/'picture_test.py').write_text('client.images.generate()')
            self.assertEqual(audit(root)['findings'][0]['severity'],'MEDIUM')

def test_positive_direct(self):
    self.assertEqual(evidence('import diffusers\ndiffusers.StableDiffusionPipeline()'), True)
EvidenceTests.test_positive_direct = test_positive_direct

def test_positive_alias(self):
    self.assertEqual(evidence('import diffusers as d\nd.StableDiffusionPipeline()'), True)
EvidenceTests.test_positive_alias = test_positive_alias

def test_positive_from(self):
    self.assertEqual(evidence('from diffusers import StableDiffusionPipeline as SD\nSD()'), True)
EvidenceTests.test_positive_from = test_positive_from

def test_positive_loader(self):
    self.assertEqual(evidence('import diffusers\ndiffusers.DiffusionPipeline.from_pretrained("x")'), True)
EvidenceTests.test_positive_loader = test_positive_loader

def test_positive_loader_alias(self):
    self.assertEqual(evidence('from diffusers import DiffusionPipeline as D\nD.from_pretrained("x")'), True)
EvidenceTests.test_positive_loader_alias = test_positive_loader_alias

def test_positive_single_file(self):
    self.assertEqual(evidence('from diffusers import FluxPipeline\nFluxPipeline.from_single_file("x")'), True)
EvidenceTests.test_positive_single_file = test_positive_single_file

def test_positive_auto(self):
    self.assertEqual(evidence('from diffusers import AutoPipelineForText2Image\nAutoPipelineForText2Image()'), True)
EvidenceTests.test_positive_auto = test_positive_auto

def test_positive_flux(self):
    self.assertEqual(evidence('from diffusers import FluxPipeline\nFluxPipeline()'), True)
EvidenceTests.test_positive_flux = test_positive_flux

def test_positive_ddpm(self):
    self.assertEqual(evidence('from diffusers import DDPMPipeline\nDDPMPipeline()'), True)
EvidenceTests.test_positive_ddpm = test_positive_ddpm

def test_positive_positional(self):
    self.assertEqual(evidence('import transformers as t\nt.pipeline("text-to-image")'), True)
EvidenceTests.test_positive_positional = test_positive_positional

def test_positive_keyword(self):
    self.assertEqual(evidence('from transformers import pipeline as p\np(task="image-generation")'), True)
EvidenceTests.test_positive_keyword = test_positive_keyword

def test_positive_hosted(self):
    self.assertEqual(evidence('client.images.generate(prompt="a")'), True)
EvidenceTests.test_positive_hosted = test_positive_hosted

def test_positive_function(self):
    self.assertEqual(evidence('from diffusers import FluxPipeline\ndef run():\n return FluxPipeline()'), True)
EvidenceTests.test_positive_function = test_positive_function

def test_negative_import(self):
    self.assertEqual(evidence('from diffusers import StableDiffusionPipeline'), False)
EvidenceTests.test_negative_import = test_negative_import

def test_negative_pil(self):
    self.assertEqual(evidence('from PIL import Image\nImage.new("RGB",(1,1))'), False)
EvidenceTests.test_negative_pil = test_negative_pil

def test_negative_plot(self):
    self.assertEqual(evidence('import matplotlib.pyplot as p\np.plot([1])'), False)
EvidenceTests.test_negative_plot = test_negative_plot

def test_negative_resize(self):
    self.assertEqual(evidence('import cv2\ncv2.resize(img,(1,1))'), False)
EvidenceTests.test_negative_resize = test_negative_resize

def test_negative_super_resolution(self):
    self.assertEqual(evidence('from realesrgan import RealESRGANer\nRealESRGANer()'), False)
EvidenceTests.test_negative_super_resolution = test_negative_super_resolution

def test_negative_detection(self):
    self.assertEqual(evidence('from ultralytics import YOLO\nYOLO("x").predict(img)'), False)
EvidenceTests.test_negative_detection = test_negative_detection

def test_negative_segmentation(self):
    self.assertEqual(evidence('mp.SelfieSegmentation()'), False)
EvidenceTests.test_negative_segmentation = test_negative_segmentation

def test_negative_classification(self):
    self.assertEqual(evidence('timm.create_model("x")'), False)
EvidenceTests.test_negative_classification = test_negative_classification

def test_negative_face(self):
    self.assertEqual(evidence('cv2.CascadeClassifier("x")'), False)
EvidenceTests.test_negative_face = test_negative_face

def test_negative_hand(self):
    self.assertEqual(evidence('mp.Hands()'), False)
EvidenceTests.test_negative_hand = test_negative_hand

def test_negative_pose(self):
    self.assertEqual(evidence('mp.Pose()'), False)
EvidenceTests.test_negative_pose = test_negative_pose

def test_negative_tracking(self):
    self.assertEqual(evidence('cv2.TrackerCSRT_create()'), False)
EvidenceTests.test_negative_tracking = test_negative_tracking

def test_negative_depth(self):
    self.assertEqual(evidence('torch.hub.load("intel-isl/MiDaS", "MiDaS")'), False)
EvidenceTests.test_negative_depth = test_negative_depth

def test_negative_flow(self):
    self.assertEqual(evidence('cv2.calcOpticalFlowFarneback(a,b)'), False)
EvidenceTests.test_negative_flow = test_negative_flow

def test_negative_ocr(self):
    self.assertEqual(evidence('easyocr.Reader(["en"])'), False)
EvidenceTests.test_negative_ocr = test_negative_ocr

def test_negative_barcode(self):
    self.assertEqual(evidence('pyzbar.decode(img)'), False)
EvidenceTests.test_negative_barcode = test_negative_barcode

def test_negative_caption(self):
    self.assertEqual(evidence('from transformers import pipeline\npipeline("image-to-text")'), False)
EvidenceTests.test_negative_caption = test_negative_caption

def test_negative_text(self):
    self.assertEqual(evidence('from transformers import pipeline\npipeline("text-generation")'), False)
EvidenceTests.test_negative_text = test_negative_text

def test_negative_speech(self):
    self.assertEqual(evidence('from transformers import pipeline\npipeline("text-to-speech")'), False)
EvidenceTests.test_negative_speech = test_negative_speech

def test_negative_video(self):
    self.assertEqual(evidence('from diffusers import AnimateDiffPipeline\nAnimateDiffPipeline()'), False)
EvidenceTests.test_negative_video = test_negative_video

def test_negative_shadow(self):
    self.assertEqual(evidence('from diffusers import StableDiffusionPipeline\ndef StableDiffusionPipeline(): pass\nStableDiffusionPipeline()'), False)
EvidenceTests.test_negative_shadow = test_negative_shadow

def test_negative_string(self):
    self.assertEqual(evidence('s = "image generation"'), False)
EvidenceTests.test_negative_string = test_negative_string

def test_negative_subprocess(self):
    self.assertEqual(evidence('import subprocess\nsubprocess.run(["generate-images"])'), False)
EvidenceTests.test_negative_subprocess = test_negative_subprocess

def test_negative_syntax(self):
    self.assertEqual(evidence('def broken('), False)
EvidenceTests.test_negative_syntax = test_negative_syntax

def test_negative_unresolved(self):
    self.assertEqual(evidence('StableDiffusionPipeline()'), False)
EvidenceTests.test_negative_unresolved = test_negative_unresolved

def test_negative_wrong_provider(self):
    self.assertEqual(evidence('from elsewhere import FluxPipeline\nFluxPipeline()'), False)
EvidenceTests.test_negative_wrong_provider = test_negative_wrong_provider

def test_negative_argument_shadow(self):
    self.assertEqual(evidence('from diffusers import FluxPipeline\ndef run(FluxPipeline):\n return FluxPipeline()'), False)
EvidenceTests.test_negative_argument_shadow = test_negative_argument_shadow

def test_negative_class_shadow(self):
    self.assertEqual(evidence('from diffusers import FluxPipeline\nclass FluxPipeline: pass\nFluxPipeline()'), False)
EvidenceTests.test_negative_class_shadow = test_negative_class_shadow
