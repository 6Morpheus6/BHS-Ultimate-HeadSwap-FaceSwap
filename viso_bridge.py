import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Tuple

# Add VisoMaster to path
viso_master_path = os.path.join(os.path.dirname(__file__), "VisoMaster")
if viso_master_path not in sys.path:
    sys.path.insert(0, viso_master_path)

from app.processors.models_processor import ModelsProcessor
from app.helpers.miscellaneous import ParametersDict

# Mock MainWindow for ModelsProcessor
class MockMainWindow:
    def __init__(self):
        self.model_loading_signal = self.MockSignal()
        self.model_loaded_signal = self.MockSignal()
        self.model_load_dialog = None
        self.parameters = {}
        self.control = {
            'DetectorModelSelection': 'RetinaFace',
            'DetectorScoreSlider': 50,
            'MaxFacesToDetectSlider': 5,
            'SimilarityTypeSelection': 'Opal',
            'RecognitionModelSelection': 'Inswapper128ArcFace',
            'LandmarkDetectToggle': True,
            'LandmarkDetectModelSelection': '203',
            'LandmarkDetectScoreSlider': 50,
            'DetectFromPointsToggle': False,
            'AutoRotationToggle': True,
            'ManualRotationEnableToggle': False,
            'ManualRotationAngleSlider': 0,
            'FrameEnhancerEnableToggle': False,
            'FrameEnhancerTypeSelection': 'RealEsrgan-x2-Plus',
            'FrameEnhancerBlendSlider': 100,
            'ShowAllDetectedFacesBBoxToggle': False,
            'ShowLandmarksEnableToggle': False,
        }
        self.default_parameters = {
            'SimilarityThresholdSlider': 50,
            'SwapModelSelection': 'Inswapper128',
            'SwapperResSelection': '128',
            'StrengthEnableToggle': False,
            'StrengthAmountSlider': 100,
            'FaceAdjEnableToggle': False,
            'KpsXSlider': 0,
            'KpsYSlider': 0,
            'KpsScaleSlider': 0,
            'FaceScaleAmountSlider': 0,
            'LandmarksPositionAdjEnableToggle': False,
            'FaceExpressionEnableToggle': False,
            'FaceRestorerEnableToggle': False,
            'FaceRestorerDetTypeSelection': 'RetinaFace',
            'FaceRestorerTypeSelection': 'GFPGANv1.4',
            'FaceRestorerBlendSlider': 100,
            'FaceFidelityWeightDecimalSlider': 0.5,
            'OccluderEnableToggle': False,
            'DFLXSegEnableToggle': False,
            'FaceParserEnableToggle': False,
            'ClipEnableToggle': False,
            'RestoreMouthEnableToggle': False,
            'RestoreEyesEnableToggle': False,
            'DifferencingEnableToggle': False,
            'AutoColorEnableToggle': False,
            'ColorEnableToggle': False,
            'JPEGCompressionEnableToggle': False,
            'FinalBlendAdjEnableToggle': False,
            'OverallMaskBlendAmountSlider': 0,
            'FaceEditorEnableToggle': False,
            'FaceMakeupEnableToggle': False,
            'HairMakeupEnableToggle': False,
            'EyeBrowsMakeupEnableToggle': False,
            'LipsMakeupEnableToggle': False,
        }
        self.target_faces = {}
        self.editFacesButton = self.MockButton()
        self.swapfacesButton = self.MockButton()
        self.faceCompareCheckBox = self.MockButton()
        self.faceMaskCheckBox = self.MockButton()

    class MockSignal:
        def emit(self, *args, **kwargs):
            pass

    class MockButton:
        def __init__(self):
            self._checked = False
        def isChecked(self):
            return self._checked
        def setChecked(self, value):
            self._checked = value

class VisoBridge:
    def __init__(self, device='cuda'):
        self.mock_win = MockMainWindow()
        self.processor = ModelsProcessor(self.mock_win, device=device)
        self.processor.switch_providers_priority("CUDA" if device == 'cuda' else "CPU")

    def process_image(self, source_img: Image.Image, target_img: Image.Image, swapper_model="Inswapper128") -> Image.Image:
        # Convert PIL to numpy RGB
        source_np = np.array(source_img.convert("RGB"))
        target_np = np.array(target_img.convert("RGB"))

        # 1. Detect source face for embedding
        source_tensor = torch.from_numpy(source_np).to(self.processor.device).permute(2,0,1)
        bboxes_s, kpss_5_s, _ = self.processor.run_detect(source_tensor, max_num=1)
        if len(kpss_5_s) == 0:
            return target_img # No face detected in source
        
        source_emb, _ = self.processor.run_recognize_direct(source_tensor, kpss_5_s[0])

        # 2. Detect target faces
        target_tensor = torch.from_numpy(target_np).to(self.processor.device).permute(2,0,1)
        bboxes_t, kpss_5_t, kpss_all_t = self.processor.run_detect(target_tensor)
        
        if len(kpss_5_t) == 0:
            return target_img # No face detected in target

        # 3. Swap each face in target
        result_tensor = target_tensor.clone()
        params = self.mock_win.default_parameters.copy()
        params['SwapModelSelection'] = swapper_model
        
        for i in range(len(kpss_5_t)):
            result_tensor, _, _ = self.processor.face_swappers.swap_core_simplified(
                result_tensor, kpss_5_t[i], s_e=source_emb, parameters=params
            )

        # Convert back to PIL
        result_np = result_tensor.permute(1,2,0).cpu().numpy().astype(np.uint8)
        return Image.fromarray(result_np)

    def process_video(self, source_img: Image.Image, target_video_path: str, output_path: str, swapper_model="Inswapper128", progress=None):
        import cv2
        
        # 1. Get source embedding
        source_np = np.array(source_img.convert("RGB"))
        source_tensor = torch.from_numpy(source_np).to(self.processor.device).permute(2,0,1)
        bboxes_s, kpss_5_s, _ = self.processor.run_detect(source_tensor, max_num=1)
        if len(kpss_5_s) == 0:
            raise ValueError("No face detected in source image")
        source_emb, _ = self.processor.run_recognize_direct(source_tensor, kpss_5_s[0])

        # 2. Open video
        cap = cv2.VideoCapture(target_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        params = self.mock_win.default_parameters.copy()
        params['SwapModelSelection'] = swapper_model

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Frame is BGR from cv2
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).to(self.processor.device).permute(2,0,1)
            
            bboxes_t, kpss_5_t, _ = self.processor.run_detect(frame_tensor)
            
            result_tensor = frame_tensor.clone()
            for i in range(len(kpss_5_t)):
                result_tensor, _, _ = self.processor.face_swappers.swap_core_simplified(
                    result_tensor, kpss_5_t[i], s_e=source_emb, parameters=params
                )
            
            result_np = result_tensor.permute(1,2,0).cpu().numpy().astype(np.uint8)
            result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
            out.write(result_bgr)
            
            frame_count += 1
            if progress:
                progress(frame_count / total_frames, desc=f"Processing frame {frame_count}/{total_frames}")

        cap.release()
        out.release()
        return output_path
