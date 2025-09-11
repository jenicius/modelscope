# The implementation here is modified based on BaSSL,
# originally Apache 2.0 License and publicly available at https://github.com/kakaobrain/bassl

import math
import os
import os.path as osp
import tempfile
import json
import time
import traceback
from typing import Any, Dict, List

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TF
from PIL import Image
from shotdetect_scenedetect_lgss import shot_detector
from tqdm import tqdm

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .get_model import get_contextual_relation_network, get_shot_encoder
from .utils.save_op import get_pred_boundary, pred2scene, scene2video

logger = get_logger()


@MODELS.register_module(
    Tasks.movie_scene_segmentation, module_name=Models.resnet50_bert)
class MovieSceneSegmentationModel(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """str -- model file root."""
        super().__init__(model_dir, *args, **kwargs)

        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        params = torch.load(model_path, map_location='cpu')

        config_path = osp.join(model_dir, ModelFile.CONFIGURATION)
        self.cfg = Config.from_file(config_path)

        def load_param_with_prefix(prefix, model, src_params):
            own_state = model.state_dict()
            for name, param in own_state.items():
                src_name = prefix + '.' + name
                own_state[name] = src_params[src_name]

            model.load_state_dict(own_state)

        self.shot_encoder = get_shot_encoder(self.cfg)
        load_param_with_prefix('shot_encoder', self.shot_encoder, params)
        self.crn = get_contextual_relation_network(self.cfg)
        load_param_with_prefix('crn', self.crn, params)

        crn_name = self.cfg.model.contextual_relation_network.name
        hdim = self.cfg.model.contextual_relation_network.params[crn_name][
            'hidden_size']
        self.head_sbd = nn.Linear(hdim, 2)
        load_param_with_prefix('head_sbd', self.head_sbd, params)

        self.shot_detector = shot_detector()
        self.shot_detector.init(**self.cfg.preprocessor.shot_detect)

        self.test_transform = TF.Compose([
            TF.Resize(size=256, interpolation=Image.BICUBIC),
            TF.CenterCrop(224),
            TF.ToTensor(),
            TF.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        sampling_method = self.cfg.dataset.sampling_method.name
        self.neighbor_size = self.cfg.dataset.sampling_method.params[
            sampling_method].neighbor_size

        self.eps = 1e-5

        # debug folder for writing failing batch context
        self._dbg_dir = osp.join(tempfile.gettempdir(), 'ms_seg_dbg')
        os.makedirs(self._dbg_dir, exist_ok=True)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        data = inputs.pop('video')
        labels = inputs['label']
        outputs = self.shared_step(data)

        loss = F.cross_entropy(
            outputs.squeeze(), labels.squeeze(), reduction='none')
        lpos = labels == 1
        lneg = labels == 0

        pp, nn = 1, 1
        wp = (pp / float(pp + nn)) * lpos / (lpos.sum() + self.eps)
        wn = (nn / float(pp + nn)) * lneg / (lneg.sum() + self.eps)
        w = wp + wn
        loss = (w * loss).sum()

        probs = torch.argmax(outputs, dim=1)

        re = dict(pred=probs, loss=loss)
        return re

    def _dump_debug(self, name: str, data: Dict[str, Any]):
        path = osp.join(self._dbg_dir, f'{int(time.time())}_{name}.json')
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info('Wrote debug context to %s', path)
        except Exception:
            logger.exception('Failed to write debug data')

    def inference(self, batch):
        """More defensive inference. This version:
        - Builds contiguous timecode requests to accommodate forward-only frame readers
        - Retries short per-shot reads if the bulk read is partial
        - Dumps a full batch context to disk for the first failing batch to ease debugging
        - Fills missing predictions with a user-configurable value (default 0.0)
        """
        logger.info('Begin scene detect ......')
        bs = self.cfg.pipeline.batch_size_per_gpu

        try:
            device = next(self.crn.parameters()).device
        except Exception:
            device = torch.device('cpu')

        shot_timecode_lst: List[Any] = batch['shot_timecode_lst']
        shot_idx_lst: List[Any] = batch['shot_idx_lst']

        shot_num = len(shot_timecode_lst)
        cnt = math.ceil(len(shot_idx_lst) / bs)

        fill_missing_with = float(getattr(self.cfg.pipeline, 'fill_missing_prediction', 0.0))

        pred_array = np.full(shot_num, np.nan, dtype=float)
        first_failure_dumped = False

        self.shot_detector.start()

        try:
            for i in tqdm(range(cnt)):
                start = i * bs
                end = min((i + 1) * bs, len(shot_idx_lst))
                batch_shot_idx_lst = shot_idx_lst[start:end]
                if len(batch_shot_idx_lst) == 0:
                    continue

                # flatten neighbor windows and robustly coerce to ints
                requested = []
                for arr in batch_shot_idx_lst:
                    try:
                        arr_np = np.array(arr, dtype=int).ravel()
                        requested.extend(arr_np.tolist())
                    except Exception:
                        try:
                            requested.append(int(arr))
                        except Exception:
                            logger.debug('Skipping uncoercible index %r', arr)

                if len(requested) == 0:
                    continue

                requested_indices = np.unique(np.array(requested, dtype=int))
                # clamp
                requested_indices = requested_indices[(requested_indices >= 0) & (requested_indices < shot_num)]
                if requested_indices.size == 0:
                    continue

                min_idx = int(requested_indices.min())
                max_idx = int(requested_indices.max())

                # build timecode map for contiguous range (makes forward-only readers happy)
                batch_timecode_lst = {j: shot_timecode_lst[j] for j in range(min_idx, max_idx + 1)}

                # defensive: dump context if any index maps to None or raises
                if any(batch_timecode_lst[j] is None for j in batch_timecode_lst):
                    logger.warning('Some timecodes are None in batch %d range %d-%d', i, min_idx, max_idx)

                # try bulk fetch
                try:
                    batch_shot_keyf_lst = self.shot_detector.get_frame_img(batch_timecode_lst, min_idx, shot_num)
                except Exception:
                    logger.exception('bulk get_frame_img failed for batch %d range %d-%d', i, min_idx, max_idx)
                    batch_shot_keyf_lst = []

                # if bulk fetch is partial, attempt per-shot retry with small sleep between tries
                expected_len = max_idx - min_idx + 1
                if len(batch_shot_keyf_lst) != expected_len:
                    logger.warning('Partial or empty bulk result (%d/%d). Attempting per-shot retry.', len(batch_shot_keyf_lst), expected_len)
                    batch_shot_keyf_lst = []
                    for j in range(min_idx, max_idx + 1):
                        try:
                            single_map = {j: shot_timecode_lst[j]}
                            res = self.shot_detector.get_frame_img(single_map, j, shot_num)
                            if not res:
                                logger.warning('Missing keyframe for shot %d during per-shot retry', j)
                                batch_shot_keyf_lst = []
                                break
                            batch_shot_keyf_lst.append(res[0])
                            # tiny sleep to avoid hammering IO / decoders
                            time.sleep(0.005)
                        except Exception:
                            logger.exception('Exception fetching single shot %d', j)
                            batch_shot_keyf_lst = []
                            break

                if len(batch_shot_keyf_lst) == 0:
                    logger.error('Failed to obtain keyframes for batch %d range %d-%d', i, min_idx, max_idx)
                    if not first_failure_dumped:
                        ctx = {
                            'batch_index': i,
                            'min_idx': min_idx,
                            'max_idx': max_idx,
                            'requested': requested_indices.tolist(),
                            'shot_timecode_sample': {j: str(shot_timecode_lst[j]) for j in range(min_idx, min(min_idx + 10, max_idx + 1))},
                            'shot_num': shot_num,
                            'cfg_shot_detect': self.cfg.preprocessor.shot_detect
                        }
                        self._dump_debug('first_failure', ctx)
                        first_failure_dumped = True
                    # skip this batch (alternatively could fill zeros)
                    continue

                # prepare inputs
                inputs = self.get_batch_input(batch_shot_keyf_lst, min_idx, batch_shot_idx_lst)
                if len(inputs) == 0:
                    logger.warning('get_batch_input produced 0 inputs for batch %d', i)
                    continue

                try:
                    input_ = torch.stack(inputs).to(device)
                except Exception:
                    logger.exception('Failed to stack inputs for batch %d; shapes: %s', i, [x.shape if hasattr(x, 'shape') else str(type(x)) for x in inputs])
                    continue

                outputs = self.shared_step(input_)
                prob = F.softmax(outputs, dim=1)

                # map back each window prediction to its center shot id
                for k, shot_idx_window in enumerate(batch_shot_idx_lst):
                    try:
                        window_arr = np.array(shot_idx_window, dtype=int).ravel()
                        center_pos = len(window_arr) // 2
                        center_shot_id = int(window_arr[center_pos])
                        if 0 <= center_shot_id < shot_num:
                            pred_array[center_shot_id] = float(prob[k, 1].cpu().detach().numpy())
                        else:
                            logger.warning('Center id %d out of range for batch %d', center_shot_id, i)
                    except Exception:
                        logger.exception('Failed to map prediction for batch %d window %d', i, k)
                        continue

        finally:
            try:
                self.shot_detector.release()
            except Exception:
                logger.exception('Exception while releasing shot_detector')

        # fill missing
        nan_count = int(np.isnan(pred_array).sum())
        if nan_count > 0:
            logger.warning('Inference finished with %d/%d missing predictions; filling with %f', nan_count, shot_num, fill_missing_with)
            pred_array[np.isnan(pred_array)] = fill_missing_with

        return {'pred': pred_array, 'sid': np.arange(shot_num)}

    def shared_step(self, inputs):
        with torch.no_grad():
            # infer shot encoder
            shot_repr = self.extract_shot_representation(inputs)
            assert len(shot_repr.shape) == 3

        # infer CRN
        _, pooled = self.crn(shot_repr, mask=None)
        # infer boundary score
        pred = self.head_sbd(pooled)
        return pred

    def save_shot_feat(self, _repr):
        feat = _repr.float().cpu().numpy()
        pth = self.cfg.dataset.img_path + '/features'
        os.makedirs(pth, exist_ok=True)

        for idx in range(_repr.shape[0]):
            name = f'shot_{str(idx).zfill(4)}.npy'
            name = osp.join(pth, name)
            np.save(name, feat[idx])

    def extract_shot_representation(self,
                                    inputs: torch.Tensor) -> torch.Tensor:
        """ inputs [b s k c h w] -> output [b d] """
        assert len(inputs.shape) == 6  # (B Shot Keyframe C H W)
        b, s, k, c, h, w = inputs.shape
        inputs = einops.rearrange(inputs, 'b s k c h w -> (b s) k c h w', s=s)
        keyframe_repr = [self.shot_encoder(inputs[:, _k]) for _k in range(k)]
        # [k (b s) d] -> [(b s) d]
        shot_repr = torch.stack(keyframe_repr).mean(dim=0)

        shot_repr = einops.rearrange(shot_repr, '(b s) d -> b s d', s=s)
        return shot_repr

    def postprocess(self, inputs: Dict[str, Any], **kwargs):
        logger.info('Generate scene .......')

        pred_dict = inputs['feat']
        shot2keyf = inputs['shot2keyf']
        thres = self.cfg.pipeline.save_threshold

        anno_dict = get_pred_boundary(pred_dict, thres)
        scene_dict_lst, scene_list, shot_num, shot_dict_lst = pred2scene(
            shot2keyf, anno_dict)
        if self.cfg.pipeline.save_split_scene:
            re_dir = scene2video(inputs['input_video_pth'], scene_list, thres)
            print(f'Split scene video saved to {re_dir}')
        return len(scene_list), scene_dict_lst, shot_num, shot_dict_lst

    def get_batch_input(self, shot_keyf_lst, shot_start_idx, shot_idx_lst):

        single_shot_feat = []
        for idx, one_shot in enumerate(shot_keyf_lst):
            # skip empty frames
            if one_shot is None or len(one_shot) == 0:
                single_shot_feat.append(None)
                continue

            one_shot = [
                self.test_transform(one_frame) for one_frame in one_shot
            ]

            try:
                one_shot = torch.stack(one_shot, dim=0)
            except Exception:
                logger.exception('Failed to stack keyframes for shot_start_idx=%d idx=%d', shot_start_idx, idx)
                single_shot_feat.append(None)
                continue

            single_shot_feat.append(one_shot)

        shot_feat = []
        for idx, shot_idx in enumerate(shot_idx_lst):
            try:
                shot_idx_arr = np.array(shot_idx, dtype=int)
            except Exception:
                shot_idx_arr = np.array([int(shot_idx)], dtype=int)

            shot_idx_ = shot_idx_arr - shot_start_idx

            if np.any(shot_idx_ < 0) or np.any(shot_idx_ >= len(single_shot_feat)):
                logger.warning('Neighbor index out of range for shot_start_idx=%d', shot_start_idx)
                continue

            neighbor_list = []
            skip = False
            for si in shot_idx_:
                val = single_shot_feat[int(si)]
                if val is None:
                    skip = True
                    break
                neighbor_list.append(val)

            if skip:
                logger.debug('Skipping shot because some neighbor keyframes are missing')
                continue

            neighbor_tensor = torch.stack(neighbor_list, dim=0)
            shot_feat.append(neighbor_tensor)

        return shot_feat

    def preprocess(self, inputs):
        logger.info('Begin shot detect......')
        self.shot_detector = shot_detector()
        self.shot_detector.init(**self.cfg.preprocessor.shot_detect)
        shot_timecode_lst, anno, shot2keyf = self.shot_detector.shot_detect(
            inputs, **self.cfg.preprocessor.shot_detect)
        logger.info('Shot detect done!')

        shot_idx_lst = []
        for idx, one_shot in enumerate(anno):
            shot_idx = int(one_shot['shot_id']) + np.arange(
                -self.neighbor_size, self.neighbor_size + 1)
            # clamp to valid timecode length
            shot_idx = np.clip(shot_idx, 0, max(len(shot_timecode_lst) - 1, 0))
            shot_idx_lst.append(shot_idx)

        return shot2keyf, anno, shot_timecode_lst, shot_idx_lst
