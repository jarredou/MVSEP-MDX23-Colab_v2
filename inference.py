# coding: utf-8

if __name__ == '__main__':
    import os

    gpu_use = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import os
import argparse
import soundfile as sf
from demucs.states import load_model
from demucs import pretrained
from demucs.apply import apply_model
import onnxruntime as ort
from time import time
import librosa
import hashlib
from scipy import signal
import gc
import yaml
from ml_collections import ConfigDict
import sys
import math
import pathlib
import warnings
from modules.tfc_tdf_v3 import TFC_TDF_net, STFT
from scipy.signal import resample_poly
from modules.segm_models import Segm_Models_Net
from contextlib import redirect_stdout

class Conv_TDF_net_trim_model(nn.Module):
    def __init__(self, device, target_name, L, n_fft, hop=1024):
        super(Conv_TDF_net_trim_model, self).__init__()
        self.dim_c = 4
        self.dim_f, self.dim_t = 3072, 256
        self.n_fft = n_fft
        self.hop = hop
        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True).to(device)
        self.target_name = target_name
        out_c = self.dim_c * 4 if target_name == '*' else self.dim_c
        self.freq_pad = torch.zeros([1, out_c, self.n_bins - self.dim_f, self.dim_t]).to(device)
        self.n = L // 2

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True, return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, self.dim_c, self.n_bins, self.dim_t])
        return x[:, :, :self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = self.freq_pad.repeat([x.shape[0], 1, 1, 1]) if freq_pad is None else freq_pad
        x = torch.cat([x, freq_pad], -2)
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, 2, self.n_bins, self.dim_t])
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
        return x.reshape([-1, 2, self.chunk_size])

    def forward(self, x):
        x = self.first_conv(x)
        x = x.transpose(-1, -2)

        ds_outputs = []
        for i in range(self.n):
            x = self.ds_dense[i](x)
            ds_outputs.append(x)
            x = self.ds[i](x)

        x = self.mid_dense(x)
        for i in range(self.n):
            x = self.us[i](x)
            x *= ds_outputs[-i - 1]
            x = self.us_dense[i](x)

        x = x.transpose(-1, -2)
        x = self.final_conv(x)
        return x

def get_models(name, device, load=True, vocals_model_type=0):
    if vocals_model_type == 2:
        model_vocals = Conv_TDF_net_trim_model(
            device=device,
            target_name='vocals',
            L=11,
            n_fft=7680
        )
    elif vocals_model_type == 3:
        model_vocals = Conv_TDF_net_trim_model(
            device=device,
            target_name='vocals',
            L=11,
            n_fft=6144
        )

    return [model_vocals]

def demix_base_mdxv3(model, mix, device):
    N = options["overlap_InstVoc"]
    mix = np.array(mix, dtype=np.float32)
    mix = torch.tensor(mix, dtype=torch.float32)
    
    try:
        S = model.num_target_instruments
    except Exception as e:
        S = model.module.num_target_instruments

    mdx_window_size = model.config.inference.dim_t * 2
    batch_size = 1
    C = model.config.audio.hop_length * (mdx_window_size - 1)
    H = C // N
    L = mix.shape[1]
    pad_size = H - (L - C) % H

    mix = torch.cat([torch.zeros(2, C - H), mix, torch.zeros(2, pad_size + C - H)], 1)
    mix = mix.to(device)
    chunks = mix.unfold(1, C, H).transpose(0, 1)
    batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]

    X = torch.zeros(S, *mix.shape).to(device) if S > 1 else torch.zeros_like(mix) 

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            cnt = 0
            for batch in batches:
                 x = model(batch)
                 for w in x:
                    X[..., cnt * H : cnt * H + C] += w
                    cnt += 1

    estimated_sources = X[..., C - H:-(pad_size + C - H)] / N
    
    if S > 1:
        return {k: v for k, v in zip(model.config.training.instruments, estimated_sources.cpu().numpy())}
    else:
        est_s = estimated_sources.cpu().numpy()
        return est_s

def demix_full_mdx23c(mix, device, model):
    if options["BigShifts"] <= 0:
        bigshifts = 1
    else:
        bigshifts = options["BigShifts"]
    shift_in_samples = mix.shape[1] // bigshifts
    shifts = [x * shift_in_samples for x in range(bigshifts)]

    results = []

    for shift in tqdm(shifts, position=0):
        shifted_mix = np.concatenate((mix[:, -shift:], mix[:, :-shift]), axis=-1)
        sources = demix_base_mdxv3(model, shifted_mix, device)["Vocals"]
        sources *= 1.0005168 # volume compensation
        restored_sources = np.concatenate((sources[..., shift:], sources[..., :shift]), axis=-1)
        results.append(restored_sources)

    sources = np.mean(results, axis=0)
    
    return sources


def demix_wrapper(mix, device, models, infer_session, overlap=0.2, bigshifts=1):
    if bigshifts <= 0:
        bigshifts = 1
    shift_in_samples = mix.shape[1] // bigshifts
    shifts = [x * shift_in_samples for x in range(bigshifts)]
    results = []
    
    for shift in tqdm(shifts, position=0):
        shifted_mix = np.concatenate((mix[:, -shift:], mix[:, :-shift]), axis=-1)
        sources = demix(shifted_mix, device, models, infer_session, overlap) * 1.021 # volume compensation
        restored_sources = np.concatenate((sources[..., shift:], sources[..., :shift]), axis=-1)
        results.append(restored_sources)
        
    sources = np.mean(results, axis=0)
    
    return sources

def demix(mix, device, models, infer_session, overlap=0.2):
    start_time = time()
    sources = []
    n_sample = mix.shape[1]
    n_fft = models[0].n_fft
    n_bins = n_fft//2+1
    trim = n_fft//2
    hop = models[0].hop
    dim_f = models[0].dim_f
    dim_t = models[0].dim_t * 2
    chunk_size = models[0].chunk_size
    org_mix = mix
    tar_waves_ = []
    mdx_batch_size = 1
    overlap = overlap
    gen_size = chunk_size-2*trim
    pad = gen_size + trim - ((mix.shape[-1]) % gen_size)
    
    mixture = np.concatenate((np.zeros((2, trim), dtype='float32'), mix, np.zeros((2, pad), dtype='float32')), 1)

    step = int((1 - overlap) * chunk_size)
    result = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
    divider = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
    total = 0
    total_chunks = (mixture.shape[-1] + step - 1) // step

    for i in range(0, mixture.shape[-1], step):
        total += 1
        start = i
        end = min(i + chunk_size, mixture.shape[-1])
        chunk_size_actual = end - start

        if overlap == 0:
            window = None
        else:
            window = np.hanning(chunk_size_actual)
            window = np.tile(window[None, None, :], (1, 2, 1))

        mix_part_ = mixture[:, start:end]
        if end != i + chunk_size:
            pad_size = (i + chunk_size) - end
            mix_part_ = np.concatenate((mix_part_, np.zeros((2, pad_size), dtype='float32')), axis=-1)
        
        
        mix_part = torch.tensor([mix_part_], dtype=torch.float32).to(device)
        mix_waves = mix_part.split(mdx_batch_size)
        
        with torch.no_grad():
            for mix_wave in mix_waves:
                _ort = infer_session
                stft_res = models[0].stft(mix_wave)
                stft_res[:, :, :3, :] *= 0 
                res = _ort.run(None, {'input': stft_res.cpu().numpy()})[0]
                ten = torch.tensor(res)
                tar_waves = models[0].istft(ten.to(device))
                tar_waves = tar_waves.cpu().detach().numpy()
                
                if window is not None:
                    tar_waves[..., :chunk_size_actual] *= window 
                    divider[..., start:end] += window
                else:
                    divider[..., start:end] += 1
                result[..., start:end] += tar_waves[..., :end-start]


    tar_waves = result / divider
    tar_waves_.append(tar_waves)
    tar_waves_ = np.vstack(tar_waves_)[:, :, trim:-trim]
    tar_waves = np.concatenate(tar_waves_, axis=-1)[:, :mix.shape[-1]]
    source = tar_waves[:,0:None]

    return source

def demix_vitlarge(model, mix, device):
    C = model.config.audio.hop_length * (2 * model.config.inference.dim_t - 1)
    N = options["overlap_VitLarge"]
    step = C // N

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            if model.config.training.target_instrument is not None:
                req_shape = (1, ) + tuple(mix.shape)
            else:
                req_shape = (len(model.config.training.instruments),) + tuple(mix.shape)

            mix = mix.to(device)
            result = torch.zeros(req_shape, dtype=torch.float32).to(device)
            counter = torch.zeros(req_shape, dtype=torch.float32).to(device)
            i = 0

            while i < mix.shape[1]:
                part = mix[:, i:i + C]
                length = part.shape[-1]
                if length < C:
                    part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
                x = model(part.unsqueeze(0))[0]
                result[..., i:i+length] += x[..., :length]
                counter[..., i:i+length] += 1.
                i += step
            estimated_sources = result / counter

    if model.config.training.target_instrument is None:
        return {k: v for k, v in zip(model.config.training.instruments, estimated_sources.cpu().numpy())}
    else:
        return {k: v for k, v in zip([model.config.training.target_instrument], estimated_sources.cpu().numpy())}


def demix_full_vitlarge(mix, device, model):
    if options["BigShifts"] <= 0:
        bigshifts = 1
    else:
        bigshifts = options["BigShifts"]
    shift_in_samples = mix.shape[1] // bigshifts
    shifts = [x * shift_in_samples for x in range(bigshifts)]

    results1 = []
    results2 = []
    
    for shift in tqdm(shifts, position=0):
        shifted_mix = torch.cat((mix[:, -shift:], mix[:, :-shift]), dim=-1)
        sources = demix_vitlarge(model, shifted_mix, device)
        sources1 = sources["vocals"] * 1.002 # volume compensation
        sources2 = sources["other"]
        restored_sources1 = np.concatenate((sources1[..., shift:], sources1[..., :shift]), axis=-1)
        restored_sources2 = np.concatenate((sources2[..., shift:], sources2[..., :shift]), axis=-1)
        results1.append(restored_sources1)
        results2.append(restored_sources2)


    sources1 = np.mean(results1, axis=0)
    sources2 = np.mean(results2, axis=0)

    return sources1, sources2


class EnsembleDemucsMDXMusicSeparationModel:
    """
    Doesn't do any separation just passes the input back as output
    """
    def __init__(self, options):
        """
            options - user options
        """

        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        if 'cpu' in options:
            if options['cpu']:
                device = 'cpu'
        # print('Use device: {}'.format(device))
        self.single_onnx = False
        if 'single_onnx' in options:
            if options['single_onnx']:
                self.single_onnx = True
                # print('Use single vocal ONNX')
        self.overlap_demucs = float(options['overlap_demucs'])
        self.overlap_MDX = float(options['overlap_VOCFT'])
        if self.overlap_demucs > 0.99:
            self.overlap_demucs = 0.99
        if self.overlap_demucs < 0.0:
            self.overlap_demucs = 0.0
        if self.overlap_MDX > 0.99:
            self.overlap_MDX = 0.99
        if self.overlap_MDX < 0.0:
            self.overlap_MDX = 0.0
        model_folder = os.path.dirname(os.path.realpath(__file__)) + '/models/'
        """
        
        remote_url = 'https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/04573f0d-f3cf25b2.th'
        model_path = model_folder + '04573f0d-f3cf25b2.th'
        if not os.path.isfile(model_path):
            torch.hub.download_url_to_file(remote_url, model_folder + '04573f0d-f3cf25b2.th')
        model_vocals = load_model(model_path)
        model_vocals.to(device)
        self.model_vocals_only = model_vocals
        """

        if options['vocals_only'] is False:
            self.models = []
            self.weights_vocals = np.array([10, 1, 8, 9])
            self.weights_bass = np.array([19, 4, 5, 8])
            self.weights_drums = np.array([18, 2, 4, 9])
            self.weights_other = np.array([14, 2, 5, 10])

            model1 = pretrained.get_model('htdemucs_ft')
            model1.to(device)
            self.models.append(model1)

            model2 = pretrained.get_model('htdemucs')
            model2.to(device)
            self.models.append(model2)

            model3 = pretrained.get_model('htdemucs_6s')
            model3.to(device)
            self.models.append(model3)

            model4 = pretrained.get_model('hdemucs_mmi')
            model4.to(device)
            self.models.append(model4)

            if 0:
                for model in self.models:
                  pass
                  # print(model.sources)
            '''
            ['drums', 'bass', 'other', 'vocals']
            ['drums', 'bass', 'other', 'vocals']
            ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano']
            ['drums', 'bass', 'other', 'vocals']
            '''

        if device == 'cpu':
            chunk_size = 200000000
            providers = ["CPUExecutionProvider"]
        else:
            chunk_size = 1000000
            providers = ["CUDAExecutionProvider"]
        if 'chunk_size' in options:
            chunk_size = int(options['chunk_size'])

        #MDXv3 init
        print("Loading InstVoc into memory")
        remote_url_mdxv3 = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/MDX23C-8KFFT-InstVoc_HQ.ckpt'
        remote_url_conf_mdxv3 = 'https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_2_stem_full_band_8k.yaml'
        if not os.path.isfile(model_folder+'MDX23C-8KFFT-InstVoc_HQ.ckpt'):
            torch.hub.download_url_to_file(remote_url_mdxv3, model_folder+'MDX23C-8KFFT-InstVoc_HQ.ckpt')
        if not os.path.isfile(model_folder+'model_2_stem_full_band_8k.yaml'):
            torch.hub.download_url_to_file(remote_url_conf_mdxv3, model_folder+'model_2_stem_full_band_8k.yaml')

        with open(model_folder + 'model_2_stem_full_band_8k.yaml') as f:
            config_mdxv3 = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

        self.model_mdxv3 = TFC_TDF_net(config_mdxv3)
        self.model_mdxv3.load_state_dict(torch.load(model_folder+'MDX23C-8KFFT-InstVoc_HQ.ckpt'))
        self.device = torch.device(device)
        self.model_mdxv3 = self.model_mdxv3.to(device)
        self.model_mdxv3.eval()

        #VitLarge init
        print("Loading VitLarge into memory")
        remote_url_vitlarge = 'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_segm_models_sdr_9.77.ckpt'
        remote_url_vl_conf = 'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_vocals_segm_models.yaml'
        if not os.path.isfile(model_folder+'model_vocals_segm_models_sdr_9.77.ckpt'):
            torch.hub.download_url_to_file(remote_url_vitlarge, model_folder+'model_vocals_segm_models_sdr_9.77.ckpt')
        if not os.path.isfile(model_folder+'config_vocals_segm_models.yaml'):
            torch.hub.download_url_to_file(remote_url_vl_conf, model_folder+'config_vocals_segm_models.yaml')

        with open(model_folder + 'config_vocals_segm_models.yaml') as f:
            config_vl = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

        self.model_vl = Segm_Models_Net(config_vl)
        self.model_vl.load_state_dict(torch.load(model_folder+'model_vocals_segm_models_sdr_9.77.ckpt'))
        self.device = torch.device(device)
        self.model_vl = self.model_vl.to(device)
        self.model_vl.eval()

        # VOCFT init
        if options['use_VOCFT'] is True:
            print("Loading VOCFT into memory")
            self.chunk_size = chunk_size
            self.mdx_models1 = get_models('tdf_extra', load=False, device=device, vocals_model_type=2)
            model_path_onnx1 = model_folder + 'UVR-MDX-NET-Voc_FT.onnx'
            remote_url_onnx1 = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Voc_FT.onnx'
            if not os.path.isfile(model_path_onnx1):
                torch.hub.download_url_to_file(remote_url_onnx1, model_path_onnx1)
            # print('Model path: {}'.format(model_path_onnx1))
            # print('Device: {} Chunk size: {}'.format(device, chunk_size))
            self.infer_session1 = ort.InferenceSession(
                model_path_onnx1,
                providers=providers,
                provider_options=[{"device_id": 0}],
            )

        self.device = device
        pass
        
    @property
    def instruments(self):

        if options['vocals_only'] is False:
            return ['bass', 'drums', 'other', 'vocals']
        else:
            return ['vocals']

    def raise_aicrowd_error(self, msg):
        """ Will be used by the evaluator to provide logs, DO NOT CHANGE """
        raise NameError(msg)
    
    def separate_music_file(
            self,
            mixed_sound_array,
            sample_rate,
            current_file_number=0,
            total_files=0,
    ):
        """
        Implements the sound separation for a single sound file
        Inputs: Outputs from soundfile.read('mixture.wav')
            mixed_sound_array
            sample_rate

        Outputs:
            separated_music_arrays: Dictionary numpy array of each separated instrument
            output_sample_rates: Dictionary of sample rates separated sequence
        """

        # print('Update percent func: {}'.format(update_percent_func))
        
        separated_music_arrays = {}
        output_sample_rates = {}
        # print(mixed_sound_array.T.shape)
        #audio = np.expand_dims(mixed_sound_array.T, axis=0)
        audio = torch.from_numpy(mixed_sound_array.T).type('torch.FloatTensor').to(self.device)

        overlap_demucs = self.overlap_demucs
        overlap_MDX = self.overlap_MDX
        shifts = 0
        overlap = overlap_demucs
        """
        # Get Demucs vocal only
        print('Processing vocals with Demucs_FT...')
        model = self.model_vocals_only
        shifts = 0
        overlap = overlap_demucs
        vocals_demucs = 0.5 * apply_model(model, audio, shifts=shifts, overlap=overlap)[0][3].cpu().numpy() \
                  + 0.5 * -apply_model(model, -audio, shifts=shifts, overlap=overlap)[0][3].cpu().numpy()
        
        model_vocals = model.cpu()
        del model_vocals
        """

        print('Processing vocals with VitLarge model...')
        vocals4, instrum4 = demix_full_vitlarge(audio, self.device, self.model_vl)
        vocals4 = match_array_shapes(vocals4, mixed_sound_array.T)
        # print('Time: {:.0f} sec'.format(time() - start_time))
        # sf.write("/content/drive/MyDrive/output/vocals4.wav", vocals4.T, 44100)
        # sf.write("instrum4.wav", instrum4.T, 44100)

        print('Processing vocals with MDXv3 InstVocHQ model...')
        sources3 = demix_full_mdx23c(mixed_sound_array.T, self.device, self.model_mdxv3)
        vocals3 = match_array_shapes(sources3, mixed_sound_array.T)
        # print('Time: {:.0f} sec'.format(time() - start_time))
        # sf.write("vocals3.wav", sources3.T, 44100)
        
        if options['use_VOCFT'] is True:
            print('Processing vocals with UVR-MDX-VOC-FT...')
            overlap = overlap_MDX
            sources1 = 0.5 * demix_wrapper(
              mixed_sound_array.T,
              self.device,
              self.mdx_models1,
              self.infer_session1,
              overlap=overlap,
              bigshifts=options['BigShifts']//5
          )
            sources1 += 0.5 * -demix_wrapper(
                -mixed_sound_array.T,
                self.device,
                self.mdx_models1,
                self.infer_session1,
                overlap=overlap,
                bigshifts=options['BigShifts']//5
            )
            vocals_mdxb1 = sources1 
            # sf.write("vocals_mdxb1.wav", vocals_mdxb1.T, 44100)
            
        print('Processing vocals: DONE!')
        
        # Vocals Weighted Multiband Ensemble :
        if options['use_VOCFT'] is False:
            weights = np.array([options["weight_InstVoc"], options["weight_VitLarge"]])
            vocals_low = lr_filter((weights[0] * vocals3.T + weights[1] * vocals4.T) / weights.sum(), 10000, 'lowpass') * 1.01055
            vocals_high = lr_filter(vocals3.T, 10000, 'highpass')
            vocals = vocals_low + vocals_high


        if options['use_VOCFT'] is True:
            weights = np.array([options["weight_VOCFT"], options["weight_InstVoc"], options["weight_VitLarge"]])
            vocals_low = lr_filter((weights[0] * vocals_mdxb1.T + weights[1] * vocals3.T + weights[2] * vocals4.T) / weights.sum(), 10000, 'lowpass') * 1.01055
            vocals_high = lr_filter(vocals3.T, 10000, 'highpass')
            vocals = vocals_low + vocals_high
        
        
        # Generate instrumental
        instrum = mixed_sound_array - vocals
        
        if options['vocals_only'] is False:
            print('Starting Demucs processing...')
            audio = np.expand_dims(instrum.T, axis=0)
            audio = torch.from_numpy(audio).type('torch.FloatTensor').to(self.device)

            all_outs = []
            print('Processing with htdemucs_ft...')
            i = 0
            overlap = overlap_demucs
            model = pretrained.get_model('htdemucs_ft')
            model.to(self.device)
            out = 0.5 * apply_model(model, audio, shifts=shifts, overlap=overlap)[0].cpu().numpy() \
                  + 0.5 * -apply_model(model, -audio, shifts=shifts, overlap=overlap)[0].cpu().numpy()
       
            out[0] = self.weights_drums[i] * out[0]
            out[1] = self.weights_bass[i] * out[1]
            out[2] = self.weights_other[i] * out[2]
            out[3] = self.weights_vocals[i] * out[3]
            all_outs.append(out)
            model = model.cpu()
            del model
            gc.collect()
            i = 1
            print('Processing with htdemucs...')
            overlap = overlap_demucs
            model = pretrained.get_model('htdemucs')
            model.to(self.device)
            out = 0.5 * apply_model(model, audio, shifts=shifts, overlap=overlap)[0].cpu().numpy() \
                  + 0.5 * -apply_model(model, -audio, shifts=shifts, overlap=overlap)[0].cpu().numpy()
    
            out[0] = self.weights_drums[i] * out[0]
            out[1] = self.weights_bass[i] * out[1]
            out[2] = self.weights_other[i] * out[2]
            out[3] = self.weights_vocals[i] * out[3]
            all_outs.append(out)
            model = model.cpu()
            del model
            gc.collect()
            i = 2
            print('Processing with htdemucs_6s...')
            overlap = overlap_demucs
            model = pretrained.get_model('htdemucs_6s')
            model.to(self.device)
            out = apply_model(model, audio, shifts=shifts, overlap=overlap)[0].cpu().numpy()
       
            # More stems need to add
            out[2] = out[2] + out[4] + out[5]
            out = out[:4]
            out[0] = self.weights_drums[i] * out[0]
            out[1] = self.weights_bass[i] * out[1]
            out[2] = self.weights_other[i] * out[2]
            out[3] = self.weights_vocals[i] * out[3]
            all_outs.append(out)
            model = model.cpu()
            del model
            gc.collect()
            i = 3
            print('Processing with htdemucs_mmi...')
            model = pretrained.get_model('hdemucs_mmi')
            model.to(self.device)
            out = 0.5 * apply_model(model, audio, shifts=shifts, overlap=overlap)[0].cpu().numpy() \
                  + 0.5 * -apply_model(model, -audio, shifts=shifts, overlap=overlap)[0].cpu().numpy()
       
            out[0] = self.weights_drums[i] * out[0]
            out[1] = self.weights_bass[i] * out[1]
            out[2] = self.weights_other[i] * out[2]
            out[3] = self.weights_vocals[i] * out[3]
            all_outs.append(out)
            model = model.cpu()
            del model
            gc.collect()
            out = np.array(all_outs).sum(axis=0)
            out[0] = out[0] / self.weights_drums.sum()
            out[1] = out[1] / self.weights_bass.sum()
            out[2] = out[2] / self.weights_other.sum()
            out[3] = out[3] / self.weights_vocals.sum()

            # other
            res = mixed_sound_array - vocals - out[0].T - out[1].T
            res = np.clip(res, -1, 1)
            separated_music_arrays['other'] = (2 * res + out[2].T) / 3.0
            output_sample_rates['other'] = sample_rate
    
            # drums
            res = mixed_sound_array - vocals - out[1].T - out[2].T
            res = np.clip(res, -1, 1)
            separated_music_arrays['drums'] = (res + 2 * out[0].T.copy()) / 3.0
            output_sample_rates['drums'] = sample_rate
    
            # bass
            res = mixed_sound_array - vocals - out[0].T - out[2].T
            res = np.clip(res, -1, 1)
            separated_music_arrays['bass'] = (res + 2 * out[1].T) / 3.0
            output_sample_rates['bass'] = sample_rate
    
            bass = separated_music_arrays['bass']
            drums = separated_music_arrays['drums']
            other = separated_music_arrays['other']
    
            separated_music_arrays['other'] = mixed_sound_array - vocals - bass - drums
            separated_music_arrays['drums'] = mixed_sound_array - vocals - bass - other
            separated_music_arrays['bass'] = mixed_sound_array - vocals - drums - other
            
            
        # vocals
        separated_music_arrays['vocals'] = vocals
        output_sample_rates['vocals'] = sample_rate
        
        # instrum
        separated_music_arrays['instrum'] = instrum

        return separated_music_arrays, output_sample_rates


def predict_with_model(options):

    output_format = options['output_format']

    for input_audio in options['input_audio']:
        if not os.path.isfile(input_audio):
            print('Error. No such file: {}. Please check path!'.format(input_audio))
            return

    output_folder = options['output_folder']
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    model = None
    model = EnsembleDemucsMDXMusicSeparationModel(options)

    for i, input_audio in enumerate(options['input_audio']):
        print('Go for: {}'.format(input_audio))
        audio, sr = librosa.load(input_audio, mono=False, sr=44100)
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio], axis=0)
        print("Input audio: {} Sample rate: {}".format(audio.shape, sr))
        result, sample_rates = model.separate_music_file(audio.T, sr, i, len(options['input_audio']))
        for instrum in model.instruments:
            output_name = os.path.splitext(os.path.basename(input_audio))[0] + '_{}.wav'.format(instrum)
            sf.write(output_folder + '/' + output_name, result[instrum], sample_rates[instrum], subtype=output_format)
            print('File created: {}'.format(output_folder + '/' + output_name))

        # instrumental part 1
        # inst = (audio.T - result['vocals']) # * 1.002
        inst = result['instrum']
        output_name = os.path.splitext(os.path.basename(input_audio))[0] + '_{}.wav'.format('instrum')
        sf.write(output_folder + '/' + output_name, inst, sr, subtype=output_format)
        print('File created: {}'.format(output_folder + '/' + output_name))
        
        if options['vocals_only'] is False:
            # instrumental part 2
            inst2 = (result['bass'] + result['drums'] + result['other']) # 1.004
            output_name = os.path.splitext(os.path.basename(input_audio))[0] + '_{}.wav'.format('instrum2')
            sf.write(output_folder + '/' + output_name, inst2, sr, subtype=output_format)
            print('File created: {}'.format(output_folder + '/' + output_name))


# Linkwitz-Riley filter
def lr_filter(audio, cutoff, filter_type, order=6, sr=44100):
    audio = audio.T
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order//2, normal_cutoff, btype=filter_type, analog=False)
    sos = signal.tf2sos(b, a)
    filtered_audio = signal.sosfiltfilt(sos, audio)
    return filtered_audio.T

# SRS
def change_sr(data, up, down):
    data = data.T
    new_data = resample_poly(data, up, down)
    return new_data.T

# Lowpass filter
def lp_filter(cutoff, data, sample_rate):
    b = signal.firwin(1001, cutoff, fs=sample_rate)
    filtered_data = signal.filtfilt(b, [1.0], data)
    return filtered_data

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def match_array_shapes(array_1:np.ndarray, array_2:np.ndarray):
    if array_1.shape[1] > array_2.shape[1]:
        array_1 = array_1[:,:array_2.shape[1]] 
    elif array_1.shape[1] < array_2.shape[1]:
        padding = array_2.shape[1] - array_1.shape[1]
        array_1 = np.pad(array_1, ((0,0), (0,padding)), 'constant', constant_values=0)
    return array_1

if __name__ == '__main__':
    m = argparse.ArgumentParser()
    m.add_argument("--input_audio", "-i", nargs='+', type=str, help="Input audio location. You can provide multiple files at once", required=True)
    m.add_argument("--output_folder", "-r", type=str, help="Output audio folder", required=True)
    m.add_argument("--cpu", action='store_true', help="Choose CPU instead of GPU for processing. Can be very slow.")
    m.add_argument("--overlap_demucs", type=float, help="Overlap of splited audio for light models. Closer to 1.0 - slower", required=False, default=0.1)
    m.add_argument("--overlap_VOCFT", type=float, help="Overlap of splited audio for heavy models. Closer to 1.0 - slower", required=False, default=0.1)
    m.add_argument("--overlap_VitLarge", type=int, help="Overlap of splited audio for heavy models. Closer to 1.0 - slower", required=False, default=1)
    m.add_argument("--overlap_InstVoc", type=int, help="MDXv3 overlap", required=False, default=1)
    m.add_argument("--weight_InstVoc", type=float, help="Weight of MDXv3 model", required=False, default=8)
    m.add_argument("--weight_VOCFT", type=float, help="Weight of VOC-FT model", required=False, default=1)
    m.add_argument("--weight_VitLarge", type=float, help="Weight of VitLarge model", required=False, default=5)
    m.add_argument("--single_onnx", action='store_true', help="Only use single ONNX model for vocals. Can be useful if you have not enough GPU memory.")
    m.add_argument("--large_gpu", action='store_true', help="It will store all models on GPU for faster processing of multiple audio files. Requires 11 and more GB of free GPU memory.")
    m.add_argument("--BigShifts", type=int, help="Managing MDX 'BigShifts' trick value.", required=False, default=7)
    m.add_argument("--vocals_only", type=bool, help="Vocals + instrumental only", required=False, default=False)
    m.add_argument("--use_VOCFT", type=bool, help="use VOCFT in vocal ensemble", required=False, default=False)
    m.add_argument("--output_format", "-f", type=str, help="Output audio format", default="FLOAT")
    m.add_argument("--quiet", "-q", action="store_true", help="Do not display most information and errors")

    options = m.parse_args().__dict__

    stdout = os.sys.stdout
    if options['quiet']:
        stdout = None

        ort.set_default_logger_severity(4)

    with redirect_stdout(stdout):
        print('GPU use: {}'.format(gpu_use))

        print("Options: ")

        print(f'BigShifts: {options["BigShifts"]}\n')

        print(f'weight_InstVoc: {options["weight_InstVoc"]}')
        print(f'weight_VitLarge: {options["weight_VitLarge"]}\n')

        print(f'overlap_InstVoc: {options["overlap_InstVoc"]}')
        print(f'overlap_VitLarge: {options["overlap_VitLarge"]}\n')
    
        print(f'use_VOCFT: {options["use_VOCFT"]}')
        if options["use_VOCFT"] is True:
            print(f'overlap_VOCFT: {options["overlap_VOCFT"]}')
            print(f'weight_VOCFT: {options["weight_VOCFT"]}\n')

        print(f'vocals_only: {options["vocals_only"]}')
    
        if options["vocals_only"] is False:
            print(f'overlap_demucs: {options["overlap_demucs"]}\n')

        print(f'output_format: {options["output_format"]}\n')

        start_time = time()
        predict_with_model(options)

        print('Time: {:.0f} sec'.format(time() - start_time))
