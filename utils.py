import os
import emoji
import torch
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False  # Prioritize performance over reproducibility
os.environ["LRU_CACHE_CAPACITY"] = "3"# reduces RAM usage massively with pytorch 1.4 or older

# Memory allocation optimization to reduce CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,roundup_power2_divisions:4"
from scipy.io.wavfile import write
from uberduck_ml_dev.vendor.tfcompat.hparam import HParams
from uberduck_ml_dev.models.tacotron2 import Tacotron2, DEFAULTS
from uberduck_ml_dev.data_loader import prepare_input_sequence
from uberduck_ml_dev.models.torchmoji import TorchMojiInterface
import time
import json
import sys
import numpy as np
import torchaudio
from scipy.signal import lfilter, firwin
import warnings
warnings.filterwarnings("ignore")
import wave
from piper import PiperVoice
from uberduck_ml_dev.text.util import convert_to_arpabet

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_CPU = DEVICE == "cpu"
print(DEVICE)

# Add paths
sys.path.append("hifi-gan")
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE
from models import Generator
from denoiser import Denoiser
def load_hifigan(path, conf_name):
    conf = conf_name
    with open(conf) as f:
        json_config = json.loads(f.read())
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    hifigan = Generator(h).to(torch.device(DEVICE))
    state_dict_g = torch.load(path, map_location=torch.device(DEVICE))
    hifigan.load_state_dict(state_dict_g["generator"])
    hifigan.eval()
    hifigan.remove_weight_norm()
    denoiser = Denoiser(hifigan, mode="zeros")
    return hifigan, h, denoiser


def ARPAconverter(text: str) -> str:
    temp = convert_to_arpabet(text=text)
    # Strings are immutable; you must reassign
    temp = temp.replace("{ ", "{")
    temp = temp.replace(" }", "}")
    temp = temp.replace("{.}", ".")
    temp = temp.replace("{?}", "?")
    temp = temp.replace("{!}", "!")
    temp = temp.replace("} .", "}.")
    temp = temp.replace("} ,", "},")
    temp = temp.replace("} !", "}!")
    temp = temp.replace("} ?", "}?")
    temp = temp.replace("} '", "}'")
    return temp



class T2S:
    def __init__(self, arch="tacotron2", vocoder_arch="hifi-gan", taco_path=None, vocoder_path=None):
        """Initialize and load all models."""
        self.taco_path = taco_path
        self.vocoder_path = vocoder_path
        self.arch = arch
        self.vocoder_arch = vocoder_arch
        self.speaker_count = 0
        self.config = DEFAULTS.values()
        self._load_mel()
        self._load_vocoder()

    def _load_mel(self):
        if self.arch == "tacotron2" or self.arch == "Tacotron Pipeline" or self.arch == "Leagcy Tacotron2":
            """Load all required models."""
            # Load speaker info from Tacotron2 checkpoint
            checkpoint = torch.load(self.taco_path, map_location=DEVICE, weights_only=False)
            if "model" in checkpoint.keys():
                checkpoint = checkpoint["model"]

            if "state_dict" in checkpoint.keys():
                checkpoint = checkpoint["state_dict"]
            
            if "speaker_embedding.weight" in checkpoint:
                self.speaker_count = len(checkpoint["speaker_embedding.weight"])
                self.config.update({
                    "has_speaker_embedding": True,
                    "n_speakers": self.speaker_count,
                    "ignore_layers": ["null"],
                })
                print(f"{self.speaker_count} speakers found in model")
            else:
                print("No speaker embedding found in model, defaulting to single speaker")
                self.speaker_count = 1

            # Initialize TorchMoji
            if "gst_lin.weight" in checkpoint:
                self.use_torchmoji = True
                self.torchmoji = TorchMojiInterface(
                    "vocabulary.json",
                    "pytorch_model.bin",
                )
                self.compute_gst = lambda texts: self.torchmoji.encode_texts(texts)
                self.config.update({
                    "gst_dim": 2304,
                    "gst_type": "torchmoji",
                    "torchmoji_vocabulary_file": "vocabulary.json",
                    "torchmoji_model_file": "pytorch_model.bin",
                })
            else:
                self.use_torchmoji = False
                
            # Update configuration
            self.config.update({
                "max_decoder_steps": 3000,  # 30 * 100
                "symbol_set": "nvidia_taco2",
                "text_cleaners": ["english_cleaners"],
                "gate_threshold": 0.05,

                #######################################################
                "p_attention_dropout": 0.00,
                "p_decoder_dropout": 0.00,
                "num_mels": 100,
                "n_fft":2048
            })
            
            # Initialize Tacotron2
            hparams = HParams(**self.config)
            self.tacotron = Tacotron2(hparams)
            self.tacotron.from_pretrained(self.taco_path, device=DEVICE)
            
            # Monkey patch inference method
            tacotron = self.tacotron
            
            @torch.no_grad()
            def custom_inference(inputs):
                """Custom inference method for Tacotron2."""
                text, input_lengths, speaker_ids, embedded_gst, *_ = inputs
                
                # Text embedding - use tacotron.embedding instead of self.embedding
                embedded_inputs = tacotron.embedding(text).transpose(1, 2)
                embedded_text = tacotron.encoder.inference(embedded_inputs, input_lengths)
                encoder_outputs = embedded_text
                
                # Speaker embedding
                if tacotron.speaker_embedding:
                    speakers = torch.arange(tacotron.n_speakers, device=DEVICE)
                    embeddings = tacotron.speaker_embedding(speakers.unsqueeze(0))[0]
                    average = torch.mean(embeddings, 0)
                    embedding_offsets = embeddings - average
                    mixed_offset = torch.sum(embedding_offsets * speaker_ids[:, None], 0)
                    embedded_speakers = mixed_offset + average
                    encoder_outputs += tacotron.spkr_lin(embedded_speakers)
                
                # Style embedding (GST)
                if tacotron.gst_lin is not None:
                    assert (
                        embedded_gst is not None
                    ), f"embedded_gst is None but gst_type was set to {tacotron.gst_type}"
                    gst_embedding = tacotron.gst_lin(embedded_gst)
                    style_weight = 1.0 + 0.7 * embedded_gst.mean()
                    encoder_outputs += style_weight * gst_embedding
                    
                # Decoder
                memory_lengths = input_lengths
                mel_outputs, gate_outputs, alignments, mel_lengths = tacotron.decoder.inference(
                    encoder_outputs, memory_lengths
                )
                mel_outputs_postnet = tacotron.postnet(mel_outputs)
                mel_outputs_postnet = mel_outputs + mel_outputs_postnet * 1.7
                
                return tacotron.parse_output(
                    [mel_outputs, mel_outputs_postnet, gate_outputs, alignments, mel_lengths]
                )
            
            # Patch the method
            self.tacotron.inference = custom_inference
            
        elif self.arch == "Piper":
            self.piper = PiperVoice.load(self.taco_path, use_cuda=torch.cuda.is_available())
        else:
            raise ValueError(f"Unsupported architecture: {self.arch}")

    def _load_vocoder(self):
        if self.arch != "Piper":
            if self.vocoder_arch == "hifi-gan":
                self.hifigan, self.h, self.denoiser = load_hifigan(self.vocoder_path, os.path.join(os.path.dirname(self.vocoder_path), "config.json"))
                self.hifigan_sr, self.h2, self.denoiser_sr = load_hifigan("Superres_Twilight_33000", "config_32k.json")
            else:
                raise ValueError(f"Unsupported vocoder architecture: {self.vocoder_arch}")
        
    def synthesize(self, text, speaker_id, torchmoji_text=None, superress=4, maxdecode=3000, gate_tresh=.25, arpaconv=True, skip_sr=False):
        """End-to-end synthesis from text to audio."""

        if self.arch == "tacotron2" or self.arch == "Tacotron Pipeline" or self.arch == "Leagcy Tacotron2":
            if arpaconv:
                text = ARPAconverter(text=text)
                print("Converted Text:", text)
            
            # Compute style embedding
            if self.use_torchmoji:
                with torch.inference_mode():
                    embedding = self.compute_gst([torchmoji_text])
                    embedding = torch.FloatTensor(embedding)
                    emojis = self.torchmoji.enc2emojis(embedding)[0]
            else:
                embedding = torch.zeros(1, 1, 200)
                emojis = []

            if not USE_CPU:
                embedding = embedding.to(DEVICE)
            
            # Prepare speaker embedding
            speaker_embedding = [0] * self.speaker_count
            speaker_embedding[speaker_id] = 1
            speaker_embedding = torch.FloatTensor(speaker_embedding)
            if not USE_CPU:
                speaker_embedding = speaker_embedding.to(DEVICE)
            
            # Prepare input sequence
            text_padded, input_lengths = prepare_input_sequence(
                [text], cpu_run=USE_CPU, arpabet=1.0, symbol_set="nvidia_taco2"
            )
            if not USE_CPU:
                text_padded = text_padded.to(DEVICE)
                input_lengths = input_lengths.to(DEVICE)
            
            # Run Tacotron2 inference
            input_ = [text_padded, input_lengths, speaker_embedding, embedding]

            with torch.inference_mode(), torch.no_grad():
                input = self.tacotron.inference(input_) # shape: [batch, n_mel, time]
                output = input[1][:1]
            def vocode_mel_spectrogram(mel_postnet):
                """Vocode mel spectrogram using HiFi-GAN with denoising."""
                with torch.inference_mode():
                    # HiFi-GAN forward
                    print("Vocode")
                    audio = self.hifigan(mel_postnet)  # [1, T]
                    audio = audio.squeeze()
                    audio *= MAX_WAV_VALUE

            
                    # Denoise (still torch)
                    print("Denoise")
                    audio_denoised = self.denoiser(audio.view(1, -1), strength=50)[:, 0]

                    # Convert to numpy
                    print("Convert numpy")
                    audio_denoised = audio_denoised.cpu().detach().numpy().reshape(-1)

                return audio_denoised.astype(np.float32)
            def resample_audio(audio, original_sr, target_sr):
                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu()
                else:
                    audio = torch.from_numpy(audio).float()

                if len(audio.shape) == 1:
                    audio = audio.unsqueeze(0)

                resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
                audio_resampled = resampler(audio)
                return audio_resampled.squeeze(0).numpy()

            def apply_super_resolution(base_audio):
                print("SR")
                """Apply HiFi-GAN super-resolution with high-pass filtering."""
                wave = base_audio.astype(np.float32) / MAX_WAV_VALUE
                wave = torch.FloatTensor(wave).to(DEVICE)
                mel = mel_spectrogram(
                    wave.unsqueeze(0),
                    self.h2.n_fft,
                    self.h2.num_mels,
                    self.h2.sampling_rate,
                    self.h2.hop_size,
                    self.h2.win_size,
                    self.h2.fmin,
                    self.h2.fmax,
                )

                sr_hat = self.hifigan_sr(mel).squeeze() * MAX_WAV_VALUE
                sr_audio = sr_hat.detach().cpu().numpy().reshape(-1)
                sr_audio = sr_audio.astype(np.float32)
                
                # Apply high-pass filter
                hp_b = firwin(401, cutoff=10500, fs=self.h2.sampling_rate, pass_zero=False)
                sr_audio = lfilter(hp_b, 1.0, sr_audio) * 1.2
                high_freqs = float(superress) * sr_audio
                return high_freqs.astype(np.float32)

            def merge_audio(original, superres, normalize=False):
                """Merge base and super-resolution audio."""
                min_len = min(len(original), len(superres))
                original = original[:min_len].astype(np.float32)
                superres = superres[:min_len].astype(np.float32)

                merged = original + superres

                if normalize:
                    peak = np.max(np.abs(merged))
                    if peak > 0:
                        merged *= 0.98 / peak

                merged = np.clip(merged, -1, 1)
                final_audio_int16 = (merged * 32767).astype(np.int16)
                return final_audio_int16
                
            # Generate audio with HiFi-GAN
            audio_denoised = vocode_mel_spectrogram(output)

            if skip_sr:
                # Skip super-resolution, just use base audio
                audio_final = audio_denoised
                sample_rate = self.h.sampling_rate
            else:
                # Apply super-resolution
                sample_rate = self.h2.sampling_rate
                audio_denoised = resample_audio(audio_denoised, self.h.sampling_rate, self.h2.sampling_rate)
                audio2_denoised = apply_super_resolution(audio_denoised)
                audio_final = merge_audio(audio_denoised, audio2_denoised, normalize=True)

            audio_pth = os.path.join("generated_audio", f"{int(time.time() * 1000)}_gen.wav")
            write(audio_pth, sample_rate, audio_final)
            
            if self.use_torchmoji:
                print(emoji.emojize(f"Top emotions detected by Torchmoji: {' '.join(emojis)}", language="alias"))
                
            return audio_pth
            
        elif self.arch == "Piper":
            audio_pth = os.path.join("generated_audio", f"{int(time.time() * 1000)}_gen.wav")
            # Create directory if it doesn't exist
            # Synthesize with Piper
            with wave.open(audio_pth, 'wb') as wav_file:
                self.piper.synthesize_wav(text=text, wav_file=wav_file)
            
            return audio_pth

        else:
            raise ValueError(f"Unsupported architecture: {self.arch}")