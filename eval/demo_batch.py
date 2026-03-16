import os
import sys
import asyncio
import argparse
import librosa
import numpy as np
import pandas as pd

from tqdm import tqdm

import sos_client

sos_client.mute = True

def sampler_binary(pdf):
    for token, prob in pdf.items():
        if token == sos_client.SOS_TOKEN.YES.value:
            return "是" if prob > thresh0 else "否"
    return "否"

def sampler_multi(pdf):
    lst_speech = [item[1] for item in pdf.items() if item[0]==sos_client.SOS_TOKEN.SPEECH.value]
    lst_bc = [item[1] for item in pdf.items() if item[0]==sos_client.SOS_TOKEN.BC.value]
    lst_noise = [item[1] for item in pdf.items() if item[0]==sos_client.SOS_TOKEN.NOISE.value]
    prob_speech = lst_speech[0] if lst_speech else None
    prob_bc = lst_bc[0] if lst_bc else None
    prob_noise = lst_noise[0] if lst_noise else None
    if prob_speech is not None and prob_speech > thresh0:
        type = "3"
    elif prob_noise is None:
        type = "2"
    elif prob_bc is None:
        type = "1"
    else:
        type = "2" if prob_bc / (prob_bc + prob_noise) > thresh1 else "1"
    return prob_noise, prob_bc, prob_speech, type

samplers = {
    "binary": sampler_binary,
    "multi": sampler_multi,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="agora_sos_models/finetuned_hf_for_inference_13_2750")
    parser.add_argument('--thresh0', type=float, default=.4)
    parser.add_argument('--thresh1', type=float, default=.5)
    parser.add_argument("--concurrence", type=int, default=4)
    parser.add_argument("--input-audio-dir", type=str, required=True)
    args = parser.parse_args()
    model_path = args.model_path
    thresh0 = args.thresh0
    thresh1 = args.thresh1
    concurrence = args.concurrence
    input_audio_dir = args.input_audio_dir

    model_seq = int(model_path.split('_')[-2])
    model_type = "binary" if model_seq <= 8 else "multi"
    files = os.listdir(input_audio_dir)
    files = [file for file in files if file.endswith(".wav")]
    files.sort(key=lambda x: os.path.getsize(os.path.join(input_audio_dir, x)))
    if not files:
        sys.exit(0)
    file_group_size = 20
    groups = [files[i:i+file_group_size] for i in range(0, len(files), file_group_size)]
    model_outputs = []
    for group in tqdm(groups):
        sr = 16000
        audios = [librosa.load(os.path.join(input_audio_dir, file), sr=sr, mono=True)[0] for file in group]
        audios = [(audio * 32768).clip(-32768, 32767).astype(np.int16) for audio in audios]
        input_audios = [(file, audio) for file, audio in zip(group, audios)]
        output = asyncio.run(sos_client.kick_model(model_path, input_audios, concurrence))
        model_outputs += output

    model_outputs = sorted(model_outputs, key=lambda x: x.identity)
    results = [samplers[model_type](item.probs) for item in model_outputs]
    results_csv = [{"prob_noise":f"{result[0]:.3f}", "prob_bc":f"{result[1]:.3f}", "prob_speech":f"{result[2]:.3f}", "type":result[3], "filename":model_output.identity} for model_output, result in zip(model_outputs, results)]
    assert len(results_csv) == len(files)

    dir, fname = os.path.split(input_audio_dir)
    df = pd.DataFrame(results_csv)
    df.to_csv(os.path.join(dir, f"{fname}_sos.csv"))
    print("done")
