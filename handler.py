from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa, os, time, torch, subprocess, base64, uvicorn
import soundfile as sf
from pydub import AudioSegment
from threading import Thread
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def split(uu_id, timestep):
    dis = timestep // 3 if timestep // 3 > 0 else 1
    out_dir = uu_id + "_dir"

    json_data[uu_id] = dict()

    json_data[uu_id]["pathes"], json_data[uu_id]["result"] = [], []

    if os.path.exists(out_dir):
        os.system(f'rm -r {out_dir}')
    os.mkdir(out_dir)

    json_data[uu_id]["length"] = librosa.get_duration(filename=uu_id)
    json_data[uu_id]["max_length"] = int(json_data[uu_id]["length"]) // timestep + 1

    fr = -dis
    to = timestep
    for i in range(json_data[uu_id]["max_length"]):
        ffmpeg_command = f'ffmpeg -i {uu_id} -bsf:v h264_mp4toannexb -loglevel quiet -map 0 -flags -global_header  -ss {fr if fr >  10 else 0} -t {timestep}  {out_dir}/slice_{fr if fr > 0 else 0}_{to}.mp3'
        # ffmpeg_command = f'ffmpeg -i {audio_file} -bsf:v h264_mp4toannexb -c copy -map 0 -flags -global_header  -f segment -segment_time 30 -segment_list adwd.m3u8 -segment_format mp3 {out_dir}/adwd%d.mp3 '

        # sem.acquire()
        subprocess.run(ffmpeg_command, shell=True)
        json_data[uu_id]["pathes"].append(f'{out_dir}/slice_{fr if fr > 0 else 0}_{to}.mp3')
        
        # sem.release()
        time.sleep(0.1)
        fr += timestep
        to = fr + timestep + dis


def load_model():
    if not os.path.exists('whisper-large-v2'):
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
        model.config.forced_decoder_ids = None
        
        model.save_pretrained('whisper-large-v2',safe_serialization=True)
        processor.save_pretrained('whisper-large-v2',safe_serialization=True)

    else:
        processor = WhisperProcessor.from_pretrained("whisper-large-v2")
        model = WhisperForConditionalGeneration.from_pretrained("whisper-large-v2")
        model.config.forced_decoder_ids = None

    model = model.to("cuda")
    return processor, model

def read_audio_ndarray(f_path):
    sound = AudioSegment.from_mp3(f_path)
    sound = sound.set_channels(1)
    sound.export(f_path, format="mp3")
    y, sr= sf.read(file = f_path)
    y = y.T
    y = librosa.resample(y, sr, 16000)

    return y

def transcribe_ndarray(audio_array, processor, model):
    input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE) 
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription[0]

def slice_transcribe(uu_id):
    i = 0

    while True:
        # for f_path in pathes:
        if json_data[uu_id]["max_length"] != 0 and i >= json_data[uu_id]["max_length"] : break
        if len(json_data[uu_id]["pathes"]) < i + 1:
            time.sleep(1)
            continue

        f_path = json_data[uu_id]["pathes"][i]
        
        # sem.acquire()
        audio_array = read_audio_ndarray(f_path)
        part_text = transcribe_ndarray(audio_array, processor, model)
        # part_text = f_path

        json_data[uu_id]["result"].append(part_text)
        print(part_text)
        # sem.release()
        i += 1

    return

# sem = Semaphore()
processor, model = load_model()
json_data = {}
dis = 1

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/hello")
def hello():
    return {'message': 'server is running'}

@app.post('/v1/upload')
async def upload(uu_id: str = Body("", title='UUID'),
                 audio_string: str = Body("", title='Audio String'),
                 time_step: int = Body(30, title='Time step to split')):  #, time_step : int = File(...)):
    
    decoding = base64.b64decode(audio_string)
    with open(uu_id, 'wb') as f:
        f.write(decoding)
        
    t1 = Thread(target = split, args =(uu_id, time_step, )) 
    t2 = Thread(target = slice_transcribe, args=(uu_id, )) 
    t1.start() 
    time.sleep(0.5)
    t2.start() 


    return  {"uu_id": uu_id, "max_length": json_data[uu_id]["length"]}

@app.post("/v1/transcribe")
async def transcribe(id : int = Body(0, title='ID'), 
                     uu_id: str = Body("", title='UUID')):
    
    cur_result = json_data[uu_id]["result"]
    while True:
        if len(cur_result) - 1 < id:
            time.sleep(1)
            continue

        return {'Part_{id}': cur_result[id]}
        

@app.post("/v1/transcribe-mass")
async def transcribe_mass(id : int = Body(0, title='ID'), 
                     uu_id: str = Body("", title='UUID')):
    
    cur_result = json_data[uu_id]["result"]

    print(len(cur_result), id)
    if len(cur_result) > id:
        return {'status':'success', 'text': cur_result[id: ]}
    else:
        return {'status':'failed'}


if __name__ == "__main__":
    uvicorn.run(app=app, host='0.0.0.0', port=7000)
