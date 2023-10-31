import librosa, os, base64, requests, time
import numpy as np

URL = 'http://64.247.206.123:15300'

audio_file = '1.mp3'
def transcrbie_id(uu_id, id):
    payload = {
        'id': id, 
        'uu_id': uu_id
    }

    response = requests.post(f"{URL}/v1/transcribe", json=payload)
    print(id, response.status_code)

    if response.status_code == 200:
        r = response.json()
        cur_result = r['Part_{id}']
        

    return cur_result

def get_transcribed_text(uu_id, file_name, timestep):
    legnth = librosa.get_duration(filename=file_name)
    max_length = int(legnth) // timestep + 1

    for id in range(max_length):
        text = transcrbie_id(uu_id, id)
        time.sleep(1)
        print(text)

    return

def _main():
    get_transcribed_text(uu_id = 'd5adf005-7113-4dfe-9bc8-89af5fe9acf3', file_name = '1.mp3', timestep = 30)

if __name__ == "__main__":
    _main()