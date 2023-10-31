import uuid
import requests, base64

URL = "http://66.23.193.2:48091"

def send_api(timestep = 30):
    audio_string = get_base64_from_file('1.mp3')

    uu_id = str(uuid.uuid4())
    print('Current UUID is ', uu_id)

    payload = {
        'uu_id': uu_id,
        'audio_string': audio_string,
        'time_step': 30
    }
    response = requests.post(url=f'{URL}/v1/upload', json=payload)

    print(response.status_code)
    if response.status_code == 200:
        result = response.json()

        print(result)


def get_base64_from_file(file_name):
    with open(file_name, "rb") as f:
        image_string = base64.b64encode(f.read()).decode('utf-8')

    return image_string

def _main():
    # audio_string = get_base64_from_file(file_name='11.mp3')
    send_api(timestep=30)
        

if __name__ == "__main__":
    # uvicorn.run(app=app, host="0.0.0.0", port=7000)
    _main()