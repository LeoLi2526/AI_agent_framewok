import pyaudio

p = pyaudio.PyAudio()
print(f"Host API count: {p.get_host_api_count()}")
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
print(f"Device count: {numdevices}")

for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
    else:
        print("Output Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
