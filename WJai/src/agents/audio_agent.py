import logging
import os
import base64
import signal
import sys
import time
import pyaudio
import wave
import json
import argparse
import threading
import audioop
from datetime import datetime
import dashscope
from dashscope.audio.qwen_omni import *
try:
    from src.utils.B64PCMPlayer import B64PCMPlayer
except ImportError:
    try:
        from ..utils.B64PCMPlayer import B64PCMPlayer
    except ImportError:
         print("Warning: B64PCMPlayer not found")
        
from dashscope.audio.qwen_omni.omni_realtime import TranscriptionParams


logger = logging.getLogger('dashscope')
logger.setLevel(logging.ERROR) 

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.propagate = False

class TranscriptionCollector:
    def __init__(self):
        self.transcripts = []
        self.full_text = ""
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.current_speaker = "Speaker 1"

    def set_speaker(self, speaker_name):
        self.current_speaker = speaker_name

    def add_transcript(self, text, is_final=True):
        if is_final:
            self.transcripts.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "speaker": self.current_speaker,
                "text": text,
                "status": "final"
            })
            self.full_text += f"[{self.current_speaker}] {text}"
        else:
            pass

    def save_to_json(self, mode, filepath="transcription_result.json"):
        data = {
            "meta": {
                "mode": mode,
                "start_time": self.start_time,
                "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "content": self.transcripts,
            "full_text": self.full_text
        }
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"[System] Result saved to {filepath}")
            print(f"[System] Full Text: {self.full_text}")
        except Exception as e:
            print(f"[Error] Failed to save JSON: {e}")

class MyCallback(OmniRealtimeCallback):
    def __init__(self, collector):
        super().__init__()
        self.collector = collector

    def on_open(self) -> None:
        print('[System] Connection opened.')

    def on_close(self, close_status_code, close_msg) -> None:
        print(f'[System] Connection closed ({close_status_code}): {close_msg}')

    def on_error(self, error) -> None:
        print(f'[System] Connection error: {error}')


    def on_event(self, response: str) -> None:
        try:
            type = response['type']
            
            # Log session creation
            if 'session.created' == type:
                pass 
                
            # Handle final transcription
            if 'conversation.item.input_audio_transcription.completed' == type:
                text = response['transcript']
                print(f"[Final] [{self.collector.current_speaker}] {text}")
                self.collector.add_transcript(text, is_final=True)
            
            #if 'response.text.done' == type or 'response.content_part.done' == type:
                # Sometimes results come in different event types depending on model version
                # Checking all response types might help debugging
                # print(f"[Debug] {type}: {response}")
                #pass

            
            if 'conversation.item.input_audio_transcription.text' == type:
                text = response['stash']
                # Overwrite the current line for real-time effect
                sys.stdout.write(f"[Realtime] [{self.collector.current_speaker}] {text}")
                sys.stdout.flush()

            if 'input_audio_buffer.speech_stopped' == type:
                # Some APIs trigger final recognition after speech stopped
                pass

            if 'input_audio_buffer.speech_started' == type:
                pass # print('[Debug] Speech Start')
            if 'input_audio_buffer.speech_stopped' == type:
                pass # print('[Debug] Speech Stop')
            
        except Exception as e:
            print(f'[Error] Event processing failed: {e}')

def init_dashscope_api_key():
    if 'DASHSCOPE_API_KEY' in os.environ:
        dashscope.api_key = os.environ['DASHSCOPE_API_KEY']
    else:
        # It is recommended to set this in environment variables instead
        dashscope.api_key = "sk-cc64ec82548f4d7885eae4c6ab2a3a8b" 

def run_mic_mode(conversation, collector, device_indices=None):
    pya = pyaudio.PyAudio()
    recording = False
    
    # Default to device 0 if none provided, but treat it as a list
    if not device_indices:
        # Try to find a default device
        try:
            default_device = pya.get_default_input_device_info()
            device_indices = [default_device['index']]
        except:
            print("[Error] No default input device found.")
            return

    current_device_idx_ptr = 0 
    
    stream = None
    resample_state = None
    capture_rate = 16000
    target_rate = 16000

    def open_mic_stream(dev_index):
        nonlocal capture_rate, resample_state
        resample_state = None
        
        # Check if device supports target rate
        supported_rate = target_rate
        try:
            if not pya.is_format_supported(target_rate, input_device=dev_index, input_channels=1, input_format=pyaudio.paInt16):
                dev_info = pya.get_device_info_by_index(dev_index)
                default_rate = int(dev_info.get('defaultSampleRate', target_rate))
                print(f"[System] Device {dev_index} does not support {target_rate}Hz. Switching to {default_rate}Hz.")
                supported_rate = default_rate
            else:
                supported_rate = target_rate
        except Exception as e:
            # Check what rate IS supported by asking device info
            try:
                dev_info = pya.get_device_info_by_index(dev_index)
                supported_rate = int(dev_info.get('defaultSampleRate', target_rate))
                print(f"[Warning] Device check failed: {e}. Trying default rate {supported_rate}.")
            except Exception as e2:
                 print(f"[Warning] Could not determine default rate: {e2}. Sticking to {target_rate}")
                 supported_rate = target_rate
        
        capture_rate = supported_rate
        buffer_frames = int(capture_rate * 0.2)
        
        try:
            s = pya.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=capture_rate,
                            input=True,
                            input_device_index=dev_index,
                            frames_per_buffer=buffer_frames,
                            stream_callback=stream_callback)
            return s
        except OSError as e:
            print(f"[Error] Failed to open device {dev_index}: {e}")
            return None

    def stream_callback(in_data, frame_count, time_info, status):
        nonlocal recording, resample_state
        if recording:
            try:
                process_data = in_data
                
                if capture_rate != target_rate:
                    # audioop.ratecv(fragment, width, nchannels, inrate, outrate, state[, weightA[, weightB]])
                    process_data, resample_state = audioop.ratecv(
                        process_data, 
                        2, # sample_width (16bit = 2 bytes)
                        1, # channels
                        capture_rate, 
                        target_rate, 
                        resample_state
                    )

                rms = audioop.rms(process_data, 2)
                # print(f"[Debug] Level: {rms:5d}  ", end="", flush=True)
                
                audio_b64 = base64.b64encode(process_data).decode('ascii')
                conversation.append_audio(audio_b64)
            except Exception as e:
                print(f"[Error] Processing audio: {e}")
        return (in_data, pyaudio.paContinue)

    # Initial setup
    current_dev_id = device_indices[current_device_idx_ptr]
    print(f"[System] Initializing with Microphone {current_device_idx_ptr+1} (Device ID: {current_dev_id})")
    stream = open_mic_stream(current_dev_id)
    if stream is None:
        return

    print("" + "="*50)
    print("      MULTI-MICROPHONE MODE")
    print(f"      Devices: {device_indices}")
    print("="*50)
    print("Commands:")
    print("  's' + Enter -> Start Recording")
    print("  'q' + Enter -> Stop & Save")
    print(f"  '1'-'{len(device_indices)}' + Enter -> Switch Microphone")
    print("="*50)

    stream.start_stream()

    try:
        while True:
            cmd = input("Command (s/q/1-9): ").strip().lower()
            if cmd == 's':
                if not recording:
                    print("[System] Recording STARTED...")
                    recording = True
                else:
                    print("[System] Already recording.")
            
            elif cmd == 'q':
                print("[System] Stopping...")
                recording = False
                break
            
            elif cmd.isdigit():
                idx = int(cmd) - 1
                if 0 <= idx < len(device_indices):
                    new_dev_id = device_indices[idx]
                    if idx == current_device_idx_ptr and stream.is_active():
                         print(f"[System] Already using Speaker {idx+1} (Device {new_dev_id})")
                         continue
                         
                    print(f"[System] Switching to Speaker {idx+1} (Device {new_dev_id})...")
                    
                    # Stop current stream
                    if stream:
                        stream.stop_stream()
                        stream.close()
                    
                    # Update state
                    current_device_idx_ptr = idx
                    collector.set_speaker(f"Speaker {idx+1}")
                    
                    # Start new stream
                    stream = open_mic_stream(new_dev_id)
                    if stream:
                        stream.start_stream()
                        print(f"[System] Switched to Speaker {idx+1}")
                    else:
                        print(f"[Error] Could not open Device {new_dev_id}. Stopping.")
                        break
                else:
                    print(f"[System] Invalid microphone number. Choose 1-{len(device_indices)}")

            else:
                print("[System] Unknown command.")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if stream and stream.is_active():
                stream.stop_stream()
            if stream:
                stream.close()
        except Exception as e:
            print(f"[Warning] Stream close error: {e}")
            
        pya.terminate()
        conversation.close()
        collector.save_to_json("microphone")

def run_file_mode(conversation, collector, file_path):
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return

    print(f"[System] Processing file: {file_path}")
    
    try:
        with wave.open(file_path, 'rb') as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
                print("[Warning] Audio format should be Mono, 16bit, 16000Hz. Results may be inaccurate.")
            
            chunk_size = 3200
            data = wf.readframes(chunk_size)
            
            while len(data) > 0:
                audio_b64 = base64.b64encode(data).decode('ascii')
                conversation.append_audio(audio_b64)
                # Slight sleep to simulate real-time stream and avoid flooding buffer too fast
                time.sleep(0.01) 
                data = wf.readframes(chunk_size)
                
        print("[System] File sent. Waiting for final results...")
        time.sleep(10) 
        
    except Exception as e:
        print(f"[Error] File processing error: {e}")
    finally:
        collector.save_to_json("file")
        conversation.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Text Transformer')
    parser.add_argument('--mode', type=str, choices=['mic', 'file'], default='mic', help='Operation mode: mic or file')
    parser.add_argument('--file_path', type=str, help='Path to audio file (required for file mode)')
    parser.add_argument('--device_indices', type=int, nargs='+', default=None, help='List of input device indices for microphone mode (e.g. 7 6)')
    
    args = parser.parse_args()
    
    if args.mode == 'file' and not args.file_path:
        print("Error: --file_path is required for file mode.")
        sys.exit(1)

    init_dashscope_api_key()
    
    collector = TranscriptionCollector()
    callback = MyCallback(collector)

    conversation = OmniRealtimeConversation(
        model='qwen3-asr-flash-realtime',
        url='wss://dashscope.aliyuncs.com/api-ws/v1/realtime',
        callback=callback,
    )

    transcription_params = TranscriptionParams(
        language='zh',
        sample_rate=16000,
        input_audio_format="pcm",
        corpus_text="这是一段中文对话"
    )

    try:
        conversation.connect()
        conversation.update_session(
            output_modalities=[MultiModality.TEXT],
            enable_input_audio_transcription=True,
            transcription_params=transcription_params,
        )

        if args.mode == 'mic':
            run_mic_mode(conversation, collector, args.device_indices)
        else:
            run_file_mode(conversation, collector, args.file_path)

    except Exception as e:
        print(f"[Error] Application failed: {e}")
