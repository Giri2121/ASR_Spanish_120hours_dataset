'''
import wave

#audio signal parameters
    1) no of channels ; either stereo [feeling like 2 sounds from left and right of earphone] or mono
    2) sample width; no of bytes for each sample
    3) frame_rate;
            also known as sample rate or sample frequency
            the number of samples recorded each second
            44100 hz is the standard sampling rate for the CD quality; which means 44100 sample values each second
    4) no_of_frames;
            total no of frames
    5) values_of_a_frame:

y la obra quedó terminada pronto a satisfacción del que debía disfrutarla
yo fruncí el entrecejo
se adelantan no se conceden
'''

import wave

def wave1(sample_file):
    print("No of channels", sample_file.getnchannels())
    print("Sample Width", sample_file.getsampwidth())
    print("Frame Rate", sample_file.getframerate())
    print("No of frames", sample_file.getnframes())
    print("Parameters", sample_file.getparams())
    #time of the audio
    print("Time of the audio", (sample_file.getnframes()/sample_file.getframerate()))


if __name__ == '__main__':
    sample_file = wave.open(r"C:\Users\vikassaigiridhar\Music\spanish_translation\New folder\asr-spanish-v1-carlfm01\asr-spanish-v1-carlfm01\1.wav","rb")
    wave1(sample_file)
    sample_file.close()