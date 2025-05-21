import numpy as np
import soundfile as sf
from spatial_audio_ai import (
    SoundNetworkStreamer, 
    SO_Playback, 
    Spatializer, 
    Scene
)

sound_streamer = SoundNetworkStreamer()

raw_sound1 = sf.read("/home/lugo/Downloads/song.wav")
sound11 = np.sin(0.5*1e-2*np.linspace(0, 44100*25*5, 44100*25*5))
sound12 = np.sin(0.7*1e-2*np.linspace(0, 44100*25*5, 44100*25*5))
sound1 = (sound11 + sound12)*0.4
sound1 = np.tile(np.expand_dims(sound1, axis=1), (1, 2))
raw_sound1 = (sound1, raw_sound1[1])

raw_sound2 = sf.read("/home/lugo/Downloads/song.wav")
sound21 = np.sin(4.8*1e-2*np.linspace(0, 44100*25*5, 44100*25*5))
sound22 = np.sin(4.9*1e-2*np.linspace(0, 44100*25*5, 44100*25*5))
sound2 = (sound21 + sound22)*0.2
sound2 = np.tile(np.expand_dims(sound2, axis=1), (1, 2))
raw_sound2 = (sound2, raw_sound2[1])        

so_a = SO_Playback(raw_sound1[0][:, 0])

spatializer = Spatializer()
scene = Scene(spatializer)
scene.register(so_a)
