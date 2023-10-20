import sys
from api_communication import *

#upload
#filename = "/path/to/foo.wav"
filename = sys.argv[1]

audio_url = upload(filename) 
save_transcript(audio_url, filename)



