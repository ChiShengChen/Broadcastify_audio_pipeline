mkdir converted_audio

for file in /media/meow/Elements/ems_call/data/data_2024all/*.mp3; do
    ffmpeg -y -i "$file" -acodec pcm_s16le -ar 44100 "converted_audio/$(basename "$file" .mp3).wav"
done
