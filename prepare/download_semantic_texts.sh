echo "Downloading semantic directing texts"

cd datasets/BEAT_SMPL/BEAT2
gdown 1EAurQkriGsO6R_6NDwMs6YAGvNQZ9-c7
python -m zipfile -e beat_english_v2.0.0.zip .
echo -e "Cleaning\n"
rm beat_english_v2.0.0.zip

echo "Downloading done!" 