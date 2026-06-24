echo "Downloading SMPLX model"

cd datasets
# gdown 1FefkJVeoI2ycwgRXydBCQEWuXTuDg5rW

gdown --id 1FefkJVeoI2ycwgRXydBCQEWuXTuDg5rW -O hub.zip

file hub.zip

python -c "import zipfile; zipfile.ZipFile('hub.zip').extractall('hub')"
# python -c "import shutil; shutil.make_archive('hub_contents', 'zip', root_dir='hub', base_dir='.')"


echo -e "Cleaning\n"
rm hub.zip

echo "Downloading done!" 
