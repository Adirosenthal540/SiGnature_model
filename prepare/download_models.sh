echo "Downloading signature base pretrained models"
mkdir -p ckp
cd ckp
gdown 13oKqblpeOcIxwxDEZxJRgftVkJZOYO8l

python -m zipfile -e official_model.zip .
echo -e "Cleaning\n"
rm official_model.zip

echo "Downloading done!"