echo "Downloading SeG SMPLX dataset"

cd datasets
gdown 1xu6ld0bjzskIS06EAYrLXEDp_TvnV-A-

python -m zipfile -e SeG_SMPLX.zip .
echo -e "Cleaning\n"
rm SeG_SMPLX.zip

echo "Downloading done!"