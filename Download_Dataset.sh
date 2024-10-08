sudo pip install gdown
cd data

mkdir -p Split_Dataset/Data
mkdir -p Split_Dataset/Groundtruth

cd Split_Dataset/Data/
gdown  https://drive.google.com/uc?id=1IgxzmajC09aUTz_awABPD_iEJ2s1VR0E -O test.npy
gdown  https://drive.google.com/uc?id=1xCoXFl0GTc5VbW7L-7joAL9ViudD0Nbh -O train.npy
gdown https://drive.google.com/uc?id=1De4n5hf6kirLpMlILD2LTFTuwsgpVhKQ -O valid.npy

cd ../Groundtruth
gdown  https://drive.google.com/uc?id=1y9T41Faf9Oq2XOtWMD1U9fZe9OHLgLjv -O test.tsv
gdown  https://drive.google.com/uc?id=1R1i74XWzILnlozwCfYItlequKIhMnHmB -O train.tsv
gdown https://drive.google.com/uc?id=1ZupxAdTOWxmKPWlD5FOwEKbkdavt5Zxk -O valid.tsv