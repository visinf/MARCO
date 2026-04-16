
Our setup follows [Jamais Vu](https://github.com/VICO-UoE/JamaisVu), [GeoAware-SC](https://github.com/Junyi42/GeoAware-SC), [Pose-For-Everything](https://github.com/luminxu/Pose-for-Everything).

Create a directory `data/` to store all datasets. After preparation, the expected structure is:

```text
data/
├── SPair-71k/
│   ├── JPEGImages/{category}/*.jpg
│   ├── PairAnnotation/{split}/*.json
│   ├── ImageAnnotation/{category}/*.json
│   ├── Segmentation/{category}/*.png
│   └── Layout/large/{trn,test}.txt
├── SPair-U/
│   ├── JPEGImages/{category}/*.jpg
│   ├── PairAnnotation/{split}/*.json
│   ├── ImageAnnotation/{category}/*.json
│   └── Layout/large/{trn,test}.txt
├── pf-pascal/
│   ├── PF-dataset-PASCAL/
│   ├── trn_pairs.csv
│   ├── val_pairs.csv
│   └── test_pairs.csv
├── ap-10k/
│   ├── JPEGImages/{family}/{species}/*.jpg
│   ├── ImageAnnotation/{family}/{species}/*.json
│   └── PairAnnotation/{split}/*.json
└── mp100_all/
    ├── mp100/
    │   ├── alpaca_face/*.jpg
    │   ├── amur_tiger_body/*.jpg
    │   ├── bed/*.jpg
    │   ├── chair/*.jpg
    │   ├── human_face/*.jpg
    │   ├── macaque_body/*.jpg
    │   ├── short_sleeved_shirt/*.jpg
    │   ├── sofa/*.jpg
    │   ├── trousers/*.jpg
    │   ├── zebra_body/*.jpg
    │   └── ...
    ├── annotations/
    │   ├── mp100_split1_train.json
    │   ├── mp100_split1_val.json
    │   ├── mp100_split1_test.json
    │   ├── ...
    │   ├── mp100_split5_train.json
    │   ├── mp100_split5_val.json
    │   └── mp100_split5_test.json
    └── pairs/
        ├── pairs_clothing.json
        ├── pairs_animal_face.json
        ├── pairs_animal_body_unseen.json
        ├── pairs_human_face.json
        └── pairs_furniture.json

```

## Instructions

Run all commands below from the `MARCO/data/` directory.  
Some datasets require `gdown` for downloads (`pip install gdown`). 

## 🧩 SPair-71k

Download and extract SPair-71k by running:

```bash
mkdir -p SPair-71k && cd SPair-71k
wget https://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz
tar -xzf SPair-71k.tar.gz --strip-components=1
rm -f SPair-71k.tar.gz
cd ..
```

## 🔍 SPair-U
For SPair-U we follow [Jamais-Vu](https://github.com/VICO-UoE/JamaisVu). It reuses the `Layout/` and `JPEGImages/` directories from `SPair-71k` via symbolic links, so if you have not already prepared SPair-71k, make sure it exists first by following the [SPair-71k](#-spair-71k) instructions above. Then, run:

```bash
mkdir -p SPair-U && cd SPair-U
wget https://groups.inf.ed.ac.uk/hbilen-data/data/JamaisVuData/SPair-U.zip
unzip SPair-U.zip
mv datasets/SPair-U/* .

# Reuse Layout and JPEGImages from SPair-71k
ln -sf ../SPair-71k/Layout
ln -sf ../SPair-71k/JPEGImages
rm -rf datasets SPair-U.zip
cd ..
```

## 🦓 AP-10K

To prepare [AP-10K](https://github.com/AlexTheBad/AP-10K), we follow the preprocessing pipeline from [Geo-Aware-SC](https://github.com/Junyi42/GeoAware-SC). Download, extract, and preprocess the dataset by running:

```bash
mkdir -p ap-10k && cd ap-10k

# Download and extract AP-10K
gdown https://drive.google.com/uc?id=1-FNNGcdtAQRehYYkGY1y4wzFNg4iWNad
unzip ap-10k.zip
mv ap-10k/* .
rm -rf ap-10k ap-10k.zip

# Go to repo root
cd ../..

# Download the official notebook and helper file
wget -O data/ap-10k_is_crowd.txt https://raw.githubusercontent.com/Junyi42/GeoAware-SC/master/data/ap-10k_is_crowd.txt
python scripts/preprocess_ap10k.py

# Cleanup and return to data/
rm data/ap-10k_is_crowd.txt
cd data
```


## 🖼️ PF-PASCAL

Download and extract PF-PASCAL by running:

```bash
mkdir -p pf-pascal && cd pf-pascal
wget https://www.di.ens.fr/willow/research/proposalflow/dataset/PF-dataset-PASCAL.zip
unzip PF-dataset-PASCAL.zip

wget https://www.robots.ox.ac.uk/~xinghui/sd4match/pf-pascal_image_pairs.zip
unzip pf-pascal_image_pairs.zip
mv pf-pascal_image_pairs/* .

rm -rf PF-dataset-PASCAL.zip pf-pascal_image_pairs.zip pf-pascal_image_pairs
cd ..
```


## 📊 MP-100

Our MP-100-based evaluation is built on top of the original [Pose-for-Everything](https://github.com/luminxu/Pose-for-Everything/blob/main/mp100/README.md) dataset. In this work, we **repurpose MP-100 as an evaluation benchmark for semantic correspondence**.
Please refer to the original MP-100 repository for the official dataset description, source-dataset attribution, and download instructions.

Below, we provide detailed guidance for preparing the subset of MP-100 categories used in our benchmark.

Start with:

```bash
mkdir -p mp100_all && cd mp100_all
```

### 1. Annotations

The official MP-100 annotation files can be downloaded from the official Google Drive folder:

```bash
mkdir -p annotations && cd annotations
gdown --folder "https://drive.google.com/drive/folders/1pzC5uEgi4AW9RO9_T1J-0xSKF12mdj1_"
mv mp100/*.json .
rmdir mp100
cd ..
```
Make sure that `annotations/` is located directly under `mp100_all/`.

### 2. Pair Definitions
Download the pair-definition files:

```bash
mkdir -p pairs && cd pairs
wget https://github.com/visinf/visinf.github.io/raw/main/MARCO/annotations/mp100_pairs.zip
unzip mp100_pairs.zip 
mv mp100_pairs/* .
rm -r mp100_pairs.zip mp100_pairs
cd ..
```
Make sure that `pairs/` is located directly under `mp100_all/`.  
This will download the following files into `pairs/`: `pairs_animal_body_unseen.json`, `pairs_animal_face.json`, `pairs_clothing.json`, `pairs_furniture.json`, and `pairs_human_face.json`. Each file contains, for its corresponding macro-domain, the paths of the image pairs used for evaluation.

### 3. Download Image Data
Create the image directory:
```bash
mkdir -p mp100 && cd mp100
```

#### 🏠 Home furniture

The home-furniture categories in our MP-100 benchmark come from [Keypoint-5](https://github.com/jiajunwu/3dinn):   
`sofa`, `table`, `bed`, `swivelchair`

Download and reorganize the dataset with:

```bash
wget http://3dinterpreter.csail.mit.edu/data/keypoint-5.zip
unzip keypoint-5.zip
rm keypoint-5.zip
for d in table sofa bed swivelchair; do
  mv "$d/images"/*.jpg "$d"/
  rmdir "$d/images"
done
rm -rf readme chair
```

This will result in 1729 images in `table`, 2000 in `sofa`, 1270 in `swivelchair` and 1480 in `bed`.

#### 👗 Apparel item

The apparel categories in our MP-100 benchmark come from [DeepFashion2](https://github.com/switchablenorms/DeepFashion2):

`short_sleeved_outwear`, `short_sleeved_shirt`, `skirt`, `short_sleeved_dress`, `vest_dress`, `long_sleeved_dress`, `long_sleeved_outwear`, `long_sleeved_shirt`, `sling`, `sling_dress`, `trousers`, `vest`

Download and preprocess the dataset with:

```bash
gdown "https://drive.google.com/uc?id=1lQZOIkO-9L0QJuk_w1K8-tRuyno-KvLK"

# Unzip the archive
# Replace YOUR_PASSWORD with the password you received
unzip -P "YOUR_PASSWORD" train.zip

bash ../../../scripts/preprocess_clothing_for_mp100.sh

rm -rf train train.zip
```

To obtain the unzip password, you must complete the official request form: [request form](https://docs.google.com/forms/d/e/1FAIpQLSeIoGaFfCQILrtIZPykkr8q_h9qQ5BoTYbjvf95aXbid0v2Bw/viewform),

### 🐼 Animal face

The animal-face categories in our MP-100 benchmark are:

`alpaca_face`, `californiansealion_face`, `chipmunk_face`, `ferret_face`, `gibbons_face`, `guanaco_face`, `proboscismonkey_face`, `arcticwolf_face`, `camel_face`, `commonwarthog_face`, `gentoopenguin_face`, `greyseal_face`, `klipspringer_face`, `fennecfox_face`, `blackbuck_face`, `capebuffalo_face`, `dassie_face`, `gerbil_face`, `grizzlybear_face`, `olivebaboon_face`, `quokka_face`, `bonobo_face`, `capybara_face`, `fallowdeer_face`, `onager_face`, `pademelon_face`

These categories come from AnimalWeb. This requires an [OpenXLab account and API credentials](https://sso.openxlab.org.cn/login) with an Access Key and Secret Key.

```bash
pip install -U openxlab
openxlab login
openxlab dataset get --dataset-repo OpenDataLab/AnimalWeb --target-path .

unrar x OpenDataLab___AnimalWeb/raw/animal_dataset_v1_c.rar OpenDataLab___AnimalWeb/

bash ../../../scripts/preprocess_animal_face_for_mp100.sh

rm -rf OpenDataLab___AnimalWeb
```
As reference examples, `ferret_face` contains 239 `.jpg` images, `greyseal_face` 180 `.jpg` images and `grizzlybear_face` 159 `.jpg` images.

#### 🧑 Human face

The human-face category in our MP-100 benchmark is:

`human_face`

This category comes from 300W and also requires an [OpenXLab account and API credentials](https://sso.openxlab.org.cn/login) with an Access Key and Secret Key.

```bash
pip install -U openxlab
openxlab login
openxlab dataset get --dataset-repo OpenDataLab/300w --target-path .

# Unpack the nested archives
cd OpenDataLab___300w/raw
mv 300w.tar.gz.00 300w.tar.gz
tar -xzf 300w.tar.gz

cd 300w
mv 300w.tar.00 300w.tar
tar -xf 300w.tar

cd ../../..
mkdir -p human_face
mv OpenDataLab___300w/raw/300w/300w/images/helen human_face/

rm -rf OpenDataLab___300w
```

After extraction, the `human_face/helen/trainset` directory should contain 2000 `.jpg` images.

#### 🐘 Animal body
The animal-body categories in our MP-100 benchmark are:
  `macaque_body`, `locust_body`, `fly_body`, `antelope_body`, `cheetah_body`, `fox_body`, `leopard_body`, `panther_body`, `rat_body`, `squirrel_body`, `beaver_body`, `deer_body`, `giraffe_body`, `lion_body`, `pig_body`, `rhino_body`, `weasel_body`, `bison_body`, `elephant_body`, `gorilla_body`, `otter_body`, `polar_bear_body`, `skunk_body`, `wolf_body`, `hippo_body`, `bobcat_body`, `raccoon_body`, `hamster_body`, `panda_body`, `rabbit_body`, `spider_monkey_body`, `zebra_body`

The categories are sourced as follows:

- `fly_body` and `locust_body` from [DeepPoseKit-Data](https://github.com/jgraving/DeepPoseKit-Data)
- `macaque_body` from [MacaquePose](https://www.pri.kyoto-u.ac.jp/datasets/macaquepose/index.html)
- all remaining categories from [AP-10K](https://github.com/AlexTheBad/AP-10K)

The `fly_body` and `locust_body` categories come from [DeepPoseKit-Data](https://github.com/jgraving/DeepPoseKit-Data). Download the repository and extract images from the HDF5 files by running:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/jgraving/DeepPoseKit-Data.git
cd DeepPoseKit-Data
git sparse-checkout set datasets/fly datasets/locust
cd ..

mkdir -p fly_body locust_body

pip install h5py pillow
python ../../../scripts/preprocess_fly_locust_for_mp100.py

rm -rf DeepPoseKit-Data
```

This step should produce 700 `.jpg` images in `locust_body` and 1500 `.jpg` images in `fly_body`. 


The `macaque_body` category comes from [MacaquePose](https://www.pri.kyoto-u.ac.jp/datasets/macaquepose/index.html). To prepare it, run:
```bash
mkdir -p macaque_body
wget -c --tries=0 --timeout=30 --read-timeout=30 \
  "http://pri.ehub.kyoto-u.ac.jp/datasets/macaquepose/download.php" \
  -O macaquepose.zip
unzip macaquepose.zip
mv v1/images/* macaque_body/
rm -rf v1 macaquepose.zip
```

For the remaining categories, we rely on **AP-10K**. Please follow the [🦓 AP-10K](#-ap-10k) instructions above to download and preprocess the dataset, including running `scripts/preprocess_ap10k.py`. After confirming that the processed dataset is available at `./data/ap-10k`, run:

```bash
bash ../../../scripts/process_ap10k_for_mp100.sh ../../ap-10k/JPEGImages
```

This will create symbolic links for the AP-10K categories used in the MP-100 benchmark.

