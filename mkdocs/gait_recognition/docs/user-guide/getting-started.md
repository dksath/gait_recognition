# User Guide
This is a user guide to install the entire project.

## Getting Started

### OpenGait 

In [github/gait_training](https://github.com/KLASS-gait-recognition/gait_training), we added an [OpenGait](https://github.com/KLASS-gait-recognition/gait_training/tree/main/OpenGait-for-local) folder that is pulled from another repo. We trained our models in this repo. For the latest updated code, please visit and clone the official [OpenGait](https://github.com/ShiqiYu/OpenGait) repo instead.

#### Installation

1. clone this repo.
    ```
    git clone https://github.com/KLASS-gait-recognition/gait_training
    ```

2. Install dependencies:
    - pytorch >= 1.6
    - torchvision
    - pyyaml
    - tensorboard
    - opencv-python
    - tqdm
    - py7zr
    
    _______
    Install dependencies by [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html):
    ```
    conda install tqdm pyyaml tensorboard opencv py7zr
    conda install pytorch==1.6.0 torchvision -c pytorch
    ```    
    Or, Install dependencies by pip:
    
    ```
    pip install tqdm pyyaml tensorboard opencv-python py7zr
    pip install torch==1.6.0 torchvision==0.7.0
    ```

    Nota bene: due to certain conflicts in dependencies, installing an older version of `setuptools` might be required.

    ```
    pip install setuptools==59.5.0
    ```

### React
To run the prototype install node.js server.

#### Installation

1. clone this repo.
    ```
    git clone https://github.com/KLASS-gait-recognition/GaitSearch
    ```

2. Install node.js and npm

    **You’ll need to have Node 14.0.0 or later version on your local development machine** (but it’s not required on the server). We recommend using the latest LTS version. You can use [nvm](https://github.com/creationix/nvm#installation) (macOS/Linux) or [nvm-windows](https://github.com/coreybutler/nvm-windows#node-version-manager-nvm-for-windows) to switch Node versions between different projects. For more information visit the official website of [node.js](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)

3. Install dependencies:
       - @chakra-ui/react >= 2.2.3,
       - @emotion/react >= 11.9.3,
       - @emotion/styled >= 11.9.3,
       - @material-ui/core >= 4.12.4,
       - @mui/material > = 5.9.0,
       - antd >= 4.21.6,
       - axios >= 0.27.2,
       - firebase >= 9.9.0,
       - framer-motion >= 6.4.3,
       - react >= 18.2.0,
       - react-awesome-button >= 6.5.1,
       - react-bootstrap-validation >= 0.1.11,
       - react-dom >= 18.2.0,
       - react-player >= 2.10.1,
       - react-router-dom >= 6.3.0,
       - react-scripts >= 2.1.3,
       - react-webcam >= 7.0.1,
       - reactstrap >= 9.1.1,
       - styled-components >= 5.3.5,
       - web-vitals >= 2.1.4
    _______
    Install dependencies by [npm](https://docs.npmjs.com/cli/v8/commands/npm-install):
    
    ```powershell
    npm install [<package-spec> ...]

    aliases: add, i, in, ins, inst, insta, instal, isnt, isnta, isntal, isntall
    ```

    Or, to automatically install based on `package.json`,

    ```
    npm install
    ```

### FastAPI

To run the model in the prototype, install FastAPI and uvicorn server. More information at [FastAPI](https://fastapi.tiangolo.com/) official website. Install OpenCV and MMDetection to run the models in the background.

#### Installation

1. Install dependencies:
       - FastAPI
       - Uvicorn
       - OpenCV
    
    _______
    Install dependencies by [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html):
    ```
    conda install -c conda-forge fastapi uvicorn

    ```    
    Or, Install dependencies by pip:
    
    ```
    pip install fastapi "uvicorn[standard]"

    ```

### MMdetection

2. Install MMdetection

    We recommend that users follow our best practices to install MMDetection. However, the whole process is highly customizable. See [MMdetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) under Customize Installation section for more information.

#### Installation

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmcv-full
```

**Step 1.** Install MMDetection.

Case a: If you develop and run mmdet directly, install it from source:

```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

Case B: If you use mmdet as a dependency or third-party package, install it with pip:

```shell
pip install mmdet
```
This is the method we used.

#### Verify the installation

To verify whether MMDetection is installed correctly, we provide some sample codes to run an inference demo.

**Step 1.** We need to download config and checkpoint files.

```shell
mim download mmdet --config yolov3_mobilenetv2_320_300e_coco --dest .
```

The downloading will take several seconds or more, depending on your network environment. When it is done, you will find two files `yolov3_mobilenetv2_320_300e_coco.py` and `yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth` in your current folder.

**Step 2.** Verify the inference demo.

Option (a). If you install mmdetection from source, just run the following command.

```shell
python demo/image_demo.py demo/demo.jpg yolov3_mobilenetv2_320_300e_coco.py yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth --device cpu --out-file result.jpg
```

You will see a new image `result.jpg` on your current folder, where bounding boxes are plotted on cars, benches, etc.

Option (b). If you install mmdetection with pip, open you python interpreter and copy&paste the following codes.

```python
from mmdet.apis import init_detector, inference_detector

config_file = 'yolov3_mobilenetv2_320_300e_coco.py'
checkpoint_file = 'yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
inference_detector(model, 'demo/cat.jpg')
```

Since we used pip to install mmdetection, we continued to use option (b). After doing so, you will see a list of arrays printed, indicating the detected bounding boxes.



**Step 3 [Optional].** Download the SCNet model that is used for silhouette extraction and bounding box detection in the prototype. In normal circumstances, this will be **automatically downloaded on first run**.

```shell
mim download mmdet --config scnet_r50_fpn_1x_coco --dest .
```
The downloading will take several seconds or more, depending on your network environment. When it is done, you will find two files `scnet_r50_fpn_1x_coco.py` and `scnet_r50_fpn_1x_coco-c3f09857.pth` in your current folder. Place them in the backend folder inside the cloned prototype [repo](#react) as such `configs/scnet/scnet_r50_fpn_1x_coco.py` & `checkpoints/scnet_r50_fpn_1x_coco-c3f09857.pth`

Alternatively, you are able to obtain the weights to SCNet-R50-FPN-1x directly [here](https://github.com/open-mmlab/mmdetection/tree/master/configs/scnet). Simply download the weights and add them to your created backend folder such as `checkpoints/scnet_r50_fpn_1x_coco-c3f09857.pth`  