## Project Structure

```
├── processed
│   ├── au_visualization.png
│   ├── aus.csv
│   └── images
│       ├── arguing.jpg
│       ├── back-off.jpg
|       └── ...
└── scripts
│    └── main.py
│
|       
└── dataset
    │── images
    │── annotations.csv
    └── attribution.csv
```

### How to run 

1. All logic is defined in ```scripts/main.py``` with helper functions. 
2. Only dependency needed is ```dataset/annotations.csv```, they are not hardcoded. 
3. For pkg dependencies, conda pkg manager is used because py-feat had some problems on arm64 osx.  
4. From root of project, run ```conda env create -f scripts/environment.yaml --prefix scripts/.env```.
5. Activate it: ```conda activate scripts/.env```.
6. From the root folder run: ```python3 scripts/main.py```.
7. First time it will take a lot of time to install models from the hugging-face.

8. Sometimes conda is having problem installing pip dependencies (py-feat, opencv-python)
    if that's the case, please recreate the environment from Step 4. (or install via pip directly)
