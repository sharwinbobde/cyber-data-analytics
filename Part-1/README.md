### Instructions to Setup and Run

1. Download the data from [brightspace](https://brightspace.tudelft.nl/d2l/le/content/196960/Home?itemIdentifier=D2L.LE.Content.ContentObject.ModuleCO-1658536) into a new folder `data`.

2. Make sure you have Python version 3 (preferrably > python 3.6) installed.

2. If you are using the Pip package manager:

- Create a a virtual environment using virtualenv: `virtualenv -p python3 env`
- Enter the environment: `source env/bin/activate`
- Install from the requirements file: `pip install -r requirements.txt`

3. If you are using the Conda package manager
   1. Create a virtual environment: `conda create --name myenv`
   2. Enter the environment: `conda activate --stack myenv` 
   3. Install from the requirements file: `conda install --file requirements.txt`

4. Run the jupyter server: `jupyter notebook .` and go to the url in your browser.

5. Enjoy!