# Nautilus
### Official Python implementation of the [Fed]Nautilus model, both centralized and federated learning versions, proposed in the paper "On Vessel Location Forecasting and the Effect of Federated Learning”, MDM Conference, 2024.


# Installation 
In order to use Nautilus in your project, download all necessary modules in your directory of choice via pip or conda, and install their corresponding dependencies, as the following commands suggest:

```Python
# Using pip/virtualenv
pip install −r requirements.txt

# Using conda
conda install --file requirements.txt
```


# Data Preprocessing
In order to perform data preprocessing on your AIS dataset(s), as defined in the paper, run the following script

```bash
python data-preprocessing.py [-h] --data {brest,piraeus,mt} [--min_dt MIN_DT] [--max_dt MAX_DT] [--min_speed MIN_SPEED] [--max_speed MAX_SPEED] [--min_pts MIN_PTS] [--shiptype] [--njobs NJOBS]
```

To follow the preprocessing workflow up to the specifications defined in the paper, adjust the above command as follows:

```bash
python data-preprocessing.py --data {brest,piraeus,mt} --shiptype --min_dt 10 --max_dt {1800,3600}
```

for the 30 min. and 60 min. variant, respectively.


## Documentation

```bash
  -h, --help            show this help message and exit
  --data                Select Dataset {brest, piraeus, mt}
  --min_dt MIN_DT       Minimum $\Delta t$ threshold (default:10 sec.)
  --max_dt MAX_DT       Maximum $\Delta t$ threshold (default:1800 sec.)
  --min_speed MIN_SPEED
                        Minimum speed threshold (stationaries; default: 1 knot)
  --max_speed MAX_SPEED
                        Maximum speed threshold (outliers; default: 50 knots)
  --min_pts MIN_PTS     Minimum points threshold for constructing a trajectory (default: 20 points)
  --shiptype            Include shiptype
  --njobs NJOBS         #CPUs (default: 200 cores)
```


# Usage (Centralized)
In order to train ```Nautilus```, run the following script:

```bash
python training-rnn-v2-indie-timesplit.py --data {brest_1800,brest_3600,piraeus_1800,piraeus_3600,mt_1800,mt_3600} [--gpuid GPUID] [--njobs NJOBS] [--crs CRS] [--bi] [--dspeed] [--dcourse] [--shiptype] [--bs BS] [--length LENGTH] [--stride STRIDE] [--patience PATIENCE] [--max_dt MAX_DT] [--skip_train]
```

For training ```Nautilus``` up to specifications defined in the paper, adjust the above command as follows:

```bash
python training-rnn-v2-indie-timesplit.py --data {brest_1800,piraeus_1800,mt_1800} --gpuid 0 --bs 1 --njobs 200 --crs {2154,2100,2100} --length 32 --stride 16 --patience 10 --shiptype --dspeed --dcourse --max_dt {1800,3600}
```

for the 30 min. and 60 min. variant, respectively.


## Documentation
```bash
    -h, --help            Show this help message and exit
    --data                Select Dataset 
                          (Options: brest_1800, brest_3600, piraeus_1800, piraeus_3600, mt_1800, mt_3600)
    --gpuid GPUID         GPU ID
    --njobs NJOBS         #CPUs
    --crs CRS             Dataset CRS (default: 3857)
    --bi                  Use Bidirectional LSTM
    --dspeed              Use first order difference of Speed
    --dcourse             Use first order difference of Course
    --shiptype            Use AIS Shiptype
    --bs BS               Batch Size
    --length LENGTH       Rolling Window Length (default: 32)
    --stride STRIDE       Rolling Window Stride (default: 16)
    --patience PATIENCE   Patience (#Epochs) for Early Stopping (default: 10)
    --max_dt MAX_DT       Maximum $\Delta t$ threshold (default:1800 sec.)
    --skip_train          Skip training; Evaluate best model @ Test Set
```


# Usage (Federated)
In order to train ```FedNautilus```, run the ```server.py``` script in order to instantiate the aggregation script, as well as the  ```client.py``` script for as many available clients (data silos):

```bash
python server.py [-h] [--bi] [--dspeed] [--dcourse] [--shiptype] [--length LENGTH] [--stride STRIDE] [--max_dt MAX_DT] [--silos SILOS] [--fraction_fit FRACTION_FIT] [--fraction_eval FRACTION_EVAL] [--num_rounds NUM_ROUNDS] [--load_check] [--port PORT] [--mu MU]

python client.py [-h] --data {brest_1800,brest_3600,piraeus_1800,piraeus_3600,mt_1800,mt_3600} [--gpuid GPUID] [--crs CRS] [--bi] [--dspeed] [--dcourse] [--shiptype] [--bs BS] [--length LENGTH] [--stride STRIDE] [--aug] [--max_dt MAX_DT] [--load_check] [--port PORT] [--silos SILOS] [--mu MU] [--fraction_fit FRACTION_FIT] [--fraction_eval FRACTION_EVAL] [--personalize] [--global_ver GLOBAL_VER] [--num_rounds NUM_ROUNDS]
```

For training ```Nautilus``` up to specifications defined in the paper, adjust the above command as follows:

```bash
python client.py --data {brest_1800,piraeus_1800,mt_1800} --gpuid 3 --bs 1 --shiptype --crs 2100 --length 32 --stride 16 --dspeed --dcourse --port 8080 --mu 1 --fraction_fit 1 --silos 3 --max_dt {1800,3600}

python server.py --shiptype --dspeed --dcourse --length 32 --stride 16 --num_rounds 70 --silos 3 --port 8080 --mu 1 --fraction_fit 1 --max_dt {1800,3600}
```

for the 30 min. and 60 min. variant, respectively. To run personalization, run the ```client.py``` script with the same parameters and append the ```--personalization``` flag, as illustrated in the following example:

```bash
python client.py --data {brest_1800,piraeus_1800,mt_1800} --gpuid 3 --bs 1 --shiptype --crs 2100 --length 32 --stride 16 --dspeed --dcourse --port 8080 --mu 1 --fraction_fit 1 --silos 3 --max_dt {1800,3600} --personalize
```


## Documentation (server)
```bash
  -h, --help            show this help message and exit
  --silos SILOS         #Data Silos (default: 3)
  --fraction_fit FRACTION_FIT
                        #clients to train per round (%)
  --fraction_eval FRACTION_EVAL
                        #clients to evaluate per round (%)
  --num_rounds NUM_ROUNDS
                        #FL Rounds (default: 170)
  --load_check          Continue from Latest FL Round
  --port PORT           Server Port
  --mu MU               Proximal $\mu$
```


## Documentation (client)
```bash
  -h, --help            show this help message and exit
  --personalize         Fine-tune the global model to the local clients data
  --global_ver GLOBAL_VER
                        Version of global model to load
  --num_rounds NUM_ROUNDS
                        Number of epochs for fine-tuning
```



# On Reproducing the Experimental Study

For the sake of convenience the preprocessed versions of the open datasets used in our experimental study can be found in the directory ```./data/{brest,piraeus}-dataset/10_sec__{1800,3600}_sec/dataset_trajectories_preprocessed_with_type.fixed.csv``` (after extracting the corresponding zip files). To extract the files, use an application, such as [7-zip](https://www.7-zip.org) (Windows), [The Unarchiver](https://theunarchiver.com) (Mac), or the following terminal commands (Linux/Mac):

```bash
ls -v 10_sec__{1800,3600}_sec.z* | xargs cat > 10_sec__{1800,3600}_sec.zip.fixed
unzip 10_sec__{1800,3600}_sec.zip.fixed
```

To reproduce the experimental study, i.e., test the performance of the models in the datasets' test set, run the following script (using the same parameters/flags as the aforementioned scripts):

```bash
python model-evaluation.py --data {brest_1800,brest_3600,piraeus_1800,piraeus_3600,mt_1800,mt_3600} [--gpuid GPUID] [--crs CRS] [--bi] [--dspeed] [--dcourse] [--shiptype] [--bs BS] [--length LENGTH] [--stride STRIDE] [--aug] [--max_dt MAX_DT] [--patience PATIENCE] [--silos SILOS] [--fraction_fit FRACTION_FIT] [--fraction_eval FRACTION_EVAL] [--cml] [--fl] [--perfl] [--global_ver GLOBAL_VER] [--mu MU]
```

## Documentation
```bash
  --cml                 Evaluate Nautilus
  --fl                  Evaluate (global)FedNautilus
  --perfl               Evaluate (per)FedNautilus
```


# Contributors
Andreas Tritsarolis; Department of Informatics, University of Piraeus

Nikos Pelekis; Department of Statistics & Insurance Science, University of Piraeus

Konstantina Bereta; Kpler

Dimitris Zissis; Department of Product & Systems Design Engineering, University of the Aegean

Yannis Theodoridis; Department of Informatics, University of Piraeus


# Citation
If you use [Fed]Nautilus in your project, we would appreciate citations to the following paper:

> Andreas Tritsarolis, Nikos Pelekis, Konstantina Bereta, Dimitris Zissis, and Yannis Theodoridis. 2024. On Vessel Location Forecasting and the Effect of Federated Learning. In Proceedings of the 25th Conference on Mobile Data Management (MDM).


# Acknowledgement
This work was supported in part by the Horizon Framework Programme of the European Union under grant agreement No. 101070279 (MobiSpaces; https://mobispaces.eu). In this work, Kpler provided the Aegean AIS dataset and the requirements of the business case.
