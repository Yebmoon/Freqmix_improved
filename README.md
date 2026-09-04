This is the code for the paper "Frequency-Domain Mixing Data Augmentation for Malicious Traffic Detection".

## Prepare data

The input traffic data should be stored in CSV format. 

`len_sequences`: packet length sequence of a network flow.

`label`: class label of the flow, where `0` and `1` denote the two classes.

Additional flow information can also be retained in the CSV file (see `flow_column_types`.)

For example:

```
len_sequences,label
60_1500_1500_60_200,0
74_128_256_512_64,1
```

The datasets we used are listed below:

* CIRA-CIC-DoHBrw2020: https://www.unb.ca/cic/datasets/dohbrw-2020.html
* CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
* CSE-CIC-IDS2018: https://www.unb.ca/cic/datasets/ids-2018.html
* MAWILab: https://mawi.wide.ad.jp/mawi/
* CSD: This is a private dataset. Due to privacy and confidentiality agreements, it cannot be made publicly available.

## Running the code

The meaning of each argument is as follows:

```
--train_path      Path to the in-distribution (ID) dataset.
--test_path       Path to the out-of-distribution (OOD) test dataset.
--model_name      Classification model. Options: lstm, CNN, Transformer, DF, TMWF_DFNet, BAPM.
--save_dir        Directory for saving model checkpoints and training logs.
--aug             Data augmentation method. Options: ours, freq_mix, mixup, noise, random_mask, reperm, rosetta, no.
--should_train    Enable model training. If omitted, the saved best model is loaded for evaluation.
--alpha           Parameter of the Beta distribution used in mixing-based augmentation.
--preaug_ratio    Ratio of valid sequence positions used for pre-augmentation.
--window          Window size used for frequency-domain augmentation.
--valid_size      Proportion of the ID dataset used as the validation set.
--test_size       Proportion of the ID dataset used as the ID test set.
```

Running example:

```
python main.py --train_path ./data/train.csv \
			   --test_path ./data/test.csv \
               --model_name lstm \
               --save_dir ./info/lstm_ours \
               --aug ours \
               --should_train
```

For LSTM, CNN, Transformer, DF, TMWF, and BAPM, experiments can be run through `main.py`.  SmartDetector and MATEC are implemented separately and should be run from their respective directories.

Note that our experiments include ten models for evaluation. The remaining two utilize publicly available implementations directly, which are:

* LUCID: https://github.com/doriguzzi/lucid-ddos.
* Whisper: https://github.com/InspiringGroup-NeoLab/CertTA (the supervised version by Yan et al. [1]).

[1] Jinzhu Yan, Zhuotao Liu, Yuyang Xie, Shiyu Liang, Lin Liu, and Ke Xu. 2025. CertTA: Certified Robustness Made Practical for Learning-Based Traffic Analysis. In Proceedings of the 34th USENIX Conference on Security Symposium. USENIX Association, USA, Article 377, 20 pages.

## Acknowledgements

In addition to the implementations of the published models used in our experiments, this repository also makes use of components from [Time-Series-Library](https://github.com/thuml/Time-Series-Library). 
