# CAST: Conditional Attribute Subsampling Toolkit
<img align="right" src="assets/overview.png" style="margin:20px 20px 0px 20px" width="330"/> This is a repository for conditional subsampling of datasets for training and evaluation. Over 50 pre-computed attributes for the WebFace42M dataset including race, gender, and image quality are provided.

Automatic evaluation is provided for Face Recognition.

### README Contents
1. [Run CC11 Benchmark](#Run)
2. [Download WebFace42M Attributes](#Download)
3. [Subsample 1:1 Verfication Sets](#Subsample)
4. [Evaluate New Verification Sets](#Evaluate)
5. [Subsample Training Sets](#Subsample)



### Run CC11 Face Recogntion Benchmark
The CAST-Challenging-11 (CC11) benchmark contains 11 sub-benchmarks which contain only hard verification pairs. The full test set contains 110,000 pairs (10k per sub-benchmark) and the validation set contains 11,000 pairs (1k per sub-benchmark). Use instructions below to run the benchmark.

Download CC11 from [here]()(~1GB) and unzip in the `data` directory. Alternatively, if WebFace 42M is downloaded on your system you can pass the directory path to the script:
```
# cc11 test set
python cc11.py --weights weights_path --arch architecture
# cc11 validation set
python cc11.py --weights weights_path --arch architecture --path data/cc11_val.bin

# cc11 test set with WebFace42M Pre-downloaded
python cc11.py --weights weights_path --arch architecture --path webface42m_root
```
<img align="right" src="assets/ex_results.png" style="margin:0px 20px 0px 20px" width="200"/>

The following architecture keys are implemented in the `model` directory: `r18`, `r34`, `r50`, `r100`, `r200`, `mbf`, `mbf_large`, `vit_t`, `vit_s`, `vit_b`. To use a different backbone architecture, import the implementation to the model directory.

The screenshot on the right shows example output from a ResNet50 trained on WebFace4M.

### Download WebFace42M Attributes
todo

### Subsample Verification Sets
instructions todo
```
python subsample.py
```

### Evaluate New Face Recognition Verification Sets
```todo```

### Subsample Training Sets
```todo```

### Citation
```
@article{CAST,
  title={CAST: Conditional Attribute Subsampling Toolkit for Fine-grained Evaluation},
  authors={},
  journal={},
  year={2022}
}
