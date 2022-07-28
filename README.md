# CAST: Conditional Attribute Subsampling Toolkit
<img align="right" src="assets/overview.png" style="margin:20px 20px 0px 20px" width="270"/> This is a repository for subsampling datasets for training and evaluation. Over 50 pre-computed attributes for the WebFace42M dataset including race, gender, and image quality can be downloaded here (todo).

Automatic evaluation is provided for Face Recognition.


### Run CC11 Face Recogntion Benchmark
The CC11 benchmark contains 11 sub-benchmarks which only contain hard verification pairs. Run the following to get results.
```
python cc11.py --weights weights_path --arch architecture
```
The following architecture keys are implemented in the `model` dir: `r10`, ``.

To use a different backbone architecture, add the implementation to the model directory.

### Subsample Verification Sets
instructions todo
```
python run.py
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
