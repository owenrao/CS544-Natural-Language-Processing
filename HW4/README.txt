# Guidelines on how to train and predict
## Directory
- HW4
    - models/ (already trained models: blstm1.pt and blstm2.pt)
    - result/ (predictions made by trained models: dev1/dev2.out and test1/test2.out)
    - script/
        - model.py (contains BiLSTM and BiLSTM-GLoVe models as well as corresponding datasets)
        - predict.py (prediction script)
        - train.py (training script)
        - utils.py (utility functions)
## Train
CMD: python <Path to train.py> <Data directory that contains "train" and "dev" files> <Select Model: blstm1/blstm2> <Only if blstm2: embedding path>
eg.: python script/train.py data blstm2 glove.6B.100d.txt

## Test
CMD: python <Path to predict.py> <Data directory that contains "train", "dev" and "test" files> <Select Model: blstm1/blstm2> <Model path> <Only if blstm2: embedding path>
eg.: python script/predict.py data blstm2 models/blstm2.pt  glove.6B.100d.txt