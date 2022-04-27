if [ ! -f data/raw/soc-pokec-profiles.txt ]; then
    wget "https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz" -P "data/Pokec/raw"
    wget "https://snap.stanford.edu/data/soc-pokec-profiles.txt.gz" -P "data/Pokec/raw"
    wget "https://snap.stanford.edu/data/soc-pokec-readme.txt" -P "data/Pokec/raw"
    gzip -d data/Pokec/raw/soc-pokec-relationships.txt.gz
    gzip -d data/Pokec/raw/soc-pokec-profiles.txt.gz
    gzip -d data/Pokec/raw/soc-pokec-readme.txt.gz
fi
python GCN.py --seed 1234 --subsample_rate 1. --dataset "pokec-pets" --epochs 500 --private False --early_stopping True --pokec_feat_type 'bert_avg' --optim_type 'adam' --split_graph False
