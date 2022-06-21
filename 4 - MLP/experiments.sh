python3 src/mlp.py --dataset original
python3 src/mlp.py --dataset original --balanced yes
python3 src/mlp.py --dataset lda --width 2 --depth 1 --lr 1e-3 --momentum 5
python3 src/mlp.py --dataset pca --width 1024 --depth 1 --lr 1e-2 --momentum .5
python3 src/mlp.py --dataset autoenc --width 32 --depth 4 --lr 1e-1 --momentum .9
python3 src/mlp.py --dataset pca_corr1 --width 256 --depth 1 --lr 1e-1 --momentum .5
python3 src/mlp.py --dataset pca_corr2 --width 128 --depth 1 --lr 1e-1 --momentum .5
python3 src/mlp.py --dataset pca_corr3 --width 512 --depth 1 --lr 1e-2 --momentum .5
python3 src/mlp.py --dataset pca --small yes --width 128 --depth 1 --lr 1e-1 --momentum 0.
python3 src/mlp.py --dataset pca_corr1 --small yes --width 256 --depth 1 --lr 1e-2 --momentum .9
python3 src/mlp.py --dataset pca_corr2 --small yes --width 256 --depth 1 --lr 1e-2 --momentum .5
python3 src/mlp.py --dataset pca_corr3 --small yes --width 128 --depth 1 --lr 1e-1 --momentum 0.
python3 src/mlp.py --dataset lda --balanced yes --width 8 --depth 2 --lr 1e-2 --momentum 0.
python3 src/mlp.py --dataset pca --balanced yes --width 256 --depth 4 --lr 1e-1 --momentum .9
python3 src/mlp.py --dataset autoenc --balanced yes --width 32 --depth 8 --lr 1e-1 --momentum .9