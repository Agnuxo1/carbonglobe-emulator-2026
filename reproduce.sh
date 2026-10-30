#!/usr/bin/env bash
# End-to-end reproduction. Needs a Kaggle API token (~/.kaggle/kaggle.json) and
# acceptance of the competition rules; the data are not redistributed here.
#
#   DATA=/path/with/space  bash reproduce.sh
#
# Disk: about 30 GB for the raw arrays plus 4 GB of derived arrays.
set -euo pipefail
DATA="${DATA:?set DATA to a directory with ~35 GB free}"
export CG_DIR="$DATA/carbonglobe" COMP_DIR="$DATA/competition" WORK_DIR="$DATA/work"
mkdir -p "$CG_DIR" "$COMP_DIR" "$WORK_DIR"

echo "== 1. download the present-day benchmark (about 25 GB)"
for f in readme.txt data_stats/data_stats.npz data_stats/age_triplet.npz \
         data_stats/mask2d.npy train.csv val.csv test.csv \
         data_global/glob_X_fea.npy; do
  kaggle datasets download zhihaow/carbonglobe -f "$f" -p "$CG_DIR" --force
done
for a in 001 010 020 030 050 070 100 150 200 250 300 350 400 450 500; do
  kaggle datasets download zhihaow/carbonglobe -f "data_global/glob_Y${a}.npy" -p "$CG_DIR" --force
done

echo "== 2. download the competition test data"
for f in public/README.md public/sites_ssp.csv public/sample_submission.csv \
         public/test_ssp126.npz public/test_ssp585.npz; do
  kaggle competitions download -c ieee-bigdata-cup-2026-ai-emulation-challenge -f "$f" -p "$COMP_DIR" --force
done

echo "== 3. derived arrays"
python src/unpack_test.py
python src/build_features.py targets
python src/build_onestep.py
python src/shift_report.py          # writes bad_channels.json

echo "== 4. the system that actually scored best: the untrained age-shifted anchor"
python src/probes.py                # P2_anchor_raw.csv is the 0.094 submission

echo "== 5. H1 ablation: own technology vs professional control, 3 seeds each"
for s in 0 1 2; do
  python src/nn_onestep.py    --exp AB_nn_s$s    --epochs 8 --tf-epochs 3 --seed $s
  python src/nn_onestep.py    --exp AB_nnhrc_s$s --epochs 8 --tf-epochs 3 --seed $s --hrc-dim 300
  python src/hrc_onestep.py   --exp AB_hrc_s$s   --size 801 --leaks 0.1,0.3,0.7 --square --seed $s
done
python src/summarize_ablation.py

echo "== 6. candidates built on top of the anchor"
python src/submit_nn.py --name S05_nn_onestep --epochs 10
python src/blend.py anchor_affine
python src/blend.py delta --model S05_nn_onestep_raw.npy
python src/blend.py blend --model S05_nn_onestep_raw.npy --weights 0.25,0.4,0.55
echo "done; candidates are in \$WORK_DIR/subs"
