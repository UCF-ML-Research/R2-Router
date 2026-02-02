source .venv/bin/activate

GPU_COUNTS=2
BATCH_SIZE=256

CSV_PATHS=(
  ./results/microsoft/phi-4/0.csv
  # ./results/microsoft/phi-4/10.csv
  # ./results/microsoft/phi-4/20.csv
  # ./results/microsoft/phi-4/30.csv
  # ./results/microsoft/phi-4/40.csv
  ./results/microsoft/phi-4/50.csv
  ./results/microsoft/phi-4/80.csv
  ./results/microsoft/phi-4/100.csv
  ./results/microsoft/phi-4/150.csv
  ./results/microsoft/phi-4/200.csv
  ./results/microsoft/phi-4/300.csv
  ./results/microsoft/phi-4/500.csv
  ./results/microsoft/phi-4/800.csv
  ./results/microsoft/phi-4/1200.csv
  ./results/microsoft/phi-4/2000.csv
  ./results/microsoft/phi-4/4000.csv
)

python judge.py ${CSV_PATHS[@]} --gpu_counts $GPU_COUNTS --batch_size $BATCH_SIZE > judge.log 2>&1