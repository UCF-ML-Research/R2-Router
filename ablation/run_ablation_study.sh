#!/usr/bin/bash
#
# Run complete embedding ablation study pipeline
#

echo "================================================================================"
echo "Embedding Model Ablation Study - Complete Pipeline"
echo "================================================================================"

# Step 1: Generate embeddings
echo ""
echo "Step 1/3: Generating embeddings for different models..."
echo "================================================================================"
python ablation/generate_embeddings.py
if [ $? -ne 0 ]; then
    echo "Error: Embedding generation failed!"
    exit 1
fi

# Step 2: Train and evaluate
echo ""
echo "Step 2/3: Training CoRE predictors and evaluating performance..."
echo "================================================================================"
python ablation/train_and_evaluate.py
if [ $? -ne 0 ]; then
    echo "Error: Training and evaluation failed!"
    exit 1
fi

# Step 3: Visualize results
echo ""
echo "Step 3/3: Generating comparison plots..."
echo "================================================================================"
python ablation/visualize_results.py
if [ $? -ne 0 ]; then
    echo "Error: Visualization failed!"
    exit 1
fi

echo ""
echo "================================================================================"
echo "Ablation study completed successfully!"
echo "Results saved to ablation/results/"
echo "================================================================================"
