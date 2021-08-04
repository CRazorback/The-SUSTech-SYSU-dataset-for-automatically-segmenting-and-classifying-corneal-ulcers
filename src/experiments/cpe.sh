python run.py --gpus 1 --gpu 0 -b 16 -e 1 \
--epoch 100 --dataset_name sustechsysu --model UNet -j 4 --test_batch_size 16 \
--save_name ./experiments/sustechsysu/cpe --lr 1e-3 --kfold --inductive_bias distance