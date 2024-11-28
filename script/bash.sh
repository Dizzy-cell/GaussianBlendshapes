python train_one.py --source_path id2_ori --model_path ./output/id2 --use_ict True
python test_one.py --source_path id2_ori --model_path ./output/id2 --use_ict True --render_train True --load_iteration 20000

python script/convert_splat.py --input_dir ./output/id2_ori/ply_train

quicksrnet 
FaceUnet