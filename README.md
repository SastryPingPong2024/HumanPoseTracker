Repository that handles the tracking of the players in a video of two people playing ping pong. Based on Humans in 4D.

Conda environment setup:
```
conda create --name pose python=3.10
conda activate pose
pip install torch
pip install -e .[all]

python demo.py \
    --img_folder example_data/images \
    --out_folder demo_out \
    --batch_size=48 --side_view --save_mesh --full_frame
    
conda install cuda
pip install git+https://github.com/brjathu/PHALP.git
```