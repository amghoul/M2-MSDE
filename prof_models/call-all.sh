sh /netscratch/alghoul/install2.sh
pip install git+https://github.com/nathanpainchaud/pytorch-summary.git@fix/layer_dict_output
python3 /netscratch/alghoul/code/novel-stereonet-v04/prof_models/StereoNet_Multi_FactorizedConv3D.py > ./prof_models/nv-sn-v04-4scales-ab13.txt
#python3 /netscratch/alghoul/code/novel-stereonet-v04/prof_models/StereoNet_Multi_FactorizedConv3D.py > ./prof_models/StereoNet_nv-sn-v04-memorysf4-chout-32-pixelacc-1.txt
