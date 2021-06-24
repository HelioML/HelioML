source ~/anaconda3/etc/profile.d/conda.sh
conda activate helioml
conda remove --name helioml_ch3 --all
rm -rf /Users/jmason86/Library/Jupyter/kernels/helioml_ch3
conda env create -f environment_ch3.yml
conda activate helioml_ch3
python -m ipykernel install --user --name=helioml_ch3
jupyter lab