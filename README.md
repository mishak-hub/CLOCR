# CLOCR

## Prerequisites
* Anaconda: [WSL Installation Guide](https://medium.com/hydroinformatics/software-development-in-linux-install-miniconda-in-wsl-27e809a0c064)

It is quite intimidating to install Anaconda on Newton. I recommend doing the following:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```
Now, you should `tmux` into your newton shell to keep whatever you're doing alive:
```bash
tmux new -s jupyter
```
You can use ctrl-b and then press `[` to be able to scroll with pgUP/pgDown. You can also press ctrl-b and type `:` if you need to scroll with your mouse. You need to hold down Shift to select text copyable with ctrl+shift+c. 

For some reason, Newton doesn't like Conda installing things with multiple threads without being in a slurm job. So, do this:
```bash
srun -p main --time=02:00:00 --ntasks-per-node 2 --pty bash
```
Then, continue through the sections below.

* Rust/Cargo: Just run the following bash command: `curl https://sh.rustup.rs -sSf | sh` ([source](https://doc.rust-lang.org/cargo/getting-started/installation.html))

## To set up llms_post-ocr_correction-main Folder
This folder hosts the training suite. `ls` into that folder, and run:
```bash
conda env create -f environment.yml
conda activate clocr
```
Newton may give you issues creating the environment. I was able to get it running on my Newton, but I have to check if there were any tweaks I made before running `conda env create`. 

As of now, the code is not finished in implementation, so don't go trying out `pipeline.py` yet!