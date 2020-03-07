# TODO

### Introduction

Update README

### CNN Q Network on AWS Deep Learning AMI (Ubuntu 16.04)

#### Step 1: Choose AMI

Create a new instance in the EC2 panel and search for the AMI "Deep Learning AMI (Ubuntu 16.04)"

Select continue

#### Step 2: Choose instance type

Select Family: `GPU Instances` Type: `p2.xlarge`

#### Step 3: Configure Instance

leave alone

#### Step 4: Add Storage

leave alone

#### Step 5: Add Tags

configure any tags you see fit

#### Step 6: Configure Security Group

allow ssh connections (default) - create new security group

#### Launch

create new keypair and download

when the instance is running connect to it using the pem file previously downloaded

```bash
# change permissions on pem file
chmod 600 ~/.ssh/p2-xlarge-drl.pem

# connect
ssh -i ~/.ssh/p2-xlarge-drl.pem ubuntu@{hostname found in ec2 dashboard}
```

Install dependencies in the environment

```bash
# install compatible python
curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash

## apt dependencies
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git

## edit ~/.bashrc and add the following lines
echo "export PATH=\"/home/ubuntu/.pyenv/bin:$PATH\"" >> ~/.bashrc
## reload env
source ~/.bashrc
echo "$(pyenv init -)" >> ~/.bashrc
echo "$(pyenv virtualenv-init -)" >> ~/.bashrc
## reload env
source ~/.bashrc

## install python 3.5.9
pyenv install 3.7.2
pyenv global 3.7.2

# install unity ml agents
git clone --branch latest_release https://github.com/Unity-Technologies/ml-agents.git
cd ml-agents/ml-agents/
pip3 install -e .
```

Install xorg
```bash
# Install Xorg
sudo apt-get update
sudo apt-get install -y xserver-xorg mesa-utils
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024

# Get the BusID information
nvidia-xconfig --query-gpu-info

# Add the BusID information to your /etc/X11/xorg.conf file
sudo sed -i 's/    BoardName      "Tesla K80"/    BoardName      "Tesla K80"\n    BusID          "0:30:0"/g' /etc/X11/xorg.conf

# Remove the Section "Files" from the /etc/X11/xorg.conf file
# And remove two lines that contain Section "Files" and EndSection
sudo vim /etc/X11/xorg.conf
```

Update the NVIDIA driver
```bash
# Download and install the latest Nvidia driver for ubuntu
# Please refer to http://download.nvidia.com/XFree86/Linux-#x86_64/latest.txt
$ wget http://download.nvidia.com/XFree86/Linux-x86_64/390.87/NVIDIA-Linux-x86_64-390.87.run
$ sudo /bin/bash ./NVIDIA-Linux-x86_64-390.67.run --accept-license --no-questions --ui=none

# Disable Nouveau as it will clash with the Nvidia driver
sudo echo 'blacklist nouveau'  | sudo tee -a /etc/modprobe.d/blacklist.conf
sudo echo 'options nouveau modeset=0'  | sudo tee -a /etc/modprobe.d/blacklist.conf
sudo echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
sudo update-initramfs -u
```

Restart EC2
```bash
sudo reboot now
```

Ensure no Xorg processes are running
```bash
sudo killall Xorg
nvidia-smi
```

Start x server and use it
```bash
# Start the X Server, press Enter to come back to the command line
sudo /usr/bin/X :0 &

# Check if Xorg process is running
# You will have a list of processes running on the GPU, Xorg should be in the list.
nvidia-smi

# Make the ubuntu use X Server for display
export DISPLAY=:0
```

Ensure it is configured
```bash
# For more information on glxgears, see ftp://www.x.org/pub/X11R6.8.1/doc/glxgears.1.html.
glxgears
# If Xorg is configured correctly, you should see the following message

# Running synchronized to the vertical refresh.  The framerate should be
# approximately the same as the monitor refresh rate.
# 137296 frames in 5.0 seconds = 27459.053 FPS
# 141674 frames in 5.0 seconds = 28334.779 FPS
# 141490 frames in 5.0 seconds = 28297.875 FPS
```

```bash
# clone environment from git
git clone https://github.com/kelstopper/drl_navigation.git

# copy headless linux app
curl https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip > Banana_Linux_NoVis.zip

unzip Banana_Linux_NoVis.zip
```