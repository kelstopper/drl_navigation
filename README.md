# TODO

### Introduction

Update README

### CNN Q Network on AWS Deep Learning AMI (Ubuntu 16.04)

#### Step 1: Choose AMI

Create a new instance in the EC2 panel and search for the AMI `ami-016ff5559334f8619` it can be found in region `us-east-1`

OR

Follow the build instruction here: [Training on Amazon Web Service](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)

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
git clone https://github.com/kelstopper/drl_navigation.git && cd drl_navigation

# copy headless linux app
curl https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip > VisualBanana_Linux.zip
unzip VisualBanana_Linux.zip

# use the pytorch env
source activate pytorch_p36
pip install unityagents

# run the cnn example, verify that it is running on CUDA in the logs
## "Training on CUDA" <<< Should be present if "Training on CPU" is present you are training on cpu and it WILL take longer and cost more
python cnn.py
```