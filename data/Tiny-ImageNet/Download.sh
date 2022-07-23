rm -rf tiny-imagenet-200/
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip -q tiny-imagenet-200.zip
python tinyimagenet-reformat.py
