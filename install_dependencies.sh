#!/usr/bin/env bash

######################################################################
# Torch dependencies install

TOPDIR=$PWD

# Build and install Torch7
mkdir tmp
cd tmp

echo "Installing Xitari ... "
cd $TOPDIR/tmp
rm -rf xitari
git clone https://github.com/deepmind/xitari.git
cd xitari
luarocks make
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
echo "Xitari installation completed"

echo "Installing Alewrap ... "
cd $TOPDIR/tmp
rm -rf alewrap
git clone https://github.com/deepmind/alewrap.git
cd alewrap
luarocks make
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
echo "Alewrap installation completed"

