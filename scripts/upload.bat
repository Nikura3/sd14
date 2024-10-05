@echo off

title Uploading script
echo Uploading your files to the cluster

REM Variables
set user=dvezzaro
set source_dir=../*
set destination_dir=/home/dvezzaro/hf_sd/

echo - local source folder:%source_dir%
echo - remote destination folder:%destination_dir%

echo Uploading...
scp -r -J %user%@labta.math.unipd.it %source_dir% %user%@labsrv8.math.unipd.it:%destination_dir%
echo Done