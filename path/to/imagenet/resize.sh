#!/bin/bash

for name in /val/*.JPEG
do
    convert -resize 256x256\! $name $name
done