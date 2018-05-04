#!/bin/bash

rm -rf logs/actioncoder/*
python models/actioncoder.py $1
