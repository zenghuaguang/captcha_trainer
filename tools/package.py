#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
from PyInstaller.__main__ import run


def package(prefix):
    opts = ['{}trainer.spec'.format(prefix), '--distpath={}dist'.format(prefix), '--workpath={}build'.format(prefix)]
    run(opts)


if __name__ == '__main__':
    try:
        package("../")
    except FileNotFoundError:
        package("/")
