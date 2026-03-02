#!/usr/bin/env python
# coding=utf-8
'''
Author:Tai Lei
Date:Thu Nov 22 12:09:33 2018
Info:
'''

import torch
import torch.nn as nn
# from torch.nn import functional as F


class CarlaNet(nn.Module):
    def __init__(self):
        super(CarlaNet, self).__init__()
        self.conv_block_left = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),

            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),

            nn.Conv2d(36, 48, kernel_size=3, stride=2),
            nn.ELU(),

            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ELU(),
        )

        self.conv_block_central = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),

            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),

            nn.Conv2d(36, 48, kernel_size=3, stride=2),
            nn.ELU(),

            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ELU(),
        )

        self.conv_block_right = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),

            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),

            nn.Conv2d(36, 48, kernel_size=3, stride=2),
            nn.ELU(),

            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ELU(),
        )

        self.conv_block_map= nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=2),
            nn.ELU(),

            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),

            nn.Conv2d(36, 48, kernel_size=3, stride=2),
            nn.ELU(),
        )


        self.img_fc = nn.Sequential(
                nn.Linear(24960, 100),
                nn.ELU(),
                nn.Linear(100, 100),
                nn.ELU(),
            )


        self.conv_block_routed_map= nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),

            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
        )


        self.out = nn.Sequential(
                nn.Linear(3700, 100),
                nn.ELU(),
                nn.Linear(100, 2),
            )


    def forward(self, img_left, img_central,img_right,map,routed_map):
        img_l = self.conv_block_left(img_left)
        img_l = img_l.view(-1, 8064)

        img_c = self.conv_block_central(img_central)
        img_c = img_c.view(-1, 8064)

        img_r = self.conv_block_left(img_right)
        img_r = img_r.view(-1, 8064)

        img_map = self.conv_block_map(map)
        img_map = img_map.view(-1,768)

        feature1 = torch.cat([img_l,img_c,img_r,img_map],dim = 1)
        feature1 = self.img_fc(feature1)

        img_routed_map = self.conv_block_routed_map(routed_map)
        img_routed_map = img_routed_map.view(-1,3600)

        feature2 = torch.cat([feature1,img_routed_map],dim = 1)
        out= self.out(feature2)
        return out

if __name__ == '__main__':
    a = torch.ones([1,3,80,200])
    b = torch.ones([1, 3, 80, 200])
    c = torch.ones([1, 3, 80, 200])
    d = torch.ones([1, 1, 50, 50])
    e = torch.ones([1, 3, 50, 50])
    model = CarlaNet()
    model.cpu()
    out = model(a,b,c,d,e)
