#!/usr/bin/env python

from __future__ import print_function
from math import *
import os
import sys
import math
import numpy as np
import rospy, time, tf
import cv2
import time
from cv_bridge import CvBridge, CvBridgeError
from time import gmtime, strftime
from threading import Thread
from sensor_msgs.msg import Image as SensorImage
from geometry_msgs.msg import Twist, Point, PointStamped, PoseWithCovarianceStamped, PoseStamped
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

Centralimage_c = np.array((480,640,3)).astype(np.uint8)
Centralimage_l = np.array((480,640,3)).astype(np.uint8)
Centralimage_r = np.array((480,640,3)).astype(np.uint8)
Centralimage_c2 = np.array((480,640,3)).astype(np.uint8)
Centralimage_l2 = np.array((480,640,3)).astype(np.uint8)
Centralimage_r2 = np.array((480,640,3)).astype(np.uint8)
Map_image = np.array((480,640,3)).astype(np.uint8)

ReceiveCentral_c = True
ReceiveCentral_l = True
ReceiveCentral_r = True
ReceiveCentral_c2 = False
ReceiveCentral_l2 = False
ReceiveCentral_r2 = False
Receivemap = True
angle = 0
velocity = 0
flag = 0
command = 0

rootpath = '/home/nvidia/train/'



def CenrtralImagecallback_central(data):
    global Centralimage_c
    global ReceiveCentral_c

    bridge = CvBridge()
    try:
        Centralimage_c = bridge.imgmsg_to_cv2(data, 'bgr8')
        #Centralimage = cv2.resize(Centralimage, (640,480), interpolation=cv2.INTER_LINEAR)
        ReceiveCentral_c = True
    except CvBridgeError as e:
        print(e)

def CenrtralImagecallback_left(data):
    global Centralimage_l
    global ReceiveCentral_l

    bridge = CvBridge()
    try:
        Centralimage_l = bridge.imgmsg_to_cv2(data, 'bgr8')
        #Centralimage = cv2.resize(Centralimage, (640,480), interpolation=cv2.INTER_LINEAR)
        ReceiveCentral_l = True
    except CvBridgeError as e:
        print(e)

def CenrtralImagecallback_right(data):
    global Centralimage_r
    global ReceiveCentral_r

    bridge = CvBridge()
    try:
        Centralimage_r = bridge.imgmsg_to_cv2(data, 'bgr8')
        #Centralimage = cv2.resize(Centralimage, (640,480), interpolation=cv2.INTER_LINEAR)
        ReceiveCentral_r = True
    except CvBridgeError as e:
        print(e)


def CenrtralImagecallback_central2(data):
    global Centralimage_c2
    global ReceiveCentral_c2

    bridge = CvBridge()
    try:
        Centralimage_c2 = bridge.imgmsg_to_cv2(data, 'bgr8')
        #Centralimage = cv2.resize(Centralimage, (640,480), interpolation=cv2.INTER_LINEAR)
        ReceiveCentral_c2 = True
    except CvBridgeError as e:
        print(e)

def CenrtralImagecallback_left2(data):
    global Centralimage_l2
    global ReceiveCentral_l2

    bridge = CvBridge()
    try:
        Centralimage_l2 = bridge.imgmsg_to_cv2(data, 'bgr8')
        #Centralimage = cv2.resize(Centralimage, (640,480), interpolation=cv2.INTER_LINEAR)
        ReceiveCentral_l2 = True
    except CvBridgeError as e:
        print(e)

def CenrtralImagecallback_right2(data):
    global Centralimage_r2
    global ReceiveCentral_r2

    bridge = CvBridge()
    try:
        Centralimage_r2 = bridge.imgmsg_to_cv2(data, 'bgr8')
        #Centralimage = cv2.resize(Centralimage, (640,480), interpolation=cv2.INTER_LINEAR)
        ReceiveCentral_r2 = True
    except CvBridgeError as e:
        print(e)

def mapCallback(data):
    global Map_image
    global Receivemap

    bridge = CvBridge()
    try:
        Map_image = bridge.imgmsg_to_cv2(data, 'bgr8')
        Receivemap = True

    except CvBridgeError as e:
        print(e)


def Callback(data):

    global angle
    global velocity

    angle = data.angular.z
    velocity = data.linear.x

def FlagCallback(data):
    global command
    global flag
    if data.buttons[0]==1:
        command = 0
    if data.buttons[1]==1:
        command = 2
    if data.buttons[2]==1:
        command = 1
    if data.buttons[3]==1:
        command = 3

    if data.buttons[7]==1:
        flag = 1
    if data.buttons[6]==1:
        flag = 0


def Save():
 
    global Centralimage_c
    global ReceiveCentral_c
    global Centralimage_l
    global ReceiveCentral_l
    global Centralimage_r
    global ReceiveCentral_r
    global Centralimage_c2
    global ReceiveCentral_c2
    global Centralimage_l2
    global ReceiveCentral_l2
    global Centralimage_r2
    global ReceiveCentral_r2
    global angle
    global velocity
    global flag
    global command
    global Map_image
    global Receivemap

    rate = rospy.Rate(50)

    while not rospy.is_shutdown():
   
        if flag == 1:

            print("creating a new experiment")
            experiment = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
            exp_dir = rootpath + experiment + '/' 
            os.makedirs(exp_dir + '/central/')
            os.makedirs(exp_dir + '/left/')
            os.makedirs(exp_dir + '/right/')
            os.makedirs(exp_dir + '/map/')
            #os.makedirs(exp_dir + '/central2/')
            #os.makedirs(exp_dir + '/left2/')
            #os.makedirs(exp_dir + '/right2/')

            lable_dir = exp_dir+ 'sync_steering.txt'
            velocity_dir = exp_dir + 'velocity.txt'
            command_dir = exp_dir + 'command.txt'
            image_dir_central = exp_dir+ '/central/'
            image_dir_left = exp_dir+ '/left/'
            image_dir_right = exp_dir+ '/right/'
            image_dir_map = exp_dir + '/map/'
            #image_dir_central2 = exp_dir + '/central2/'
            #image_dir_left2 = exp_dir + '/left2/'
            #image_dir_right2 = exp_dir + '/right2/'

            i = 1

            while not rospy.is_shutdown():
                if ReceiveCentral_c and ReceiveCentral_l and ReceiveCentral_r and Receivemap:
                    file = open(lable_dir, 'a')
                    file.write(str(angle)+'\n')
                    file = open(velocity_dir, 'a')
                    file.write(str(velocity)+'\n')
                    file = open(command_dir, 'a')
                    file.write(str(command)+'\n')
                    print (i)
                    cv2.imwrite(os.path.join(image_dir_central, 'frame_'+ format(str(i), '0>5s')+ ".jpg"), Centralimage_c)
                    cv2.imwrite(os.path.join(image_dir_left, 'frame_'+ format(str(i), '0>5s')+ ".jpg"), Centralimage_l)
                    cv2.imwrite(os.path.join(image_dir_right, 'frame_'+ format(str(i), '0>5s')+ ".jpg"), Centralimage_r)
                    cv2.imwrite(os.path.join(image_dir_map, 'frame_' + format(str(i), '0>5s') + ".jpg"), Map_image)
                #cv2.imwrite(os.path.join(image_dir_central2, 'frame_' + format(str(i), '0>5s') + ".jpg"),Centralimage_c2)
                #cv2.imwrite(os.path.join(image_dir_left2, 'frame_' + format(str(i), '0>5s') + ".jpg"),
                #            Centralimage_l2)
                #cv2.imwrite(os.path.join(image_dir_right2, 'frame_' + format(str(i), '0>5s') + ".jpg"),
                #            Centralimage_r2)
                    i = i + 1
                    time.sleep(0.25)


                if flag == 0:
                    print("experiment done")
                    break

        if flag == 0:
            
            while True:
                if flag != 0:
                    break
                print("Pause")
                time.sleep(1)




if __name__ == "__main__":
    

    rospy.init_node('robots')
 
    rospy.Subscriber('/camera/central', SensorImage, CenrtralImagecallback_central, queue_size=1)
    rospy.Subscriber('/camera/left', SensorImage, CenrtralImagecallback_left, queue_size=1)
    rospy.Subscriber('/camera/right', SensorImage, CenrtralImagecallback_right, queue_size=1)
    #rospy.Subscriber('/camera/b_c', SensorImage, CenrtralImagecallback_central2, queue_size=1)
    #rospy.Subscriber('/camera/b_l', SensorImage, CenrtralImagecallback_left2, queue_size=1)
    #rospy.Subscriber('/camera/b_r', SensorImage, CenrtralImagecallback_right2, queue_size=1)
    rospy.Subscriber('/scout/map', SensorImage, mapCallback, queue_size=1)

    rospy.Subscriber('/cmd_vel', Twist, Callback, queue_size=1)
    rospy.Subscriber('/joy', Joy, FlagCallback, queue_size=1)

    print ("Started exploration.")

    #Thread(target=request_pos_at_rate, name="request_pos_at_rate Thread", args=[50]).start()

    Save()
    #Thread(target=Save, name="Save Thread").start()
    
