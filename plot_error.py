import os, time
import numpy as np
import matplotlib.pyplot as plt

import ctypes
import _ctypes
from ctypes import cdll

import sys

import pickle
import csv

from pyquaternion import Quaternion

def convertAngularVelocity2Quaternion(av,dt):
    angle=dt*av    
    s=np.sqrt(sum(angle*angle))
    s0=np.sqrt(sum(av*av))
 
    w=np.cos(s/2.0)
    x=(angle[0]/s) * np.sin(s/2.0)
    y=(angle[1]/s) * np.sin(s/2.0)
    z=(angle[2]/s) * np.sin(s/2.0)
        
    return Quaternion(w,x,y,z)
    
def getCsvData(filename):
    csv_file=open(filename,'r')
    csv_reader=csv.reader(csv_file)
    label_csv=[]

    for row in csv_reader:    
        label_csv.append(row)           
    csv_file.close()

    return label_csv


def makeErrorBag_1fingers(ground_position_allsequence,test_position_allsequence,fingername,frameMax):
    errorBag={}
    errorBag[fingername[0]]=[]
        
    for frame in range(frameMax):
        ground_position=np.array(ground_position_allsequence[frame],'float')
        test_position=np.array(test_position_allsequence[frame],'float')
        
        if len(ground_position)!=2*len(fingername):
            print("the number of fingertip is missing in ground truth..",frame)
            continue
        
        #(xi,yi)->(ei)
        error=np.linalg.norm(ground_position[0:2]-test_position[2*(3):2*(4)])
        #bag of error
        errorBag[fingername[0]].append(error)
            
    return errorBag
        

if __name__=="__main__":   
    frameMax=2370   #2039
    speed_threshold=60  #65
    fingername=['little']
    
    #get fast/slow frames
    fast_frames=[]
    fastExtremely_frames=[]
    slow_frames=[]
    
    slow_frames=range(300)
    
    gyropath='dataset/angularVelocity/'
    for fr in range(300,frameMax):
        av_sum=0
        av_num=0
        for j in range(3):
            avbag=np.asarray(getCsvData(gyropath+'%d.txt'%(fr-1+j)),'float32')
            av_num+=len(avbag)
            for av in avbag:
                av_sum+=np.linalg.norm(av)
                
        if av_sum<speed_threshold:
            fast_frames.append(fr) 
            #print('fast',av_sum/av_num,av_num)
        else:
            fastExtremely_frames.append(fr)
            
            #print('very fast',av_sum/av_num,av_num)
            
            
    #slow_frames=range(400)
    #fast_frames=range(400,frameMax)       
    print('slow frames',len(slow_frames))
    print('fast frames',len(fast_frames))
    print('fast extremely frames',len(fastExtremely_frames))
    
    #read result from algorithm
    position_allsequence_ground=getCsvData('results/groundtruth/position2D.txt')
    position_allsequence_poseREN=getCsvData('results/poseREN/position2D.txt')
    position_allsequence_uvr=getCsvData('results/fusion/position2D.txt')
    position_allsequence_epfl=getCsvData('results/epfl2017/position2D.txt')
    position_allsequence_forth=getCsvData('results/PSO/position2D.txt')
    position_allsequence_ismar=getCsvData('results/ismar2018-gyro/position2D.txt')
    

    methodname=["{}\n{}".format('Oikonomidis', '2011'),
                "{}\n{}".format('Tkach', '2017'),
                "{}\n{}".format('Park', '2018'),
                "{}\n{}".format('Chen', '2019'),
                'Our work']

    
    errorBag={} 
    errorBag[methodname[3]]=makeErrorBag_1fingers(position_allsequence_ground,position_allsequence_poseREN,fingername,frameMax)    
    errorBag[methodname[4]]=makeErrorBag_1fingers(position_allsequence_ground,position_allsequence_uvr,fingername,frameMax)    
    errorBag[methodname[1]]=makeErrorBag_1fingers(position_allsequence_ground,position_allsequence_epfl,fingername,frameMax)    
    errorBag[methodname[0]]=makeErrorBag_1fingers(position_allsequence_ground,position_allsequence_forth,fingername,frameMax)    
    errorBag[methodname[2]]=makeErrorBag_1fingers(position_allsequence_ground,position_allsequence_ismar,fingername,frameMax)  

    #show boxplot on fast frames
    error=[] 
    for i in range(len(methodname)):
        error.append(np.asarray(errorBag[methodname[i]]['little'])[slow_frames])
        error.append(np.asarray(errorBag[methodname[i]]['little'])[fast_frames])
        error.append(np.asarray(errorBag[methodname[i]]['little'])[fastExtremely_frames])
        
    if not 'fig' in locals():
        fig,ax =plt.subplots()    
        
    ax.cla()
    ax.set_ylabel('2D error [pixel]')
    
    ax.boxplot(error,showfliers=False,positions=[1,1.5,2, 3,3.5,4, 5,5.5,6, 7,7.5,8, 9,9.5,10])
    #plt.xticks(range(1,len(methodname)+1), methodname)
    
    
    ax.set_xticks([1.2, 3.2, 5.2, 7.2, 9.2])
    ax.set_xticklabels(methodname)
    
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    
    ax.set_ylim(0,60)
    
    #print('average',np.mean(error[0]),np.mean(error[1]))
    #print('median',np.median(error[0]),np.median(error[1]))
    #print('stdev',np.std(error[0]),np.std(error[1]))        
        
        
    #debug 
    
    error_fastExtremely_frames=np.zeros((len(fastExtremely_frames),3))
    error_fastExtremely_frames[:,0]=fastExtremely_frames
    error_fastExtremely_frames[:,1]=error[2]
    error_fastExtremely_frames[:,2]=error[5]
    error_fastExtremely_frames=list(error_fastExtremely_frames)
    
    
        
        
        
        
        