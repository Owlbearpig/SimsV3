# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:40:19 2020

@author: talebf
"""

import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path
import csv 
now = datetime.datetime.now()
date = now.strftime("%Y-%m-%d")
filename2="square_Prause_01"
z1=0.2
E=0.8
E1=0.02
speed=1200
# numepoint=100
# theta = np.linspace(0, 2*np.pi, numepoint)
radius1=25
delt=0.32
x=100
y=80
layerB =10

numberx=1
numbery=1
x0=0
y0=0
wx=radius1*2+1
wy=radius1*2+1
z1=0.20
figure, axes = plt.subplots(1)
with open(filename2+'.gcode','a',newline='') as csvfile:
              spamwriter=csv.writer(csvfile,delimiter=' ')
              # spamwriter.writerow(["T0"])
              # spamwriter.writerow(["M82"])
              spamwriter.writerow(["G92", 'E0'])
              spamwriter.writerow(["M109", 'S220'])
              spamwriter.writerow(["G1", 'F15000', 'X9','Y6','Z2'])
              spamwriter.writerow(["G280"])

              spamwriter.writerow(["M107"])
              spamwriter.writerow(["M204", 'S625'])
              spamwriter.writerow(["M205", 'X6','Y6'])
              spamwriter.writerow(["G1", 'F600', 'Z2.27'])
              spamwriter.writerow(['G1','F4285.7','X'+str(x0+radius1),'Y'+str(y0+radius1),'Z'+str(2.27)])
              spamwriter.writerow(["M204", 'S500'])
              spamwriter.writerow(["M205", 'X5','Y5'])
              spamwriter.writerow(["G1" ,"E-0.80000" ,"F2100.00000"])
              spamwriter.writerow(["G1" ,"Z0.600" ,"F10800.000"])
              
              
# =============================================================================
# Base printing              
# =============================================================================
k=0              
for k in range(layerB):
    if k==0:
        E1=0.028
    else:
        E1=0.028
        
    delt=0.32
    x1=x-wx/2#+(k)*(0.1) 
    x2=x+wx*numberx -wx/2
    y1=y+(wy*numbery)-wy/2 
    y2=y+-wy/2 
    h=z1+0.2*k
    speed=1200
    # speed=speed+(k%2)*(speed/4)
    x1=x1+2*delt
    y1=y1-2*delt
    x2=x2-2*delt
    y2=y2+2*delt

   
         
    with open(filename2+'.gcode','a',newline='') as csvfile:  
        spamwriter=csv.writer(csvfile,delimiter=' ')
        if k==0:
            
            spamwriter.writerow(['G1','F4285.7','X'+str(x1),'Y'+str(y1),'Z'+str(np.round(h,2)),'E'+str(np.round(-E,5))])
            spamwriter.writerow([";layer:"+str(k)])
            spamwriter.writerow(["M204", 'S500'])
            spamwriter.writerow(["M205", 'X5','Y5'])
            spamwriter.writerow(["M140", 'S100'])
            spamwriter.writerow(["M140", 'S110'])
            spamwriter.writerow(["G1", 'F1200', 'E'+str(np.round(E,5))])
        else:
            spamwriter.writerow(['G1','F4285.7','X'+str(x1),'Y'+str(y1),'Z'+str(np.round(h,2)),'E'+str(np.round(-E,5))])
            spamwriter.writerow([";layer:"+str(k)])
            # spamwriter.writerow(["M140", 'S110'])
            spamwriter.writerow(["M106", 'S85']) 
            spamwriter.writerow(["M204", 'S750'])
            spamwriter.writerow(["M205", 'X7.5','Y7.5']) 
            spamwriter.writerow(["G92"," E0.0"])
            
        spamwriter.writerow([";Mesh "])
        spamwriter.writerow(["G1", 'F'+str(speed), 'Z'+str(np.round(h,2))])
        
        # spamwriter.writerow(["G1", 'F1200', 'E'+str(np.round(-E,5))])
        
        
    for i in range(3):
        # plt.axes()
        line=plt.Line2D((x1,x2),(y1,y1),lw=0.36) 
        line2=plt.Line2D((x2,x2),(y1,y2),lw=0.36)  
        line3=plt.Line2D((x2,x1),(y2,y2),lw=0.36)
        line4=plt.Line2D((x1,x1),(y2,y1),lw=0.36) 
        plt.gca().add_line(line)
        plt.gca().add_line(line2)
        plt.gca().add_line(line3)
        plt.gca().add_line(line4)
    
        
        with open(filename2+'.gcode','a',newline='') as csvfile:
                  spamwriter=csv.writer(csvfile,delimiter=' ')
                  # spamwriter.writerow([f";Wall {i}"])
                  # spamwriter.writerow(["G1", 'F'+str(2*speed), 'Z'+str(np.round(h,2))])
                  # spamwriter.writerow(["G1", 'F1500', 'E'+str(np.round(E,5))])
                  E=(abs(x2-x1))*E1
                  line=['G1','F'+str(speed),'X'+str(np.round(x2,2)),'Y'+str(np.round(y1,2)),'E'+str(np.round(E,5))]
                  spamwriter.writerow(line)
                  E=(abs(y2-y1)*E1)
                  line2=['G1','X'+str(np.round(x2,2)),'Y'+str(np.round(y2,2)),'E'+str(np.round(E,5))]
                  spamwriter.writerow(line2)
                  E=(abs(x2-x1)*E1)
                  line3=['G1','X'+str(np.round(x1,2)),'Y'+str(np.round(y2,2)),'E'+str(np.round(E,5))]
                  spamwriter.writerow(line3)
                  E=(abs(y2-y1)*E1)
                  line4=['G1','X'+str(np.round(x1,2)),'Y'+str(np.round(y1,2)),'E'+str(np.round(E,5))]
                  spamwriter.writerow(line4)
                  if i==2:
                      spamwriter.writerow(["M204", 'S625'])
                      spamwriter.writerow(["M205", 'X6','Y6'])
                      spamwriter.writerow(['G1','F4285.7','X'+str(np.round(x1+3*delt,2)),'Y'+str(np.round(y1-3*delt,2)),'E0.0'])
                      spamwriter.writerow(["M204", 'S500'])
                      spamwriter.writerow(["M205", 'X5','Y5'])
                            
                  else:
                      spamwriter.writerow(["M204", 'S625'])
                      spamwriter.writerow(["M205", 'X6','Y6'])
                      y1+=delt
                      y2-=delt
                      x1-=delt
                      x2+=delt
                      spamwriter.writerow(['G1','F4285.7','X'+str(np.round(x1,2)),'Y'+str(np.round(y1,2)),'E0.0'])
                      spamwriter.writerow(["M204", 'S500'])
                      spamwriter.writerow(["M205", 'X5','Y5'])
                      spamwriter.writerow(["G1", "F1200"])
                  
                  
    x1=x1+3*delt
    y1=y1-3*delt
    x2=x2-3*delt
    y2=y2+3*delt
        
    line=plt.Line2D((x1,x2),(y1,y1),lw=0.36,color="k") 
    line2=plt.Line2D((x2,x2),(y1,y2),lw=0.36,color="k")  
    line3=plt.Line2D((x2,x1),(y2,y2),lw=0.36,color="k")          
    line4=plt.Line2D((x1,x1),(y2,y1),lw=0.36,color="k")
    plt.gca().add_line(line)
    plt.gca().add_line(line2)
    plt.gca().add_line(line3)
    plt.gca().add_line(line4)          
    with open(filename2+'.gcode','a',newline='') as csvfile:
        
        spamwriter=csv.writer(csvfile,delimiter=' ')
        spamwriter.writerow([";Skine"])
        spamwriter.writerow(['G1','F4285.7','X'+str(np.round(x1,2)),'Y'+str(np.round(y1,2)),'E'+str(np.round(-E,5))])
        spamwriter.writerow(["M204", 'S500'])
        spamwriter.writerow(["M205", 'X5','Y5'])
        spamwriter.writerow(['G1','Z'+str(np.round(h,2)),'E0.8'])
        E=(abs(x2-x1))*E1
        line=['G1','F'+str(speed),'X'+str(np.round(x2,2)),'Y'+str(np.round(y1,2)),'E'+str(np.round(E,5))]
        spamwriter.writerow(line)
        E=(abs(y2-y1)*E1)
        line2=['G1','X'+str(np.round(x2,2)),'Y'+str(np.round(y2,2)),'E'+str(np.round(E,5))]
        spamwriter.writerow(line2)
        E=(abs(x2-x1)*E1)
        line3=['G1','X'+str(np.round(x1,2)),'Y'+str(np.round(y2,2)),'E'+str(np.round(E,5))]
        spamwriter.writerow(line3)
        E=(abs(y2-y1)*E1)
        line4=['G1','X'+str(np.round(x1,2)),'Y'+str(np.round(y1,2)),'E'+str(np.round(E,5))]
        spamwriter.writerow(line4)
        spamwriter.writerow(["M204", 'S625'])
        spamwriter.writerow(["M205", 'X6','Y6'])


    
    print(x1,x2,y1,y2)
    delt2=0.30
    x1=x1+delt/4
    y1=y1-delt/4
    x2=x2-delt/4
    y2=y2+delt/4
    i=0
    
    lines=abs(int((y2-y1)/(delt2)))
    # print("odd k:",k, lines)
    for i in range(lines):
        # print("i1k",i)
        line5=plt.Line2D((x1,x2),(y1,y1),lw=0.36,color="r")
        plt.gca().add_line(line5)
        line6=plt.Line2D((x2,x2),(y1,y1-delt2),lw=0.36,color="b")
        plt.gca().add_line(line6)
        line7=plt.Line2D((x2,x1),(y1-delt2,y1-delt2),lw=0.36,color="b")
        plt.gca().add_line(line7)
        line8=plt.Line2D((x1,x1),(y1-delt2,y1-2*delt2),lw=0.36,color="r")
        plt.gca().add_line(line8)
        # print("111")
        with open(filename2+'.gcode','a',newline='') as csvfile:
              spamwriter=csv.writer(csvfile,delimiter=' ')
              spamwriter.writerow(["G1", "E0.08"])
              #line=['G0','X'+str(x1),'Y'+str(y1),'Z'+str(0)]
              #spamwriter.writerow(line)
              E=(abs(x2-x1))*E1
              line1=['G1','F'+str(speed),'X'+str(np.round(x2,2)),'Y'+str(np.round(y1,2)),'E'+str(np.round(E,5))]
              spamwriter.writerow(line1)
              spamwriter.writerow(["M204", 'S625'])
              spamwriter.writerow(["M205", 'X6','Y6'])
              line2=['G1','F4285.7','X'+str(np.round(x2,2)),'Y'+str(np.round(y1-delt2,2)),'E'+str(np.round(0,5))]
              spamwriter.writerow(line2)
              #spamwriter.writerow(["M204", 'S500'])
              #spamwriter.writerow(["M205", 'X5','Y5'])
              #E=E+(abs(x2-x1))*E1
              line3=['G1','F4200','X'+str(np.round(x1,2)),'Y'+str(np.round(y1-delt2,2)),'E'+str(np.round(0,5))]
              spamwriter.writerow(line3)
              spamwriter.writerow(["M204", 'S500'])
              spamwriter.writerow(["M205", 'X5','Y5'])
              #line4=['G0','F4285.7','X'+str(np.round(x1,2)),'Y'+str(np.round(y1+2*delt,2))]
              #spamwriter.writerow(line4)
              #spamwriter.writerow(["M204", 'S500'])
              #spamwriter.writerow(["M205", 'X5','Y5'])
        y1=y1-delt2
           
    
              
                
        
    plt.axis('scaled')    
    
    plt.title('Circle using Parametric Equation')
    plt.show()
with open(filename2+'.gcode','a',newline='') as csvfile:
              spamwriter=csv.writer(csvfile,delimiter=' ')
              spamwriter.writerow(["G1",'F1500', 'E'+str(E-30)])
              spamwriter.writerow(["M140",'S0'])
              spamwriter.writerow(["M204", 'S3000'])
              spamwriter.writerow(["M205",  'X20','Y20'])
              spamwriter.writerow(["M107"])
              spamwriter.writerow(["G91"])
              spamwriter.writerow(['G1','F15000','X8.0','Z10.5','E-4.5'])
              spamwriter.writerow(['G1','F10000','Z11.5','E4.5'])
              spamwriter.writerow(["G90"])
              spamwriter.writerow(["M82"])
              spamwriter.writerow(["M104", 'S0'])
              spamwriter.writerow(["M104", 'T1','S0'])
              spamwriter.writerow([';End of Gcode'])
        