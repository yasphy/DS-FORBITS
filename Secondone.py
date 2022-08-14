import cv2 as cv
import numpy as np
import streamlit as st
import funct
from urllib.request import urlopen, Request
from skimage.color import convert_colorspace
import work
import torch.nn as nn
st.title("WaterMark Maker and Equation solver")
option = st.radio("Select what you want", ("WaterMark",
                  "Fluid Mechanics", "Image processing","Neural Networks"))
        
if (option == "WaterMark"):
    file1 = st.file_uploader("Select the files in  such a way that second file is watermark file", type=[
                             "jpg", 'png'], accept_multiple_files=True)
    k = []
    for uploaded_file in file1:
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv.imdecode(file_bytes, 1)
        # Now do something with the image! For example, let's display it:
        k.append(opencv_image)
    if len(k) > 1:
        for i in range(2):
            st.image(k[i], channels="BGR")
    if (file1 is not None) and (len(k) >= 2):
        k1 = k[0].shape
        k12 = k[1].shape
        st.write("shapes are ", str(k1), "and", str(k12))
        if (k1 != k12):
            h, w, c = k[0].shape
            k[1] = cv.resize(k[1], (w, h))
            k13 = k[0].shape
            k14 = k[1].shape
            st.write("shapes are ", str(k13), "and", str(k14))
            o = st.number_input("Choose a number between 1 and 0")
            dst = cv.addWeighted(k[1], o, k[0], 1-o, 0)
            st.image(dst, channels='BGR')
elif(option == "Neural Networks"):
    f1 = st.file_uploader("Select your files", type=["jpg", 'png', "jpeg"],accept_multiple_files=True)
    dirw=st.text_input("enter the directory path")
    try:
        if f1 is not None:
            if len(f1)>0:
                lo=st.slider("Choose file",min_value=0,max_value=len(f1))
            uploaded_file=f1[lo]
            bytes_data = uploaded_file.read()
            st.write("filename:", uploaded_file.name)
            #st.write(bytes_data)
            #file_bytes = np.asarray(bytearray(f1.read()), dtype=np.uint8)
            Mk=work.datasetmaker(uploaded_file.name)
            st.image((Mk[0]).permute(1,2,0).numpy())
            ch1=st.selectbox("Choose type", options=("Int","Tuple"))
            ch2=st.selectbox("Choose layer",options=("Conv2d","LazyConv2D","Maxpool2d","Avgpool2d","LLPool2d","ReplicationPad","ReflectionPad","Sequence"))
            if ch1=="Tuple":
                        if ch2=="Conv2d":
                            kernel_size1 = st.slider("Select kernel size1", min_value=1,
                               max_value=999, step=1)
                            stride1 = st.slider("Select stride1", min_value=1,
                               max_value=999, step=1)
                            padding1 = st.slider("Select padding1", min_value=0, max_value=100)
                            kernel_size2 = st.slider("Select kernel size2", min_value=1,
                               max_value=127, step=1)
                            stride2 = st.slider("Select stride2", min_value=1,
                               max_value=255, step=1)
                            padding2 = st.slider("Select padding2", min_value=1, max_value=100)
                            m=nn.Conv2d(3, 3, kernel_size=(kernel_size1,kernel_size2), stride=(stride1,stride2), padding=(padding1,padding2), dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
                            k=(m(Mk).detach().numpy())[0]
                            #k=work.image(k)
                            st.image(work.image(k),clamp=True,channels="BGR")
                        elif ch2=="LLPool2d":
                            kernel_size1 = st.slider("Select kernel size1", min_value=1,
                               max_value=999, step=1)
                            stride1 = st.slider("Select stride1", min_value=1,
                               max_value=999, step=1)
                            kernel_size2 = st.slider("Select kernel size2", min_value=1,
                               max_value=127, step=1)
                            stride2 = st.slider("Select stride2", min_value=1,
                               max_value=255, step=1)
                            m=nn.LPPool2d(kernel_size=(kernel_size1,kernel_size2), stride=(stride1,stride2), ceil_mode=False)
                            k=(m(Mk).detach().numpy())[0]
                            st.image(work.image(k),clamp=True,channels="BGR")
                        elif ch2=="Maxpool2d":
                            kernel_size1 = st.slider("Select kernel size1", min_value=1,
                               max_value=999, step=1)
                            stride1 = st.slider("Select stride1", min_value=1,
                               max_value=999, step=1)
                            padding1 = st.slider("Select padding1", min_value=0, max_value=100)
                            kernel_size2 = st.slider("Select kernel size2", min_value=1,
                               max_value=127, step=1)
                            stride2 = st.slider("Select stride2", min_value=1,
                               max_value=255, step=1)
                            padding2 = st.slider("Select padding2", min_value=1, max_value=100)
                            m=nn.MaxPool2d( kernel_size=(kernel_size1,kernel_size2), stride=(stride1,stride2), padding=(padding1,padding2), dilation=1)
                            k=(m(Mk).detach().numpy())[0]
                            st.image(work.image(k),clamp=True,channels="BGR")
                        elif ch2=="Avgpool2d":
                            kernel_size1 = st.slider("Select kernel size1", min_value=1,
                               max_value=999, step=1)
                            stride1 = st.slider("Select stride1", min_value=1,
                               max_value=999, step=1)
                            padding1 = st.slider("Select padding1", min_value=0, max_value=100)
                            kernel_size2 = st.slider("Select kernel size2", min_value=1,
                               max_value=127, step=1)
                            stride2 = st.slider("Select stride2", min_value=1,
                               max_value=255, step=1)
                            padding2 = st.slider("Select padding2", min_value=1, max_value=100)
                            m=nn.AvgPool2d(kernel_size=(kernel_size1,kernel_size2), stride=(stride1,stride2), padding=(padding1,padding2), ceil_mode=False, count_include_pad=True, divisor_override=None)
                            k=(m(Mk).detach().numpy())[0]
                            st.image(work.image(k),clamp=True,channels="BGR")
                        elif ch2=="ReplicationPad":
                            padding1=st.slider("Select padding1",min_value=1, max_value=10000)
                            padding2=st.slider("Select padding2",min_value=1, max_value=10000)
                            padding3=st.slider("Select padding3",min_value=1, max_value=10000)
                            padding4=st.slider("Select padding4",min_value=1, max_value=10000)
                            m=nn.ReplicationPad2d((padding1,padding2,padding3,padding4))
                            k=(m(Mk).detach().numpy())[0]
                            st.image(work.image(k),clamp=True,channels="BGR")
                        elif ch2=="ReflectionPad":
                            padding1=st.slider("Select padding1",min_value=1, max_value=1000)
                            padding2=st.slider("Select padding2",min_value=1, max_value=1000)
                            padding3=st.slider("Select padding3",min_value=1, max_value=1000)
                            padding4=st.slider("Select padding4",min_value=1, max_value=1000)
                            m=nn.ReflectionPad2d((padding1,padding2,padding3,padding4))
                            k=(m(Mk).detach().numpy())[0]
                            st.image(work.image(k),clamp=True,channels="BGR")
                        elif ch2=="Sequence":
                            num=st.number_input("Enter number of layers",min_value=1)
                            layers=[]
                            for i in range(num):
                                st.write("Choose"+str(i)+"layer")
                                ch2=st.selectbox("Choose layer",options=("Conv2d","LazyConv2D","Maxpool2d","Avgpool2d","LLPool2d","ReplicationPad","ReflectionPad"))
                                if ch2=="Conv2d":
                                    kernel_size1 = st.slider("Select kernel size1", min_value=1,
                                       max_value=999, step=1)
                                    stride1 = st.slider("Select stride1", min_value=1,
                                       max_value=999, step=1)
                                    padding1 = st.slider("Select padding1", min_value=0, max_value=100)
                                    kernel_size2 = st.slider("Select kernel size2", min_value=1,
                                       max_value=127, step=1)
                                    stride2 = st.slider("Select stride2", min_value=1,
                                       max_value=255, step=1)
                                    padding2 = st.slider("Select padding2", min_value=1, max_value=100)
                                    m=nn.Conv2d(3, 3, kernel_size=(kernel_size1,kernel_size2), stride=(stride1,stride2), padding=(padding1,padding2), dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
                                    layers.append(m)
                                elif ch2=="LLPool2d":
                                    kernel_size1 = st.slider("Select kernel size1", min_value=1,
                                       max_value=999, step=1)
                                    stride1 = st.slider("Select stride1", min_value=1,
                                       max_value=999, step=1)
                                    kernel_size2 = st.slider("Select kernel size2", min_value=1,
                                       max_value=127, step=1)
                                    stride2 = st.slider("Select stride2", min_value=1,
                                       max_value=255, step=1)
                                    m=nn.LPPool2d(kernel_size=(kernel_size1,kernel_size2), stride=(stride1,stride2), ceil_mode=False)
                                    layers.append(m)
                                elif ch2=="Maxpool2d":
                                    kernel_size1 = st.slider("Select kernel size1", min_value=1,
                                       max_value=999, step=1)
                                    stride1 = st.slider("Select stride1", min_value=1,
                                       max_value=999, step=1)
                                    padding1 = st.slider("Select padding1", min_value=0, max_value=100)
                                    kernel_size2 = st.slider("Select kernel size2", min_value=1,
                                       max_value=127, step=1)
                                    stride2 = st.slider("Select stride2", min_value=1,
                                       max_value=255, step=1)
                                    padding2 = st.slider("Select padding2", min_value=1, max_value=100)
                                    m=nn.MaxPool2d( kernel_size=(kernel_size1,kernel_size2), stride=(stride1,stride2), padding=(padding1,padding2), dilation=1)
                                    layers.append(m)
                                elif ch2=="Avgpool2d":
                                    kernel_size1 = st.slider("Select kernel size1", min_value=1,
                                       max_value=999, step=1)
                                    stride1 = st.slider("Select stride1", min_value=1,
                                       max_value=999, step=1)
                                    padding1 = st.slider("Select padding1", min_value=0, max_value=100)
                                    kernel_size2 = st.slider("Select kernel size2", min_value=1,
                                       max_value=127, step=1)
                                    stride2 = st.slider("Select stride2", min_value=1,
                                       max_value=255, step=1)
                                    padding2 = st.slider("Select padding2", min_value=1, max_value=100)
                                    m=nn.AvgPool2d(kernel_size=(kernel_size1,kernel_size2), stride=(stride1,stride2), padding=(padding1,padding2), ceil_mode=False, count_include_pad=True, divisor_override=None)
                                    layers.append(m)
                                elif ch2=="ReplicationPad":
                                    padding1=st.slider("Select padding1",min_value=1, max_value=100)
                                    padding2=st.slider("Select padding2",min_value=1, max_value=100)
                                    padding3=st.slider("Select padding3",min_value=1, max_value=100)
                                    padding4=st.slider("Select padding4",min_value=1, max_value=100)
                                    m=nn.ReplicationPad2d((padding1,padding2,padding3,padding4))
                                    layers.append(m)
                                elif ch2=="ReflectionPad":
                                    padding1=st.slider("Select padding1",min_value=1, max_value=1000)
                                    padding2=st.slider("Select padding2",min_value=1, max_value=1000)
                                    padding3=st.slider("Select padding3",min_value=1, max_value=1000)
                                    padding4=st.slider("Select padding4",min_value=1, max_value=1000)
                                    m=nn.ReflectionPad2d((padding1,padding2,padding3,padding4))
                                    layers.append(m)
                                else:
                                    m=nn.LazyConv2d(3, kernel_size=(kernel_size1,kernel_size2), stride=(stride1,stride2), padding=(padding1,padding2), dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
                                    layers.append(m)
                                seq=nn.Sequential(*layers)
                                k=(seq(Mk).detach().numpy())[0]
                                st.image(work.image(k),clamp=True,channels="BGR")
                        else:
                            m=nn.LazyConv2d(3, kernel_size=(kernel_size1,kernel_size2), stride=(stride1,stride2), padding=(padding1,padding2), dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
                            k=(m(Mk).detach().numpy())[0]
                            st.image(work.image(k),clamp=True,channels="BGR")
            else:
                if ch2=="Conv2d":
                            kernel_size = st.slider("Select kernel size", min_value=1,
                                   max_value=100, step=1)
                            stride = st.slider("Select stride", min_value=1,
                                   max_value=100, step=1)
                            padding = st.slider("Select padding", min_value=0, max_value=100)
                            dilation=st.slider("Select Dilation",min_value=1, max_value=100)
                            groups=st.slider("Select Groups(not useful in pooling layers)",min_value=1, max_value=999)
                            m=nn.Conv2d(3, 3, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
                            k=(m(Mk).detach().numpy())[0]
                            k=work.image(k)
                            st.image(k,clamp=True,channels="BGR")
                elif ch2=="Avgpool2d":
                            kernel_size = st.slider("Select kernel size", min_value=1,
                                   max_value=100, step=1)
                            stride = st.slider("Select stride", min_value=1,
                                   max_value=100, step=1)
                            padding = st.slider("Select padding", min_value=0, max_value=100)
                            m=nn.AvgPool2d(kernel_size,stride, padding, ceil_mode=False, count_include_pad=True, divisor_override=None)
                            k=(m(Mk).detach().numpy())[0]
                            st.image(work.image(k),clamp=True,channels="BGR")
                elif ch2=="Maxpool2d":
                            kernel_size = st.slider("Select kernel size", min_value=1,
                                   max_value=100, step=1)
                            stride = st.slider("Select stride", min_value=1,
                                   max_value=100, step=1)
                            padding = st.slider("Select padding", min_value=0, max_value=100)
                            m=nn.MaxPool2d(kernel_size, stride, padding, dilation=1)
                            k=(m(Mk).detach().numpy())[0]
                            st.image(work.image(k),clamp=True,channels="BGR")
                elif ch2=="ReplicationPad":
                    padding=st.slider("Select padding",min_value=1, max_value=1000)
                    m=nn.ReplicationPad2d(padding)
                    k=(m(Mk).detach().numpy())[0]
                    st.image(work.image(k),clamp=True,channels="BGR")
                elif ch2=="ReflectionPad":
                    padding=st.slider("Select padding",min_value=1, max_value=1000)
                    m=nn.ReflectionPad2d(padding)
                    k=(m(Mk).detach().numpy())[0]
                    st.image(work.image(k),clamp=True,channels="BGR")
                elif ch2=="LLPool2d":
                            norm_type=st.number_input("Enter norm",0,10)
                            kernel_size = st.slider("Select kernel size", min_value=1,
                                   max_value=100, step=1)
                            stride = st.slider("Select stride", min_value=1,
                                   max_value=100, step=1)
                            padding = st.slider("Select padding", min_value=0, max_value=100)
                            m=nn.LPPool2d(norm_type,kernel_size, stride, ceil_mode=False)
                            k=(m(Mk).detach().numpy())[0]
                            st.image(work.image(k),clamp=True,channels="BGR")
                elif ch2=="Sequence":
                            layer1=[]
                            layer2=[]
                            ch2=st.selectbox("Choose 1st layer",options=("Conv2d","LazyConv2D","Maxpool2d","Avgpool2d","LLPool2d","ReplicationPad","ReflectionPad"))
                            if ch2=="Conv2d":
                                        kernel_size = st.slider("Select kernel size", min_value=1,
                                               max_value=100, step=1)
                                        stride = st.slider("Select stride", min_value=1,
                                               max_value=100, step=1)
                                        padding = st.slider("Select padding", min_value=0, max_value=100)
                                        dilation=st.slider("Select Dilation",min_value=1, max_value=100)
                                        groups=st.slider("Select Groups(not useful in pooling layers)",min_value=1, max_value=999)
                                        m=nn.Conv2d(3, 3, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
                                        layer1=m
                                        ch3=st.selectbox("Choose 2nd layer",options=("Conv2d","LazyConv2D","Maxpool2d","Avgpool2d","LLPool2d","ReplicationPad","ReflectionPad"))
                                        if ch3=="Conv2d":
                                                    kernel_size = st.slider("Select kernel size", min_value=1,
                                                           max_value=100, step=1)
                                                    stride = st.slider("Select stride", min_value=1,
                                                           max_value=100, step=1)
                                                    padding = st.slider("Select padding", min_value=0, max_value=100)
                                                    dilation=st.slider("Select Dilation",min_value=1, max_value=100)
                                                    groups=st.slider("Select Groups(not useful in pooling layers)",min_value=1, max_value=999)
                                                    m=nn.Conv2d(3, 3, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
                                                    layer2=m
                                        elif ch3=="Avgpool2d":
                                                    kernel_size = st.slider("Select kernel size", min_value=1,
                                                           max_value=100, step=1)
                                                    stride = st.slider("Select stride", min_value=1,
                                                           max_value=100, step=1)
                                                    padding = st.slider("Select padding", min_value=0, max_value=100)
                                                    m=nn.AvgPool2d(kernel_size,stride, padding, ceil_mode=False, count_include_pad=True, divisor_override=None)
                                                    layer2=m
                                        elif ch3=="Maxpool2d":
                                                    kernel_size = st.slider("Select kernel size", min_value=1,
                                                           max_value=100, step=1)
                                                    stride = st.slider("Select stride", min_value=1,
                                                           max_value=100, step=1)
                                                    padding = st.slider("Select padding", min_value=0, max_value=100)
                                                    m=nn.MaxPool2d(kernel_size, stride, padding, dilation=1)
                                                    layer2=m
                                        elif ch3=="ReplicationPad":
                                            padding=st.slider("Select padding",min_value=1, max_value=1000)
                                            m=nn.ReplicationPad2d(padding)
                                            layer2=m
                                        elif ch3=="ReflectionPad":
                                            padding=st.slider("Select padding",min_value=1, max_value=1000)
                                            m=nn.ReflectionPad2d(padding)
                                            layer2=m
                                        elif ch3=="LLPool2d":
                                                    norm_type=st.number_input("Enter norm",0,10)
                                                    kernel_size = st.slider("Select kernel size", min_value=1,
                                                           max_value=100, step=1)
                                                    stride = st.slider("Select stride", min_value=1,
                                                           max_value=100, step=1)
                                                    padding = st.slider("Select padding", min_value=0, max_value=100)
                                                    m=nn.LPPool2d(norm_type,kernel_size, stride, ceil_mode=False)
                                                    layer2=m
                                        else:
                                                kernel_size = st.slider("Select kernel size", min_value=1,
                                                           max_value=100, step=1)
                                                stride = st.slider("Select stride", min_value=1,
                                                           max_value=100, step=1)
                                                padding = st.slider("Select padding", min_value=0, max_value=100)
                                                m=nn.LazyConv2d(3,kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
                                                layer2=m
                                        t=nn.Sequential(layer1,layer2)
                                        k=(t(Mk).detach().numpy())[0]
                                        st.image(work.image(k),clamp=True,channels="BGR")
                            elif ch2=="Avgpool2d":
                                        kernel_size = st.slider("Select kernel size", min_value=1,
                                               max_value=100, step=1)
                                        stride = st.slider("Select stride", min_value=1,
                                               max_value=100, step=1)
                                        padding = st.slider("Select padding", min_value=0, max_value=100)
                                        m=nn.AvgPool2d(kernel_size,stride, padding, ceil_mode=False, count_include_pad=True, divisor_override=None)
                                        layer1=m
                                        ch3=st.selectbox("Choose 2nd layer",options=("Conv2d","LazyConv2D","Maxpool2d","Avgpool2d","LLPool2d","ReplicationPad","ReflectionPad"))
                                        if ch3=="Conv2d":
                                                    kernel_size = st.slider("Select kernel size", min_value=1,
                                                           max_value=100, step=1)
                                                    stride = st.slider("Select stride", min_value=1,
                                                           max_value=100, step=1)
                                                    padding = st.slider("Select padding", min_value=0, max_value=100)
                                                    dilation=st.slider("Select Dilation",min_value=1, max_value=100)
                                                    groups=st.slider("Select Groups(not useful in pooling layers)",min_value=1, max_value=999)
                                                    m=nn.Conv2d(3, 3, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
                                                    layer2=m
                                        elif ch3=="Avgpool2d":
                                                    kernel_size = st.slider("Select kernel size", min_value=1,
                                                           max_value=100, step=1)
                                                    stride = st.slider("Select stride", min_value=1,
                                                           max_value=100, step=1)
                                                    padding = st.slider("Select padding", min_value=0, max_value=100)
                                                    m=nn.AvgPool2d(kernel_size,stride, padding, ceil_mode=False, count_include_pad=True, divisor_override=None)
                                                    layer2=m
                                        elif ch3=="Maxpool2d":
                                                    kernel_size = st.slider("Select kernel size", min_value=1,
                                                           max_value=100, step=1)
                                                    stride = st.slider("Select stride", min_value=1,
                                                           max_value=100, step=1)
                                                    padding = st.slider("Select padding", min_value=0, max_value=100)
                                                    m=nn.MaxPool2d(kernel_size, stride, padding, dilation=1)
                                                    layer2=m
                                        elif ch3=="ReplicationPad":
                                            padding=st.slider("Select padding",min_value=1, max_value=1000)
                                            m=nn.ReplicationPad2d(padding)
                                            layer2=m
                                        elif ch3=="ReflectionPad":
                                            padding=st.slider("Select padding",min_value=1, max_value=1000)
                                            m=nn.ReflectionPad2d(padding)
                                            layer2=m
                                        elif ch3=="LLPool2d":
                                                    norm_type=st.number_input("Enter norm",0,10)
                                                    kernel_size = st.slider("Select kernel size", min_value=1,
                                                           max_value=100, step=1)
                                                    stride = st.slider("Select stride", min_value=1,
                                                           max_value=100, step=1)
                                                    padding = st.slider("Select padding", min_value=0, max_value=100)
                                                    m=nn.LPPool2d(norm_type,kernel_size, stride, ceil_mode=False)
                                                    layer2=m
                                        else:
                                                kernel_size = st.slider("Select kernel size", min_value=1,
                                                           max_value=100, step=1)
                                                stride = st.slider("Select stride", min_value=1,
                                                           max_value=100, step=1)
                                                padding = st.slider("Select padding", min_value=0, max_value=100)
                                                m=nn.LazyConv2d(3,kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
                                                layer2=m
                                        t=nn.Sequential(layer1,layer2)
                                        k=(t(Mk).detach().numpy())[0]
                                        st.image(work.image(k),clamp=True,channels="BGR")
                            elif ch2=="Maxpool2d":
                                        kernel_size = st.slider("Select kernel size", min_value=1,
                                               max_value=100, step=1)
                                        stride = st.slider("Select stride", min_value=1,
                                               max_value=100, step=1)
                                        padding = st.slider("Select padding", min_value=0, max_value=100)
                                        m=nn.MaxPool2d(kernel_size, stride, padding, dilation=1)
                                        layer1=m
                                        ch3=st.selectbox("Choose 2nd layer",options=("Conv2d","LazyConv2D","Maxpool2d","Avgpool2d","LLPool2d","ReplicationPad","ReflectionPad"))
                                        if ch3=="Conv2d":
                                                    kernel_size = st.slider("Select kernel size", min_value=1,
                                                           max_value=100, step=1)
                                                    stride = st.slider("Select stride", min_value=1,
                                                           max_value=100, step=1)
                                                    padding = st.slider("Select padding", min_value=0, max_value=100)
                                                    dilation=st.slider("Select Dilation",min_value=1, max_value=100)
                                                    groups=st.slider("Select Groups(not useful in pooling layers)",min_value=1, max_value=999)
                                                    m=nn.Conv2d(3, 3, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
                                                    layer2=m
                                        elif ch3=="Avgpool2d":
                                                    kernel_size = st.slider("Select kernel size", min_value=1,
                                                           max_value=100, step=1)
                                                    stride = st.slider("Select stride", min_value=1,
                                                           max_value=100, step=1)
                                                    padding = st.slider("Select padding", min_value=0, max_value=100)
                                                    m=nn.AvgPool2d(kernel_size,stride, padding, ceil_mode=False, count_include_pad=True, divisor_override=None)
                                                    layer2=m
                                        elif ch3=="Maxpool2d":
                                                    kernel_size = st.slider("Select kernel size", min_value=1,
                                                           max_value=100, step=1)
                                                    stride = st.slider("Select stride", min_value=1,
                                                           max_value=100, step=1)
                                                    padding = st.slider("Select padding", min_value=0, max_value=100)
                                                    m=nn.MaxPool2d(kernel_size, stride, padding, dilation=1)
                                                    layer2=m
                                        elif ch3=="ReplicationPad":
                                            padding=st.slider("Select padding",min_value=1, max_value=1000)
                                            m=nn.ReplicationPad2d(padding)
                                            layer2=m
                                        elif ch3=="ReflectionPad":
                                            padding=st.slider("Select padding",min_value=1, max_value=1000)
                                            m=nn.ReflectionPad2d(padding)
                                            layer2=m
                                        elif ch3=="LLPool2d":
                                                    norm_type=st.number_input("Enter norm",0,10)
                                                    kernel_size = st.slider("Select kernel size", min_value=1,
                                                           max_value=100, step=1)
                                                    stride = st.slider("Select stride", min_value=1,
                                                           max_value=100, step=1)
                                                    padding = st.slider("Select padding", min_value=0, max_value=100)
                                                    m=nn.LPPool2d(norm_type,kernel_size, stride, ceil_mode=False)
                                                    layer2=m
                                        else:
                                                kernel_size = st.slider("Select kernel size", min_value=1,
                                                           max_value=100, step=1)
                                                stride = st.slider("Select stride", min_value=1,
                                                           max_value=100, step=1)
                                                padding = st.slider("Select padding", min_value=0, max_value=100)
                                                m=nn.LazyConv2d(3,kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
                                                layer2=m
                                        t=nn.Sequential(layer1,layer2)
                                        k=(t(Mk).detach().numpy())[0]
                                        st.image(work.image(k),clamp=True,channels="BGR")
                            elif ch2=="ReplicationPad":
                                padding=st.slider("Select padding",min_value=1, max_value=1000)
                                m=nn.ReplicationPad2d(padding)
                                layer1=m
                                ch3=st.selectbox("Choose 2nd layer",options=("Conv2d","LazyConv2D","Maxpool2d","Avgpool2d","LLPool2d","ReplicationPad","ReflectionPad"))
                                if ch3=="Conv2d":
                                    kernel_size = st.slider("Select kernel size", min_value=1,
                                           max_value=100, step=1)
                                    stride = st.slider("Select stride", min_value=1,
                                           max_value=100, step=1)
                                    padding = st.slider("Select padding", min_value=0, max_value=100)
                                    dilation=st.slider("Select Dilation",min_value=1, max_value=100)
                                    groups=st.slider("Select Groups(not useful in pooling layers)",min_value=1, max_value=999)
                                    m=nn.Conv2d(3, 3, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
                                    layer2=m
                                elif ch3=="Avgpool2d":
                                            kernel_size = st.slider("Select kernel size", min_value=1,
                                                   max_value=100, step=1)
                                            stride = st.slider("Select stride", min_value=1,
                                                   max_value=100, step=1)
                                            padding = st.slider("Select padding", min_value=0, max_value=100)
                                            m=nn.AvgPool2d(kernel_size,stride, padding, ceil_mode=False, count_include_pad=True, divisor_override=None)
                                            layer2=m
                                elif ch3=="Maxpool2d":
                                            kernel_size = st.slider("Select kernel size", min_value=1,
                                                   max_value=100, step=1)
                                            stride = st.slider("Select stride", min_value=1,
                                                   max_value=100, step=1)
                                            padding = st.slider("Select padding", min_value=0, max_value=100)
                                            m=nn.MaxPool2d(kernel_size, stride, padding, dilation=1)
                                            layer2=m
                                elif ch3=="ReplicationPad":
                                    padding=st.slider("Select padding",min_value=1, max_value=1000)
                                    m=nn.ReplicationPad2d(padding)
                                    layer2=m
                                elif ch3=="ReflectionPad":
                                    padding=st.slider("Select padding",min_value=1, max_value=1000)
                                    m=nn.ReflectionPad2d(padding)
                                    layer2=m
                                elif ch3=="LLPool2d":
                                            norm_type=st.number_input("Enter norm",0,10)
                                            kernel_size = st.slider("Select kernel size", min_value=1,
                                                   max_value=100, step=1)
                                            stride = st.slider("Select stride", min_value=1,
                                                   max_value=100, step=1)
                                            padding = st.slider("Select padding", min_value=0, max_value=100)
                                            m=nn.LPPool2d(norm_type,kernel_size, stride, ceil_mode=False)
                                            layer2=m
                                else:
                                        kernel_size = st.slider("Select kernel size", min_value=1,
                                                   max_value=100, step=1)
                                        stride = st.slider("Select stride", min_value=1,
                                                   max_value=100, step=1)
                                        padding = st.slider("Select padding", min_value=0, max_value=100)
                                        m=nn.LazyConv2d(3,kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
                                        layer2=m
                                t=nn.Sequential(layer1,layer2)
                                k=(t(Mk).detach().numpy())[0]
                                st.image(work.image(k),clamp=True,channels="BGR")
                            elif ch2=="ReflectionPad":
                                padding=st.slider("Select padding",min_value=1, max_value=1000)
                                m=nn.ReflectionPad2d(padding)
                                layer1=m
                                ch3=st.selectbox("Choose 2nd layer",options=("Conv2d","LazyConv2D","Maxpool2d","Avgpool2d","LLPool2d","ReplicationPad","ReflectionPad"))
                                if ch3=="Conv2d":
                                    kernel_size = st.slider("Select kernel size", min_value=1,
                                           max_value=100, step=1)
                                    stride = st.slider("Select stride", min_value=1,
                                           max_value=100, step=1)
                                    padding = st.slider("Select padding", min_value=0, max_value=100)
                                    dilation=st.slider("Select Dilation",min_value=1, max_value=100)
                                    groups=st.slider("Select Groups(not useful in pooling layers)",min_value=1, max_value=999)
                                    m=nn.Conv2d(3, 3, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
                                    layer2=m
                                elif ch3=="Avgpool2d":
                                            kernel_size = st.slider("Select kernel size", min_value=1,
                                                   max_value=100, step=1)
                                            stride = st.slider("Select stride", min_value=1,
                                                   max_value=100, step=1)
                                            padding = st.slider("Select padding", min_value=0, max_value=100)
                                            m=nn.AvgPool2d(kernel_size,stride, padding, ceil_mode=False, count_include_pad=True, divisor_override=None)
                                            layer2=m
                                elif ch3=="Maxpool2d":
                                            kernel_size = st.slider("Select kernel size", min_value=1,
                                                   max_value=100, step=1)
                                            stride = st.slider("Select stride", min_value=1,
                                                   max_value=100, step=1)
                                            padding = st.slider("Select padding", min_value=0, max_value=100)
                                            m=nn.MaxPool2d(kernel_size, stride, padding, dilation=1)
                                            layer2=m
                                elif ch3=="ReplicationPad":
                                    padding=st.slider("Select padding",min_value=1, max_value=1000)
                                    m=nn.ReplicationPad2d(padding)
                                    layer2=m
                                elif ch3=="ReflectionPad":
                                    padding=st.slider("Select padding",min_value=1, max_value=1000)
                                    m=nn.ReflectionPad2d(padding)
                                    layer2=m
                                elif ch3=="LLPool2d":
                                            norm_type=st.number_input("Enter norm",0,10)
                                            kernel_size = st.slider("Select kernel size", min_value=1,
                                                   max_value=100, step=1)
                                            stride = st.slider("Select stride", min_value=1,
                                                   max_value=100, step=1)
                                            padding = st.slider("Select padding", min_value=0, max_value=100)
                                            m=nn.LPPool2d(norm_type,kernel_size, stride, ceil_mode=False)
                                            layer2=m
                                else:
                                        kernel_size = st.slider("Select kernel size", min_value=1,
                                                   max_value=100, step=1)
                                        stride = st.slider("Select stride", min_value=1,
                                                   max_value=100, step=1)
                                        padding = st.slider("Select padding", min_value=0, max_value=100)
                                        m=nn.LazyConv2d(3,kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
                                        layer2=m
                                t=nn.Sequential(layer1,layer2)
                                k=(t(Mk).detach().numpy())[0]
                                st.image(work.image(k),clamp=True,channels="BGR")
                            elif ch2=="LLPool2d":
                                        norm_type=st.number_input("Enter norm",0,10)
                                        kernel_size = st.slider("Select kernel size", min_value=1,
                                               max_value=100, step=1)
                                        stride = st.slider("Select stride", min_value=1,
                                               max_value=100, step=1)
                                        padding = st.slider("Select padding", min_value=0, max_value=100)
                                        m=nn.LPPool2d(norm_type,kernel_size, stride, ceil_mode=False)
                                        layer1=m
                                        ch3=st.selectbox("Choose 2nd layer",options=("Conv2d","LazyConv2D","Maxpool2d","Avgpool2d","LLPool2d","ReplicationPad","ReflectionPad"))
                                        if ch3=="Conv2d":
                                            kernel_size = st.slider("Select kernel size", min_value=1,
                                                   max_value=100, step=1)
                                            stride = st.slider("Select stride", min_value=1,
                                                   max_value=100, step=1)
                                            padding = st.slider("Select padding", min_value=0, max_value=100)
                                            dilation=st.slider("Select Dilation",min_value=1, max_value=100)
                                            groups=st.slider("Select Groups(not useful in pooling layers)",min_value=1, max_value=999)
                                            m=nn.Conv2d(3, 3, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
                                            layer2=m
                                        elif ch3=="Avgpool2d":
                                                    kernel_size = st.slider("Select kernel size", min_value=1,
                                                           max_value=100, step=1)
                                                    stride = st.slider("Select stride", min_value=1,
                                                           max_value=100, step=1)
                                                    padding = st.slider("Select padding", min_value=0, max_value=100)
                                                    m=nn.AvgPool2d(kernel_size,stride, padding, ceil_mode=False, count_include_pad=True, divisor_override=None)
                                                    layer2=m
                                        elif ch3=="Maxpool2d":
                                                    kernel_size = st.slider("Select kernel size", min_value=1,
                                                           max_value=100, step=1)
                                                    stride = st.slider("Select stride", min_value=1,
                                                           max_value=100, step=1)
                                                    padding = st.slider("Select padding", min_value=0, max_value=100)
                                                    m=nn.MaxPool2d(kernel_size, stride, padding, dilation=1)
                                                    layer2=m
                                        elif ch3=="ReplicationPad":
                                            padding=st.slider("Select padding",min_value=1, max_value=1000)
                                            m=nn.ReplicationPad2d(padding)
                                            layer2=m
                                        elif ch3=="ReflectionPad":
                                            padding=st.slider("Select padding",min_value=1, max_value=1000)
                                            m=nn.ReflectionPad2d(padding)
                                            layer2=m
                                        elif ch3=="LLPool2d":
                                                    norm_type=st.number_input("Enter norm",0,10)
                                                    kernel_size = st.slider("Select kernel size", min_value=1,
                                                           max_value=100, step=1)
                                                    stride = st.slider("Select stride", min_value=1,
                                                           max_value=100, step=1)
                                                    padding = st.slider("Select padding", min_value=0, max_value=100)
                                                    m=nn.LPPool2d(norm_type,kernel_size, stride, ceil_mode=False)
                                                    layer2=m
                                        else:
                                                kernel_size = st.slider("Select kernel size", min_value=1,
                                                           max_value=100, step=1)
                                                stride = st.slider("Select stride", min_value=1,
                                                           max_value=100, step=1)
                                                padding = st.slider("Select padding", min_value=0, max_value=100)
                                                m=nn.LazyConv2d(3,kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
                                                layer2=m
                                        t=nn.Sequential(layer1,layer2)
                                        k=(t(Mk).detach().numpy())[0]
                                        st.image(work.image(k),clamp=True,channels="BGR")
                            else:
                                    kernel_size = st.slider("Select kernel size", min_value=1,
                                               max_value=100, step=1)
                                    stride = st.slider("Select stride", min_value=1,
                                               max_value=100, step=1)
                                    padding = st.slider("Select padding", min_value=0, max_value=100)
                                    m=nn.LazyConv2d(3,kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
                                    layer1=m
                                    ch3=st.selectbox("Choose 2nd layer",options=("Conv2d","LazyConv2D","Maxpool2d","Avgpool2d","LLPool2d","ReplicationPad","ReflectionPad"))
                                    if ch3=="Conv2d":
                                        kernel_size = st.slider("Select kernel size", min_value=1,
                                               max_value=100, step=1)
                                        stride = st.slider("Select stride", min_value=1,
                                               max_value=100, step=1)
                                        padding = st.slider("Select padding", min_value=0, max_value=100)
                                        dilation=st.slider("Select Dilation",min_value=1, max_value=100)
                                        groups=st.slider("Select Groups(not useful in pooling layers)",min_value=1, max_value=999)
                                        m=nn.Conv2d(3, 3, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
                                        layer2=m
                                    elif ch3=="Avgpool2d":
                                                kernel_size = st.slider("Select kernel size", min_value=1,
                                                       max_value=100, step=1)
                                                stride = st.slider("Select stride", min_value=1,
                                                       max_value=100, step=1)
                                                padding = st.slider("Select padding", min_value=0, max_value=100)
                                                m=nn.AvgPool2d(kernel_size,stride, padding, ceil_mode=False, count_include_pad=True, divisor_override=None)
                                                layer2=m
                                    elif ch3=="Maxpool2d":
                                                kernel_size = st.slider("Select kernel size", min_value=1,
                                                       max_value=100, step=1)
                                                stride = st.slider("Select stride", min_value=1,
                                                       max_value=100, step=1)
                                                padding = st.slider("Select padding", min_value=0, max_value=100)
                                                m=nn.MaxPool2d(kernel_size, stride, padding, dilation=1)
                                                layer2=m
                                    elif ch3=="ReplicationPad":
                                        padding=st.slider("Select padding",min_value=1, max_value=1000)
                                        m=nn.ReplicationPad2d(padding)
                                        layer2=m
                                    elif ch3=="ReflectionPad":
                                        padding=st.slider("Select padding",min_value=1, max_value=1000)
                                        m=nn.ReflectionPad2d(padding)
                                        layer2=m
                                    elif ch3=="LLPool2d":
                                                norm_type=st.number_input("Enter norm",0,10)
                                                kernel_size = st.slider("Select kernel size", min_value=1,
                                                       max_value=100, step=1)
                                                stride = st.slider("Select stride", min_value=1,
                                                       max_value=100, step=1)
                                                padding = st.slider("Select padding", min_value=0, max_value=100)
                                                m=nn.LPPool2d(norm_type,kernel_size, stride, ceil_mode=False)
                                                layer2=m
                                    else:
                                            kernel_size = st.slider("Select kernel size", min_value=1,
                                                       max_value=100, step=1)
                                            stride = st.slider("Select stride", min_value=1,
                                                       max_value=100, step=1)
                                            padding = st.slider("Select padding", min_value=0, max_value=100)
                                            m=nn.LazyConv2d(3,kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
                                            layer2=m
                                    t=nn.Sequential(layer1,layer2)
                                    k=(t(Mk).detach().numpy())[0]
                                    st.image(work.image(k),clamp=True,channels="BGR")
                else:
                        kernel_size = st.slider("Select kernel size", min_value=1,
                                   max_value=100, step=1)
                        stride = st.slider("Select stride", min_value=1,
                                   max_value=100, step=1)
                        padding = st.slider("Select padding", min_value=0, max_value=100)
                        m=nn.LazyConv2d(3,kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
                        k=(m(Mk).detach().numpy())[0]
                        st.image(work.image(k),clamp=True,channels="BGR")
        #st.image(k.permute(1,2,0).numpy())
            
    except:
        print("Invalid")
           
    
    #x,y=symbols('x y')
    #lk = st.text_area("Write", value="Ok")
    #lk1 = st.write(latex(lk))
    # m=diff(expr,x,e,y,f)
    # st.latex(latex(expr))
    # st.latex(latex(m))
        
elif (option == "Image processing"):
    opt23 = st.radio("Do you wish to upload from file or url", ("File", "Url"))
    f1 = []
    if (opt23 == "File"):
        f1 = st.file_uploader("Select your files", type=["jpg", 'png', "jpeg"])
        if f1 is not None:
            file_bytes = np.asarray(bytearray(f1.read()), dtype=np.uint8)
            opencv_image = cv.imdecode(file_bytes, 1)
            opt = st.selectbox("Select the process option", ("Thresholding",
                               "Gradient", "Morphological transform", "Blur", "Edges"))
            if (opt == "Edges"):
                j1 = st.slider("Select dim1", min_value=0, max_value=1000)
                j2 = st.slider("Select dim2", min_value=0, max_value=1000)
                m = funct.edges(opencv_image, j1, j2)
                st.image(m)
            elif (opt == "Thresholding"):
                k2 = st.selectbox("Select threshold object",
                                  ("Global", "Adaptive"))
                k3 = st.slider("Select Ad1", min_value=1,
                               max_value=127, step=2)
                k4 = st.slider("Select Ad2", min_value=1,
                               max_value=255, step=2)
                k5 = st.slider("Select Ad3", min_value=1, max_value=10)
                if (k2 == "Global"):
                    m, m1, m2, m3, m4 = funct.threshold(
                        opencv_image, option=k2, adv1=k3, adv2=(k4), adv3=(k5))
                    st.image(m, caption="Binary thresh")
                    st.image(m1, caption="Inverse binary")
                    st.image(m2, caption="Truncated")
                    st.image(m3, caption="To zero")
                    st.image(m4, caption="Inverse zero")
                else:
                    m, m1 = funct.threshold(
                        opencv_image, adv1=k3, adv2=(k4), adv3=(k5))
                    st.image(m, caption="Mean Adaptive thresh")
                    st.image(m1, caption="Gaussian Adaptive thresh")
            elif (opt == "Morphological transform"):
                j1 = st.slider("Select dimension of kernel",
                               min_value=1, max_value=500)
                j2 = st.slider("Select denominator",
                               min_value=1, max_value=256)
                j3 = st.slider("Select value", min_value=0, max_value=80)
                j4 = st.selectbox("Select the option", ("2D FIlt", "erosion", "dilation",
                                  "open morph", "close morph", "morph grad", "topht", "Black"))
                j8 = funct.morph(opencv_image, j1, j2, j3, j4)
                st.image(j8, caption=j4, channels="BGR")
            elif (opt == "Gradient"):
                opt89 = st.selectbox(
                    "Select the option", ("Laplacian", "Sobelx", "sobely", "combo"))
                c = st.slider("Select ksize", min_value=1,
                              max_value=31, step=2)
                if (opt89 == "combo"):
                    s1, s2, s3 = funct.gradient(opencv_image, c=c)
                    st.image(s1, clamp=True)
                    st.image(s2, clamp=True)
                    st.image(s3, clamp=True)
                    m = st.selectbox(
                        "Select Color", ("HSV", "RGB CIE", "XYZ", "YUV", "YIQ", "YPbPr", "YCbCr", "YDbDr"))
                    l = convert_colorspace(s1, "RGB", m)
                    st.image(l, clamp=True)
                    l1 = convert_colorspace(s1, "RGB", m)
                    st.image(l1, clamp=True)
                    l2 = convert_colorspace(s3, "RGB", m)
                    st.image(l2, clamp=True)
                else:
                    s12 = funct.gradient(opencv_image, option=opt89, c=c)
                    st.image(s12, caption=opt89, clamp=True)
                    m = st.selectbox(
                        "Select Color", ("HSV", "RGB CIE", "XYZ", "YUV", "YIQ", "YPbPr", "YCbCr", "YDbDr"))

                    l = convert_colorspace(s12, "RGB", m)
                    st.image(l, clamp=True)
            else:
                s1 = st.selectbox(
                    'Option choose', ("Median Blur", "Gaussian blur", "Bilateral", 'blur'))
                s2 = st.slider("Select A", min_value=1, max_value=301, step=2)
                s3 = st.slider("select B", min_value=1, max_value=301, step=2)
                s4 = st.slider("select C", min_value=1, max_value=301, step=2)
                yt = funct.blur(opencv_image, s1, s2, s3, s4)
                st.image(yt, caption=s1, channels="BGR")
                m = st.selectbox("Select Color", ("HSV", "RGB CIE",
                                 "XYZ", "YUV", "YIQ", "YPbPr", "YCbCr", "YDbDr"))
                l = convert_colorspace(yt, "RGB", m)
                st.image(l, clamp=True)
    else:
        wew = st.text_input("Write the url")
        if wew != "":
            req = Request(wew, headers={'User-Agent': 'Mozilla/5.0'})
            f1 = urlopen(req)
            if f1 is not None:
                file_bytes = np.asarray(bytearray(f1.read()), dtype=np.uint8)
                opencv_image = cv.imdecode(file_bytes, 1)
                opt = st.selectbox("Select the process option", ("Thresholding",
                                   "Gradient", "Morphological transform", "Blur", "Edges"))
                if (opt == "Edges"):
                    j1 = st.slider("Select dim1", min_value=0, max_value=1000)
                    j2 = st.slider("Select dim2", min_value=0, max_value=1000)
                    m = funct.edges(opencv_image, j1, j2)
                    st.image(m)
                elif (opt == "Thresholding"):
                    k2 = st.selectbox("Select threshold object",
                                      ("Global", "Adaptive"))
                    k3 = st.slider("Select Ad1", min_value=1,
                                   max_value=127, step=2)
                    k4 = st.slider("Select Ad2", min_value=1,
                                   max_value=255, step=2)
                    k5 = st.slider("Select Ad3", min_value=1, max_value=10)
                    if (k2 == "Global"):
                        m, m1, m2, m3, m4 = funct.threshold(
                            opencv_image, option=k2, adv1=k3, adv2=(k4), adv3=(k5))
                        st.image(m1, caption="Inverse binary")
                        st.image(m2, caption="Truncated")
                        st.image(m3, caption="To zero")
                        st.image(m4, caption="Inverse zero")
                        t = np.concatenate((m1, m2, m3, m4), axis=0)
                        st.image(t)
                    else:
                        m, m1 = funct.threshold(
                            opencv_image, adv1=k3, adv2=(k4), adv3=(k5))
                        st.image(m, caption="Mean Adaptive thresh")
                        st.image(m1, caption="Gaussian Adaptive thresh")
                        t = np.concatenate((m1, m), axis=1)
                        st.image(t)
                elif (opt == "Morphological transform"):
                    j1 = st.slider("Select dimension of kernel",
                                   min_value=1, max_value=50)
                    j2 = st.slider("Select denominator",
                                   min_value=1, max_value=256)
                    j3 = st.slider("Select value", min_value=-1, max_value=80)
                    j4 = st.selectbox("Select the option", ("2D FIlt", "erosion", "dilation",
                                      "open morph", "close morph", "morph grad", "topht", "Black"))
                    j8 = funct.morph(opencv_image, j1, j2, j3, j4)
                    st.image(j8, caption=j4, channels="BGR")
                    m = st.selectbox(
                        "Select Color", ("HSV", "RGB CIE", "XYZ", "YUV", "YIQ", "YPbPr", "YCbCr", "YDbDr"))
                    l = convert_colorspace(j8, "RGB", m)
                    st.image(l, clamp=True)
                elif (opt == "Gradient"):
                    opt89 = st.selectbox(
                        "Select the option", ("Laplacian", "Sobelx", "sobely", "combo"))
                    c = st.slider("Select ksize", min_value=1,
                                  max_value=31, step=2)
                    if (opt89 == "combo"):
                        s1, s2, s3 = funct.gradient(opencv_image, c=c)
                        st.image(s1, clamp=True)
                        st.image(s2, clamp=True)
                        st.image(s3, clamp=True)
                        m = st.selectbox(
                            "Select Color", ("HSV", "RGB CIE", "XYZ", "YUV", "YIQ", "YPbPr", "YCbCr", "YDbDr"))
                        l = convert_colorspace(s1, "RGB", m)
                        st.image(l, clamp=True)
                        l1 = convert_colorspace(s1, "RGB", m)
                        st.image(l1, clamp=True)
                        l2 = convert_colorspace(s3, "RGB", m)
                        st.image(l2, clamp=True)
                    else:
                        s12 = funct.gradient(opencv_image, option=opt89, c=c)
                        st.image(s12, caption=opt89, clamp=True)
                        m = st.selectbox(
                            "Select Color", ("HSV", "RGB CIE", "XYZ", "YUV", "YIQ", "YPbPr", "YCbCr", "YDbDr"))
                        l = convert_colorspace(s12, "RGB", m)
                        st.image(l, clamp=True)
                else:
                    s1 = st.selectbox(
                        'Option choose', ("Median Blur", "Gaussian blur", "Bilateral", 'blur'))
                    s2 = st.slider("Select A", min_value=1,
                                   max_value=301, step=2)
                    s3 = st.slider("select B", min_value=1,
                                   max_value=301, step=2)
                    s4 = st.slider("select C", min_value=1,
                                   max_value=301, step=2)
                    yt = funct.blur(opencv_image, s1, s2, s3, s4)
                    st.image(yt, caption=s1, channels="BGR")
                    m = st.selectbox(
                        "Select Color", ("HSV", "RGB CIE", "XYZ", "YUV", "YIQ", "YPbPr", "YCbCr", "YDbDr"))
                    l = convert_colorspace(yt, "RGB", m)
                    st.image(l, clamp=True)
else:
    st.write("For Continuum assumption to hold true,Knudson number(Kn)<0.01")
    st.latex(r'''Kn=\frac{\lambda}{L}''')
    st.markdown(
        'Where λ is Mean Free Path and L is Characteristic Length(Diameter of pipe in pipe flow)')
    st.write(r'''Dynamic viscosity μ is defined as''')
    st.latex(r'''\tau=\mu\frac{du}{dy}''')
    st.write("And kinematic viscosity v is defined as")
    st.latex(r'''v=\frac{\mu}{\rho}''')
    st.write(
        r'''Where τ is shear stress and du/dy is shear strain rate(or velocity gradient)''')
    st.write("This formula is only valid for Newtonian fluids for non-Newtonian fluids following the general formula")
    st.latex(r'''\tau=m|\frac{du}{dy}|^{n-1}\frac{du}{dy}''')
    st.write("Following plot shows variation where m is flow consistency index and n is flow behaviour index")
    st.image("https://www.researchgate.net/profile/Issam-Ashqer/publication/275526328/figure/fig2/AS:614162166202397@1523439079676/Fig21-These-are-a-Bingham-plastics-b-pseudoplastic-fluids-and-c-dilatant.png", width=400)
    st.write("Bulk Modulus has the following formula")
    st.latex(r'''K=\frac{dp}{-dV/V}''')
    st.write("For isothermal process: K=p")
    st.write("For adiabatic process:K=γP")
    st.write(
        "Hydrostatic law and pascal's law determine how pressure is measured in fluids")
    st.latex(
        r'''P=\rho g h \implies h=\frac{P}{\rho g} \text {(Defined  as  pressure  head)}''')
    st.write("Pressure is measured in terms of gauge, absolute and vaccum pressure")
    st.write("Devices used for pressure measurement are Piezometer,Manometer(All types) and pressure gauges")
    st.write(
        "Hydrostatic forces on inclined surfaces is given by the following formula")
    st.latex(
        r'''F=\rho gA \bar{h} \text{ with center of pressure being given by } h*=\bar{h}+\frac{Isin^2\theta}{A\bar{h}}''')
    st.write(
        "Here I is the area moment of inertia of the body about the considered axis")
    st.write("For forces on curved surfaces, we use")
    st.latex(r'''F(x)=\rho gA \bar{h} \text{ and } F(y)=\rho V_{(in)} g''')
    st.write("**Fluid Kinematics**")
    st.write("Steady flow:Fluid characteristics do not change with time")
    st.write("Uniform flow:Fluid characteristics do not change with space")
    st.latex(
        r'''\vec{V}=u(x,y,z)\hat{i}+v(x,y,z)\hat{j}+w(x,y,z)\hat{k} \text{ represents general fluid velocity field or flow}''')
    st.latex(r'''\text{ General expression of acceleraration in x is given by } a_x=\underbrace{u\frac{\partial u}{\partial x}+v\frac{\partial u}{\partial y}+w\frac{\partial u}{\partial z}}_\text{Convective acceleration}+\underbrace{\frac{\partial u}{\partial t}}_\text{ Local acceleration}''')
    st.latex(
        r'''\text{For finding streamline equation use } \vec{V}\times\vec{dS}=0''')
    st.write("OR")
    st.latex(r'''\frac{dx}{u}=\frac{dy}{v}=\frac{dz}{w}''')
    st.latex(
        r'''\text{ If } \vec{\nabla}\times\vec{V}=0 \text{,flow is called irrotational}''')
    st.write(
        "The curl quantity is called vortitcity of the flow and rotation in flow is defined as")
    st.latex(r'''\omega=\frac{\vec{\nabla}\times\vec{V}}{2}''')
    st.write("Continuity equation,velocity potential and stream functions")
    st.latex(
        r'''\text{Continuity equation:} \frac{\partial \rho}{\partial t}+\vec{\nabla}\cdot\vec{\rho V}=0''')
    st.latex(
        r'''\vec{\nabla}\cdot\vec{u}=\frac{\partial u}{\partial x}+\frac{\partial u}{\partial y}+\frac{\partial u}{\partial z}''')
    st.latex(r'''\text{Velocity potential is denoted by} \phi''')
    st.latex(
        r'''\frac{\partial \phi}{\partial x}=-u, \frac{\partial \phi}{\partial y}=-v , \frac{\partial \phi}{\partial z}=-w''')
    st.latex(
        r'''\text{Stream function is denoted by } \psi \text{ and is valid only for 2D flow}''')
    st.latex(
        r'''\frac{\partial \psi}{\partial x}=-v, \frac{\partial \psi}{\partial y}=+u''')
    st.write("A point worth noting is that stream function won't change physically if you revrese signs on both differential equations")
    st.write("**Fluid dynamics**")
    st.write(
        "Euler's equation and Bernoulli's equation drive fluid dynamics at elementary level")
    st.latex(r'''\frac{P}{\rho g}+\frac{v^2}{2g}+z= \text{ constant } \text{ is the formal definition of Bernoulli's equation }''')
    
