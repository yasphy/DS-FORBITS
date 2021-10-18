# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 17:11:14 2021

@author: admin
"""
import cv2 as cv
import numpy as np
import streamlit as st
import funct
from urllib.request import urlopen,Request
from skimage.color import convert_colorspace
st.title("WaterMark Maker and Equation solver")
option=st.radio("Select what you want",("WaterMark","Equation solver","Fluid Mechanics","Image processing"))
if (option=="WaterMark"):
    file1=st.file_uploader("Select the files in  such a way that second file is watermark file",type=["jpg",'png'],accept_multiple_files=True)
    k=[]
    for uploaded_file in file1:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv.imdecode(file_bytes, 1)
        # Now do something with the image! For example, let's display it:
        k.append(opencv_image)
    if len(k)>1:
        for i in range(2):
            st.image(k[i], channels="BGR")
    if (file1 is not None) and (len(k)>=2):
        k1=k[0].shape
        k12=k[1].shape
        st.write("shapes are ",str(k1),"and",str(k12))
        if (k1!=k12):
            h,w,c=k[0].shape
            k[1]=cv.resize(k[1],(w,h))
            k13=k[0].shape
            k14=k[1].shape
            st.write("shapes are ",str(k13),"and",str(k14))
            o=st.number_input("Choose a number between 1 and 0")
            dst=cv.addWeighted(k[1],o,k[0],1-o,0)
            st.image(dst,channels='BGR')
elif (option=="Equation solver"):
    st.write("For exact differential equation of the given form")
    st.latex(r'''\frac{dy}{dx}=Mdx+Ndy''')
    st.write("Necessary condition is")
    st.latex(r'''\frac{\partial M}{\partial y}=\frac{\partial N}{\partial x}''')
    st.markdown("In case of nonexactness, Intergrating factors are used which is F(assuming it to be a function of x only)")
    st.latex(r'''\frac{dF}{Fdx}=\frac{\frac{\partial M}{\partial y}-\frac{\partial N}{\partial x}}{N}''')
    st.write("And for I.F. to be a function of y only")
    st.latex(r'''\frac{dF}{Fdx}=\frac{\frac{\partial M}{\partial y}-\frac{\partial N}{\partial x}}{-M}''')
    st.write("Some important shortcut formulae to remember")
    st.latex(r'''1.)d(xy)=xdy+ydx''')
    st.latex(r'''2.)d(\frac{x}{y})=\frac{ydx-xdy}{y^2}''')
    st.latex(r'''3.)d(x^2+y^2)=2ydy+2xdx''')
    st.latex(r'''4.)d(\log(\frac{x}{y}))=\frac{ydx-xdy}{xy}''')
    st.latex(r'''5.)d(\arctan(\frac{x}{y}))=\frac{ydx-xdy}{x^2+y^2}''')
    st.write("Linear Differential equations are of following form")
    st.latex(r'''\frac{dy}{dx}+yP(x)=Q(x)''')
    st.write("And their solution is given by following formula")
    st.latex(r'''y\int e^{\int P(x)dx}dx=\int Q(x) e^{\int P(x)dx}dx''')
    st.latex(r'''\text{Here } e^{\int P(x)dx} \text{ is called integrating factor}''')
    st.write("Bernoulli Equation is special kind of LDE and is of following form")
    st.latex(r'''\frac{dy}{dx}+yP(x)=Q(x)y^{n}''')
    st.write("For solving just divide the equation by coeffiecient of Q(x) and make an appropriate substitution")
    st.write("Cases are possible where reduction of orrder for a 2nd order ODE is possile if one of the independent or dependent variable is missing")
    st.latex(r'''\text{a.) If x is missing } \frac{dy}{dx}=P \text{ and } \frac{d^{2}y}{dx^{2}}=P\frac{dP}{dy}''')
    st.latex(r'''\text{b.) If y is missing } \frac{dy}{dx}=P \text{ and } \frac{d^{2}y}{dx^{2}}=\frac{dP}{dx}''')
    st.write("2nd Order Differential equations are of following form")
    st.latex(r''' y''+P(x)y'+Q(x)y=R(x) ''')   
    st.write("When R(x)=0, its called homogeneous equation")
    st.latex(r'''\text{Theorem 1:If } y_1 \text{ is a particular solution of non homomgeneous equation and } y_2 \text{ is the general solution of homogeneous solution then } y=y_1+y_2 \text{ is the general solution of the differential equation}''')
    st.latex(r'''\text{Theorem 2:If }y_1,y_2 \text{ are 2 particular linearly independent solutions of 2nd order Linear Homogeneous ODE then }cy_1+dy_2 \text{ is its general solution where c and d are arbitrary constants.}''')
    st.latex(r'''\text{Wronskian is defined as } W(y_1,y_2)=\begin{vmatrix}
   y_1 & y_1' \\
   y_2 & y_2'
\end{vmatrix}''')
    st.latex(r'''\text{Wronskian Theorem:}W(y_1,y_2) \text{ is either identically 0(Linearly dependent) or never 0(Linearly independent) for a pair} y_1,y_2 \text{ which are solutions(Non-trivial) of 2nd order Linear homogeneous ODE provided P(x) and Q(x) are continous in the interval where the solutions are given.}''')
    st.write("For linear independent solution of 2nd order Linear Homogeneous ODE , we can multiply solution y with v(x) such that")
    st.latex(r'''v=\int \frac{1}{y^2} e^{\int -Pdx} dx''')
    st.write("**Method of undetermined coefficients**")
    st.latex(r'''\text{For solving equation of type } y''+P(x)y'+Q(x)y=R(x) \text{ where R(x) is of form } e^{ax},Asin(x)+Bcos(x) \text{ and } a_0+a_1x+a_{2}x^2+....''')
    st.latex(r'''\text{The solution is of form:} xR(x) \text{ or } x^2R(x) \text{ with undetermined coefficients } A_n''')
    st.write("**Method of variation of parameters**")
    st.write("For finding particular solutions of 2nd order ODE,we use this method")
    st.latex(r'''\text{If } y_1,y_2 \text{ are general solution of the homogeneous form of our ODE then we multiply them by } v_1=-\int \frac{y_1R(x)}{W(y_1,y_2)} \text{ and } v_2=\int \frac{y_2R(x)}{W(y_1,y_2)} \Rightarrow y_p=v_1y_1+v_2y_2''')
    st.latex(r'''\text{If R(x) happens to be a sum or difference of either of the three forms then we can separately find solutions for each case and add or subtract according to the question=>Superposition principle is applied)}''')
    st.write("**Operator method**")
    st.latex(r'''\text{Any differential equation can be written in operator form where D=} \frac{d}{dx} \Rightarrow \frac{dy}{dx}+r(x)y=l(x)<=>(D-r)y=l(x) \Rightarrow \frac{l(x)}{D-r}=e^{rx} \int e^{-rx}l(x)dx=y''')
    st.write("**Some special notes**")
    st.latex(r'''\text{Bessel's equation:} x^{2}y''+xy'+(x^2-p^2)y=0''')
    st.image("https://www.accessengineeringlibrary.com/binary/mheaeworks/1d6532c8b2902e63/5de03eb02a7bc0927cc4bf3ce6e28730f07ecd096d58d72d67963522a643660c/p2001b2afg4630007vpp.png")
    st.latex(r'''\text{Normal form of Bessel's equation:} u''+(1+\frac{1-4p^2}{4x^2})u=0''')
    st.write("**Sturm separation theorem**")
    st.latex(r'''\text{If } y''+u(x)y=0,z''+w(x)z=0 \text{ are two differential equations such that } u(x)>w(x) \forall x\in R \text{ then y has atleast one zero between two consectutive zeros of z}''')
    st.latex(r'''\text{Commonly comparison is made with } y''+ty=0 \text{,where t is constant, which has solutions } \sin{\sqrt{t}},\cos{\sqrt{t}}''')
    st.write("**Power series solutions of Differential equations**")
    st.write("Works on assumption that the solution of the differential equation is a power series which has a real radius of convergence")
    st.write("Some common steps")
    st.latex(r'''y=\sum_{n=0}^{\infin} a_nx^n''')
    st.latex(r'''y'=\sum_{n=1}^{\infin} na_nx^{n-1}''')
    st.latex(r'''{y''=\sum_{n=2}^{\infin} n(n-1)a_nx^{n-2}}''')
    st.latex(r'''\Darr''')
    st.latex(r'''\text{A recursive relation in the coefficients is obtained}''')
    st.write("**Singular points**")
    st.latex(r'''\text{x=a is a regular singular point if either P(x) or Q(x) or both are non-analytic at x=a but xP(x) and } x^2Q(x) \text{ is analytic at x=a}''')
    st.write("**Frobenius solution of differential equation**")
    st.latex(r'''\text{Solution is of form } y=x^{m}\sum_{n=0}^{\infin} a_nx^n \text{ and is applicable if point of power series expansion is a regular singular point}''')
    
elif (option=="Image processing"):
    opt23=st.radio("Do you wish to upload from file or url", ("File","Url"))
    f1=[]
    if (opt23=="File"):
        f1=st.file_uploader("Select your files",type=["jpg",'png',"jpeg"])
        if f1 is not None:
            file_bytes = np.asarray(bytearray(f1.read()), dtype=np.uint8)
            opencv_image = cv.imdecode(file_bytes, 1)
            opt=st.selectbox("Select the process option", ("Thresholding","Gradient","Morphological transform","Blur","Edges"))
            if (opt=="Edges"):
                j1=st.slider("Select dim1",min_value=0,max_value=1000)
                j2=st.slider("Select dim2",min_value=0,max_value=1000)
                m=funct.edges(opencv_image,j1,j2)
                st.image(m)
            elif (opt=="Thresholding"):
                k2=st.selectbox("Select threshold object", ("Global","Adaptive"))
                k3=st.slider("Select Ad1",min_value=1,max_value=127,step=2)
                k4=st.slider("Select Ad2",min_value=1,max_value=255,step=2)
                k5=st.slider("Select Ad3",min_value=1,max_value=10)
                if (k2=="Global"):
                    m,m1,m2,m3,m4=funct.threshold(opencv_image,option=k2,adv1=k3,adv2=(k4),adv3=(k5))
                    st.image(m,caption="Binary thresh")
                    st.image(m1,caption="Inverse binary")
                    st.image(m2,caption="Truncated")
                    st.image(m3,caption="To zero")
                    st.image(m4,caption="Inverse zero")
                else:
                    m,m1=funct.threshold(opencv_image,adv1=k3,adv2=(k4),adv3=(k5))
                    st.image(m,caption="Mean Adaptive thresh")
                    st.image(m1,caption="Gaussian Adaptive thresh")
            elif (opt=="Morphological transform"):
                j1=st.slider("Select dimension of kernel",min_value=1,max_value=500)
                j2=st.slider("Select denominator",min_value=1,max_value=256)
                j3=st.slider("Select value",min_value=0,max_value=80)
                j4=st.selectbox("Select the option", ("2D FIlt","erosion","dilation","open morph","close morph","morph grad","topht","Black"))
                j8=funct.morph(opencv_image,j1, j2,j3,j4)
                st.image(j8,caption=j4,channels="BGR")
            elif (opt=="Gradient"):
                    opt89=st.selectbox("Select the option", ("Laplacian","Sobelx","sobely","combo"))
                    c=st.slider("Select ksize",min_value=1,max_value=31,step=2)
                    if (opt89=="combo"):
                        s1,s2,s3=funct.gradient(opencv_image,c=c)
                        st.image(s1,clamp=True)
                        st.image(s2,clamp=True)
                        st.image(s3,clamp=True)
                        m=st.selectbox("Select Color",("HSV", "RGB CIE", "XYZ", "YUV", "YIQ", "YPbPr", "YCbCr", "YDbDr"))
                        l=convert_colorspace(s1,"RGB",m)
                        st.image(l,clamp=True)
                        l1=convert_colorspace(s1,"RGB",m)
                        st.image(l1,clamp=True)
                        l2=convert_colorspace(s3,"RGB",m)
                        st.image(l2,clamp=True)
                    else:
                        s12=funct.gradient(opencv_image,option=opt89,c=c)
                        st.image(s12,caption=opt89,clamp=True)
                        m=st.selectbox("Select Color",("HSV", "RGB CIE", "XYZ", "YUV", "YIQ", "YPbPr", "YCbCr", "YDbDr"))

                        l=convert_colorspace(s12,"RGB",m)
                        st.image(l,clamp=True)
            else:
                    s1=st.selectbox('Option choose', ("Median Blur","Gaussian blur","Bilateral",'blur'))
                    s2=st.slider("Select A",min_value=1,max_value=301,step=2)
                    s3=st.slider("select B",min_value=1,max_value=301,step=2)
                    s4=st.slider("select C",min_value=1,max_value=301,step=2)
                    yt=funct.blur(opencv_image,s1,s2,s3,s4)
                    st.image(yt,caption=s1,channels="BGR")
                    m=st.selectbox("Select Color",("HSV", "RGB CIE", "XYZ", "YUV", "YIQ", "YPbPr", "YCbCr", "YDbDr"))
                    l=convert_colorspace(yt,"RGB",m)
                    st.image(l,clamp=True)
    else:
        wew=st.text_input("Write the url")
        if wew!="":
            req=Request(wew,headers={'User-Agent': 'Mozilla/5.0'})
            f1=urlopen(req)
            if f1 is not None:
                file_bytes = np.asarray(bytearray(f1.read()), dtype=np.uint8)
                opencv_image = cv.imdecode(file_bytes, 1)
                opt=st.selectbox("Select the process option", ("Thresholding","Gradient","Morphological transform","Blur","Edges"))
                if (opt=="Edges"):
                    j1=st.slider("Select dim1",min_value=0,max_value=1000)
                    j2=st.slider("Select dim2",min_value=0,max_value=1000)
                    m=funct.edges(opencv_image,j1,j2)
                    st.image(m)
                elif (opt=="Thresholding"):
                    k2=st.selectbox("Select threshold object", ("Global","Adaptive"))
                    k3=st.slider("Select Ad1",min_value=1,max_value=127,step=2)
                    k4=st.slider("Select Ad2",min_value=1,max_value=255,step=2)
                    k5=st.slider("Select Ad3",min_value=1,max_value=10)
                    if (k2=="Global"):
                        m,m1,m2,m3,m4=funct.threshold(opencv_image,option=k2,adv1=k3,adv2=(k4),adv3=(k5))
                        st.image(m1,caption="Inverse binary")
                        st.image(m2,caption="Truncated")
                        st.image(m3,caption="To zero")
                        st.image(m4,caption="Inverse zero")
                        t=np.concatenate((m1,m2,m3,m4), axis=0)
                        st.image(t)
                    else:
                        m,m1=funct.threshold(opencv_image,adv1=k3,adv2=(k4),adv3=(k5))
                        st.image(m,caption="Mean Adaptive thresh")
                        st.image(m1,caption="Gaussian Adaptive thresh")
                        t=np.concatenate((m1,m), axis=1)
                        st.image(t)
                elif (opt=="Morphological transform"):
                    j1=st.slider("Select dimension of kernel",min_value=1,max_value=50)
                    j2=st.slider("Select denominator",min_value=1,max_value=256)
                    j3=st.slider("Select value",min_value=-1,max_value=80)
                    j4=st.selectbox("Select the option", ("2D FIlt","erosion","dilation","open morph","close morph","morph grad","topht","Black"))
                    j8=funct.morph(opencv_image,j1, j2,j3,j4)
                    st.image(j8,caption=j4,channels="BGR")
                    m=st.selectbox("Select Color",("HSV", "RGB CIE", "XYZ", "YUV", "YIQ", "YPbPr", "YCbCr", "YDbDr"))
                    l=convert_colorspace(j8,"RGB",m)
                    st.image(l,clamp=True)
                elif (opt=="Gradient"):
                    opt89=st.selectbox("Select the option", ("Laplacian","Sobelx","sobely","combo"))
                    c=st.slider("Select ksize",min_value=1,max_value=31,step=2)
                    if (opt89=="combo"):
                        s1,s2,s3=funct.gradient(opencv_image,c=c)
                        st.image(s1,clamp=True)
                        st.image(s2,clamp=True)
                        st.image(s3,clamp=True)
                        m=st.selectbox("Select Color",("HSV", "RGB CIE", "XYZ", "YUV", "YIQ", "YPbPr", "YCbCr", "YDbDr"))
                        l=convert_colorspace(s1,"RGB",m)
                        st.image(l,clamp=True)
                        l1=convert_colorspace(s1,"RGB",m)
                        st.image(l1,clamp=True)
                        l2=convert_colorspace(s3,"RGB",m)
                        st.image(l2,clamp=True)
                    else:
                        s12=funct.gradient(opencv_image,option=opt89,c=c)
                        st.image(s12,caption=opt89,clamp=True)
                        m=st.selectbox("Select Color",("HSV", "RGB CIE", "XYZ", "YUV", "YIQ", "YPbPr", "YCbCr", "YDbDr"))
                        l=convert_colorspace(s12,"RGB",m)
                        st.image(l,clamp=True)
                else:
                    s1=st.selectbox('Option choose', ("Median Blur","Gaussian blur","Bilateral",'blur'))
                    s2=st.slider("Select A",min_value=1,max_value=301,step=2)
                    s3=st.slider("select B",min_value=1,max_value=301,step=2)
                    s4=st.slider("select C",min_value=1,max_value=301,step=2)
                    yt=funct.blur(opencv_image,s1,s2,s3,s4)
                    st.image(yt,caption=s1,channels="BGR")
                    m=st.selectbox("Select Color",("HSV", "RGB CIE", "XYZ", "YUV", "YIQ", "YPbPr", "YCbCr", "YDbDr"))
                    l=convert_colorspace(yt,"RGB",m)
                    st.image(l,clamp=True)
else:
    st.write("For Continuum assumption to hold true,Knudson number(Kn)<0.01")
    st.latex(r'''Kn=\frac{\lambda}{L}''')
    st.markdown('Where λ is Mean Free Path and L is Characteristic Length(Diameter of pipe in pipe flow)')
    st.write(r'''Dynamic viscosity μ is defined as''')
    st.latex(r'''\tau=\mu\frac{du}{dy}''')
    st.write("And kinematic viscosity v is defined as")
    st.latex(r'''v=\frac{\mu}{\rho}''')
    st.write(r'''Where τ is shear stress and du/dy is shear strain rate(or velocity gradient)''')
    st.write("This formula is only valid for Newtonian fluids for non-Newtonian fluids following the general formula")
    st.latex(r'''\tau=m|\frac{du}{dy}|^{n-1}\frac{du}{dy}''')
    st.write("Following plot shows variation where m is flow consistency index and n is flow behaviour index")
    st.image("https://www.researchgate.net/profile/Issam-Ashqer/publication/275526328/figure/fig2/AS:614162166202397@1523439079676/Fig21-These-are-a-Bingham-plastics-b-pseudoplastic-fluids-and-c-dilatant.png",width=400)
    st.write("Bulk Modulus has the following formula")
    st.latex(r'''K=\frac{dp}{-dV/V}''')
    st.write("For isothermal process: K=p")
    st.write("For adiabatic process:K=γP")
    st.write("Hydrostatic law and pascal's law determine how pressure is measured in fluids")
    st.latex(r'''P=\rho g h \implies h=\frac{P}{\rho g} \text {(Defined  as  pressure  head)}''')
    st.write("Pressure is measured in terms of gauge, absolute and vaccum pressure")
    st.write("Devices used for pressure measurement are Piezometer,Manometer(All types) and pressure gauges")
    st.write("Hydrostatic forces on inclined surfaces is given by the following formula")
    st.latex(r'''F=\rho gA \bar{h} \text{ with center of pressure being given by } h*=\bar{h}+\frac{Isin^2\theta}{A\bar{h}}''')
    st.write("Here I is the area moment of inertia of the body about the considered axis")
    st.write("For forces on curved surfaces, we use")
    st.latex(r'''F(x)=\rho gA \bar{h} \text{ and } F(y)=\rho V_{(in)} g''')
    st.write("**Fluid Kinematics**")
    st.write("Steady flow:Fluid characteristics do not change with time")
    st.write("Uniform flow:Fluid characteristics do not change with space")
    st.latex(r'''\vec{V}=u(x,y,z)\hat{i}+v(x,y,z)\hat{j}+w(x,y,z)\hat{k} \text{ represents general fluid velocity field or flow}''')
    st.latex(r'''\text{ General expression of acceleraration in x is given by } a_x=\underbrace{u\frac{\partial u}{\partial x}+v\frac{\partial u}{\partial y}+w\frac{\partial u}{\partial z}}_\text{Convective acceleration}+\underbrace{\frac{\partial u}{\partial t}}_\text{ Local acceleration}''')
    st.latex(r'''\text{For finding streamline equation use } \vec{V}\times\vec{dS}=0''')
    st.write("OR")
    st.latex(r'''\frac{dx}{u}=\frac{dy}{v}=\frac{dz}{w}''')
    st.latex(r'''\text{ If } \vec{\nabla}\times\vec{V}=0 \text{,flow is called irrotational}''')
    st.write("The curl quantity is called vortitcity of the flow and rotation in flow is defined as")
    st.latex(r'''\omega=\frac{\vec{\nabla}\times\vec{V}}{2}''')
    st.write("Continuity equation,velocity potential and stream functions")
    st.latex(r'''\text{Continuity equation:} \frac{\partial \rho}{\partial t}+\vec{\nabla}\cdot\vec{\rho V}=0''')
    st.latex(r'''\vec{\nabla}\cdot\vec{u}=\frac{\partial u}{\partial x}+\frac{\partial u}{\partial y}+\frac{\partial u}{\partial z}''')
    st.latex(r'''\text{Velocity potential is denoted by} \phi''')
    st.latex(r'''\frac{\partial \phi}{\partial x}=-u, \frac{\partial \phi}{\partial y}=-v , \frac{\partial \phi}{\partial z}=-w''')
    st.latex(r'''\text{Stream function is denoted by } \psi \text{ and is valid only for 2D flow}''')
    st.latex(r'''\frac{\partial \psi}{\partial x}=-v, \frac{\partial \psi}{\partial y}=+u''')
    st.write("A point worth noting is that stream function won't change physically if you revrese signs on both differential equations")
    st.write("**Fluid dynamics**")
    st.write("Euler's equation and Bernoulli's equation drive fluid dynamics at elementary level")
    

	 
