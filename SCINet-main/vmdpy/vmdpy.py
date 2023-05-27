# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 19:24:58 2019

@author: Vinícius Rezende Carvalho
"""
import numpy as np

def  VMD(f, alpha, tau, K, DC, init, tol):
    """
    u,u_hat,omega = VMD(f, alpha, tau, K, DC, init, tol)
    Variational mode decomposition
    Python implementation by Vinícius Rezende Carvalho - vrcarva@gmail.com
    code based on Dominique Zosso's MATLAB code, available at:
    https://www.mathworks.com/matlabcentral/fileexchange/44765-variational-mode-decomposition
    Original paper:
    Dragomiretskiy, K. and Zosso, D. (2014) ‘Variational Mode Decomposition’, 
    IEEE Transactions on Signal Processing, 62(3), pp. 531–544. doi: 10.1109/TSP.2013.2288675.
    
    
    Input and Parameters:
    ---------------------
    f       - 即将被分解的一维时域信号
    alpha   - 惩罚因子，数据保真度约束的平衡参数
    tau     - 双重上升的时间步长（为零噪声选择0）
    K       - 分解模态数
    DC      - 直流分量
    init    - 0 = all omegas start at 0
                       1 = all omegas start uniformly distributed
                      2 = all omegas initialized randomly
    tol     -  收敛准则的容忍度； 通常在1e-6左右

    Output:
    -------
    u       - 分解模态的集合
    u_hat   - 模态的频谱
    omega   - 被估计的模态中心频率
               omega 是一个矩阵，他储存了Niter-1组中心频率值，
               形状为(Niter-1, K),Niter在vmdpy.py中定义，K为分解数量
               omega矩阵储存了中心频率收敛过程中的数据，
               所以一般取最后一行来用，这是最终的中心频率
    """
    #如果f的长度是偶数，则保留原始值。否则，将f数组的最后一个元素丢弃。

    if len(f)%2:
       f = f[:-1]

    # 输入信号的周期和采样频率
    fs = 1./len(f)

    #将输入信号f复制一份，反转一半的时间序列，并将其与原始信号f拼接在一起，以得到一组对称的信号，准备进行频域处理。
    ltemp = len(f)//2
    h=np.flip(f[:ltemp],axis = 0)
    fMirr =  np.append(np.flip(f[:ltemp],axis = 0),f)
    fMirr = np.append(fMirr,np.flip(f[-ltemp:],axis = 0))

    # Time Domain 0 to T (of mirrored signal)
    T = len(fMirr)
    t = np.arange(1,T+1)/T  
    
    # Spectral Domain discretization
    freqs = t-0.5-(1/T)

    # 最大迭代次数（如果没有收敛，则不会收敛）
    Niter = 500
    # 为进一步的泛化：为每种模式设置单独的 alpha
    Alpha = alpha*np.ones(K)
    
    # 计算傅里叶变换后的频率离散化
    f_hat = np.fft.fftshift((np.fft.fft(fMirr)))
    #构造并居中f_hat。通过复制f_hat得到f_hat_plus，将其前一半设置为0。
    f_hat_plus = np.copy(f_hat) #copy f_hat
    f_hat_plus[:T//2] = 0

    # Initialization of omega_k
    omega_plus = np.zeros([Niter, K])

    #如果init值为1，则将所有omega值均匀分布。
    # 如果init值为2，则随机初始化所有omega值。否则，所有omega值均为0。
    if init == 1:
        for i in range(K):
            omega_plus[0,i] = (0.5/K)*(i)
    elif init == 2:
        omega_plus[0,:] = np.sort(np.exp(np.log(fs) + (np.log(0.5)-np.log(fs))*np.random.rand(1,K)))
    else:
        omega_plus[0,:] = 0
            
    # 如果存在直流分量，则将其对应的omega值设置为0。
    if DC:
        omega_plus[0,0] = 0
    
    #开始设置其它参数。lambda_hat初始化为空；
    lambda_hat = np.zeros([Niter, len(freqs)], dtype = complex)


    # uDiff指示函数是否已收敛；
    uDiff = tol+np.spacing(1) # update step
    n = 0 #n计算迭代次数；
    sum_uk = 0 # sum_uk是一个累加器；

    #u_hat_plus保存每一组迭代的值。
    u_hat_plus = np.zeros([Niter, len(freqs), K],dtype=complex)    

   #对于每一次迭代，如果函数未收敛且循环次数小于最大迭代次数，执行如下操作：
    while ( uDiff > tol and  n < Niter-1 ): # not converged and below iterations limit
        # 更新第1模态的累加器
        k = 0
        sum_uk = u_hat_plus[n,:,K-1] + sum_uk - u_hat_plus[n,:,0]
        
        # 通过残差的维纳滤波器更新第一种模式的频谱
        u_hat_plus[n+1,:,k] = (f_hat_plus - sum_uk - lambda_hat[n,:]/2)/(1.+Alpha[k]*(freqs - omega_plus[n,k])**2)
        
        #  更新第1种模式的中心频率，若不是0则进行更新
        if not(DC):
            omega_plus[n+1,k] = np.dot(freqs[T//2:T],(abs(u_hat_plus[n+1, T//2:T, k])**2))/np.sum(abs(u_hat_plus[n+1,T//2:T,k])**2)

        # 更新其他模式
        for k in np.arange(1,K):
            #累加器
            sum_uk = u_hat_plus[n+1,:,k-1] + sum_uk - u_hat_plus[n,:,k]
            # 模式谱
            u_hat_plus[n+1,:,k] = (f_hat_plus - sum_uk - lambda_hat[n,:]/2)/(1+Alpha[k]*(freqs - omega_plus[n,k])**2)
            # 中心频率
            omega_plus[n+1,k] = np.dot(freqs[T//2:T],(abs(u_hat_plus[n+1, T//2:T, k])**2))/np.sum(abs(u_hat_plus[n+1,T//2:T,k])**2)
            
        #  双重上升
        lambda_hat[n+1,:] = lambda_hat[n,:] + tau*(np.sum(u_hat_plus[n+1,:,:],axis = 1) - f_hat_plus)
        
        # 循环计数器
        n = n+1
        
        #是否已经达到收敛？
        uDiff = np.spacing(1)
        for i in range(K):
            uDiff = uDiff + (1/T)*np.dot((u_hat_plus[n,:,i]-u_hat_plus[n-1,:,i]),np.conj((u_hat_plus[n,:,i]-u_hat_plus[n-1,:,i])))

        uDiff = np.abs(uDiff)        
            
    # 后处理和清理
    
    #若早已收敛，则丢弃空白部分
    Niter = np.min([Niter,n])
    omega = omega_plus[:Niter,:]
    
    idxs = np.flip(np.arange(1,T//2+1),axis = 0)
    # 信号重构
    u_hat = np.zeros([T, K],dtype = complex)
    u_hat[T//2:T,:] = u_hat_plus[Niter-1,T//2:T,:]
    u_hat[idxs,:] = np.conj(u_hat_plus[Niter-1,T//2:T,:])
    u_hat[0,:] = np.conj(u_hat[-1,:])    
    
    u = np.zeros([K,len(t)])
    for k in range(K):
        u[k,:] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:,k])))
        
    # 移除镜像部分
    u = u[:,T//4:3*T//4]

    # 重新计算频谱
    u_hat = np.zeros([u.shape[1],K],dtype = complex)
    for k in range(K):
        u_hat[:,k]=np.fft.fftshift(np.fft.fft(u[k,:]))

    return u, u_hat, omega
