#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   LFR.py
@Contact :   
@License :   

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/8/9 18:16   linsher      1.0         None
'''
import os
import shutil
from tools import formater
import time
def generate(path,network_dir,comm_dir,N=1000,k=20,maxk=50,mu=0.1,minc=10,maxc=100,
             on=0,om=0,t_degree=2.0,t_comm=1.0):
    param_list = [' -N ' + str(N) , ' -k ' + str(k) ,
                  ' -maxk ' + str(maxk),' -mu ' + '{:.1f}'.format(mu),
                  ' -minc ' + str(minc), ' -maxc ' + str(maxc),
                  ' -on ' + str(on), ' -om ' + str(om),
                  ' -t1 ' + '{:.1f}'.format(t_degree), ' -t2 ' + '{:.1f}'.format(t_comm)]

    params = ''.join(param_list)

    suffix = params.strip()
    suffix = suffix.replace(' ', '_')
    suffix = suffix.replace('-', '')
    network_name = 'network_' + suffix
    comms_name = 'community_' + suffix

    cmd = '"' + path + '"' + params


    cwd = os.getcwd()

    # dir = path [:path.rfind('\\')+1]

    network_file = cwd + '/network.dat'
    comm_file =  cwd + '/community.dat'
    static_file = cwd + '/statistics.dat'
    seed_file = cwd + '/time_seed.dat'

    if os.path.exists(comm_file):
        os.remove(comm_file)
    if os.path.exists(network_file):
        os.remove(network_file)
    if os.path.exists(seed_file):
        os.remove(seed_file)
    if os.path.exists(static_file):
        os.remove(static_file)

    s = os.popen(cmd)
    # print(s.read())
    time.sleep(0.2*N/1000)
    print(network_file)
    if not os.path.exists(network_file):
        print('generate network fail!')
        return

    # os.rename(network_file,network_name)
    print(network_dir + network_name + '.txt')
    time.sleep(1)
    target_network_path = network_dir + network_name + '.txt'
    target_comms_path = comm_dir + comms_name + '.txt'

    if os.path.exists(target_network_path):
        os.remove(target_network_path)
        time.sleep(0.5)
    if os.path.exists(target_comms_path):
        os.remove(target_comms_path)
        time.sleep(0.5)
    time.sleep(1)
    os.rename(network_file,target_network_path)
    formater.interface(comm_file, target_comms_path)
    time.sleep(0.5)
    os.remove(comm_file)
    os.remove(seed_file)
    os.remove(static_file)
    time.sleep(0.5)




path = r'G:\research\CMtools\binary_networks(LFR)\Release\binary_networks.exe'

# for i in range(1,6):
#     k = i*1000
#     generate(path,network_dir = '../dataset/LFR_data/',
#              comm_dir= '../label/LFR_data/',
#              N=k,k=20,maxk=50,mu=0.7,t_degree=0.5)

for i in range(1,8):
    mu = i*0.1
    generate(path,network_dir = '../dataset/LFR_data/',
             comm_dir= '../label/LFR_data/',
             k=20,maxk=100,mu=mu,t_degree=2)

# for i in range(1,8):
#     mu = i*0.1
#     generate(path,network_dir = '../dataset/LFR_data/',
#              comm_dir= '../label/LFR_data/',
#              k=10,mu=mu,t_degree=0.2)

# for i in range(1,8):
#     mu = i*0.1
#     generate(path,network_dir = '../dataset/LFR_data/',
#              comm_dir= '../label/LFR_data/',
#              mu=mu,t_degree=0.5)
