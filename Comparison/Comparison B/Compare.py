# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:38:55 2023

@author: 17100
"""

import numpy as np 
import matplotlib.pyplot as plt

nsam=10
iteq=200
ite=4000
N=201
N1=351
N2=501
tim_stp=50
p_level=0.2

#Our
def simulate_minority(N, ite, iteq):
    s=2 # Strategy Size
    m=5 # Memory Size
    stra_ags=[[] for i in range(N)] # Strategies of agents, each with a list containing s stratwgies
    choice=[0 for i in range(N)] #Record the choice of agents each round.
    result=[] #Results of all rounds.
    count=[] #Number of buyers in all rounds.
    r_max=[] # For each round, record the wealthiest agent's real score.
    r_min=[] # For each round, record the poorest agent's real score.
    r_mean=[] # For each round, record the agents' average real score.
    memory="00000"
    state=0
    d=-4#Threshold
    
    
    for i in range(N):
        c0=0
        while c0<2: #Prepare strategies for all agents.
            stp = "".join([np.random.choice(["0", "1"]) for k in range(int(2**m))])
            if stp not in stra_ags[i]:
                stra_ags[i].append(stp)
                c0+=1
            else:
                continue
    vir_score=[[0,0] for i in range(N)]
    real_score=[0 for i in range(N)]

    t=0

    while t<=ite:
        c1 = 0
        if t==0:
            for i in range(N):
                choice[i]=int(np.random.choice(stra_ags[i])[state])
                c1+=int(np.random.choice(stra_ags[i])[state])
            ecs=abs(float((N-c1-c1))/N)
            if c1>N//2:
                result.append("0")
                for i in range(N):
                    if choice[i]==0:
                        real_score[i]+=ecs
                    else:
                        real_score[i]-=ecs
                for j in range(N):
                    if stra_ags[j][0][state]=="0":
                        vir_score[j][0]+=ecs
                    else:
                        vir_score[j][0]-=ecs
                    if stra_ags[j][1][state]=="0":
                        vir_score[j][1]+=ecs
                    else:
                        vir_score[j][1]-=ecs
            else:
                result.append("1")
                for i in range(N):
                    if choice[i]==1:
                        real_score[i]+=ecs
                    else:
                        real_score[i]-=ecs
                for j in range(N):
                    if stra_ags[j][0][state]=="1":
                        vir_score[j][0]+=ecs
                    else:
                        vir_score[j][0]-=ecs
                    if stra_ags[j][1][state]=="1":
                        vir_score[j][1]+=ecs
                    else:
                        vir_score[j][1]-=ecs
            memory=memory[1:]+result[-1] #Update memory and states
            state=int(2**4*int(memory[0])+2**3*int(memory[1])+2**2*int(memory[2])+2**1*int(memory[3])+int(memory[4]))
            t+=1   
        
        else:
            if t<iteq:
                for i in range(N):
                    if vir_score[i][0]>vir_score[i][1]:
                        choice[i]=int(stra_ags[i][0][state])
                        c1+=int(stra_ags[i][0][state])
                    elif vir_score[i][0]<vir_score[i][1]:
                        choice[i]=int(stra_ags[i][1][state])
                        c1+=int(stra_ags[i][1][state])
                    else:
                        choice[i]=int(np.random.choice(stra_ags[i])[state])
                        c1+=choice[i]
                ecs=abs(float((N-c1-c1))/N) #Linear Score
                        
                if c1>N//2:
                    result.append("0")
                    for i in range(N):
                        if choice[i]==0:
                            real_score[i]+=ecs
                        else:
                            real_score[i]-=ecs
                    for j in range(N):
                        if stra_ags[j][0][state]=="0":
                            vir_score[j][0]+=ecs
                        else:
                            vir_score[j][0]-=ecs
                        if stra_ags[j][1][state]=="0":
                            vir_score[j][1]+=ecs
                        else:
                            vir_score[j][1]-=ecs
                else:
                    result.append("1")
                    for i in range(N):
                        if choice[i]==1:
                            real_score[i]+=ecs
                        else:
                            real_score[i]-=ecs
                    for j in range(N):
                        if stra_ags[j][0][state]=="1":
                            vir_score[j][0]+=ecs
                        else:
                            vir_score[j][0]-=ecs
                        if stra_ags[j][1][state]=="1":
                            vir_score[j][1]+=ecs
                        else:
                            vir_score[j][1]-=ecs
                memory=memory[1:]+result[-1]
                state=int(2**4*int(memory[0])+2**3*int(memory[1])+2**2*int(memory[2])+2**1*int(memory[3])+int(memory[4]))
                st=stra_ags[real_score.index(max(real_score))]
                for i in range(N): #Eliminate poor players
                    if real_score[i]<d:
                        vir_score[i]=[0,0]
                        real_score[i]=0
                        stra_ags[i]=[np.random.choice(st)]
                        c0=0
                        while c0<1:
                            stp = "".join([np.random.choice(["0", "1"]) for k in range(int(2**m))])
                            if stp not in stra_ags[i]:
                                stra_ags[i].append(stp)
                                c0+=1
                            else:
                                continue
                t+=1
                        
            elif t>=iteq and t<=ite:
                for i in range(N):
                    if vir_score[i][0]>vir_score[i][1]:
                        choice[i]=int(stra_ags[i][0][state])
                        c1+=int(stra_ags[i][0][state])
                    elif vir_score[i][0]<vir_score[i][1]:
                        choice[i]=int(stra_ags[i][1][state])
                        c1+=int(stra_ags[i][1][state])
                    else:
                        choice[i]=int(np.random.choice(stra_ags[i])[state])
                        c1+=choice[i]
                count.append(N-c1)
                ecs=abs(float((N-c1-c1))/N)
                if c1>N//2:
                    result.append("0")
                    for i in range(N):
                        if choice[i]==0:
                            real_score[i]+=ecs
                        else:
                            real_score[i]-=ecs
                    for j in range(N):
                        if stra_ags[j][0][state]=="0":
                            vir_score[j][0]+=ecs
                        else:
                            vir_score[j][0]-=ecs
                        if stra_ags[j][1][state]=="0":
                            vir_score[j][1]+=ecs
                        else:
                            vir_score[j][1]-=ecs
                else:
                    result.append("1")
                    for i in range(N):
                        if choice[i]==1:
                            real_score[i]+=ecs
                        else:
                            real_score[i]-=ecs
                    for j in range(N):
                        if stra_ags[j][0][state]=="1":
                            vir_score[j][0]+=ecs
                        else:
                            vir_score[j][0]-=ecs
                        if stra_ags[j][1][state]=="1":
                            vir_score[j][1]+=ecs
                        else:
                            vir_score[j][1]-=ecs
                memory=memory[1:]+result[-1]
                state=int(2**4*int(memory[0])+2**3*int(memory[1])+2**2*int(memory[2])+2**1*int(memory[3])+int(memory[4]))
                r_max.append(max(real_score))
                r_min.append(min(real_score))
                r_mean.append(np.average(real_score))
                st=stra_ags[real_score.index(max(real_score))]
                for i in range(N):
                    if real_score[i]<d:
                        vir_score[i]=[0,0]
                        real_score[i]=0
                        stra_ags[i]=[np.random.choice(st)]
                        c0=0
                        while c0<1:
                            stp = "".join([np.random.choice(["0", "1"]) for k in range(int(2**m))])
                            if stp not in stra_ags[i]:
                                stra_ags[i].append(stp)
                                c0+=1
                            else:
                                continue
                    
            t+=1
    return count, real_score, r_max, r_mean, r_min


#Li, Savit v1
def simulate_minority2(N, ite, iteq, tim_stp, p_level):
    s=2 # Strategy Size
    m=5 # Memory Size
    stra_ags=[[] for i in range(N)] # Strategies of agents, each with a list containing s stratwgies
    choice=[0 for i in range(N)] #Record the choice of agents each round.
    result=[] #Results of all rounds.
    count=[] #Number of buyers in all rounds.
    r_max=[] # For each round, record the wealthiest agent's real score.
    r_min=[] # For each round, record the poorest agent's real score.
    r_mean=[] # For each round, record the agents' average real score.
    memory="00000"
    state=0
    d=-4#Threshold
    
    
    for i in range(N):
        c0=0
        while c0<2: #Prepare strategies for all agents.
            stp = "".join([np.random.choice(["0", "1"]) for k in range(int(2**m))])
            if stp not in stra_ags[i]:
                stra_ags[i].append(stp)
                c0+=1
            else:
                continue
    vir_score=[[0,0] for i in range(N)]
    real_score=[0 for i in range(N)]
    real_score2=[0 for i in range(N)]# Record real scores in each generation.
    t=0

    while t<=ite:
        c1 = 0
        if t==0:
            for i in range(N):
                choice[i]=int(np.random.choice(stra_ags[i])[state])
                c1+=choice[i]
            ecs=abs(float((N-c1-c1))/N)
            if c1>N//2:
                result.append("0")
                for i in range(N):
                    if choice[i]==0:
                        real_score[i]+=ecs
                        real_score2[i]+=ecs
                    else:
                        real_score[i]-=ecs
                        real_score2[i]-=ecs
                for j in range(N):
                    if stra_ags[j][0][state]=="0":
                        vir_score[j][0]+=ecs
                    else:
                        vir_score[j][0]-=ecs
                    if stra_ags[j][1][state]=="0":
                        vir_score[j][1]+=ecs
                    else:
                        vir_score[j][1]-=ecs
            else:
                result.append("1")
                for i in range(N):
                    if choice[i]==1:
                        real_score[i]+=ecs
                        real_score2[i]+=ecs
                    else:
                        real_score[i]-=ecs
                        real_score2[i]-=ecs
                for j in range(N):
                    if stra_ags[j][0][state]=="1":
                        vir_score[j][0]+=ecs
                    else:
                        vir_score[j][0]-=ecs
                    if stra_ags[j][1][state]=="1":
                        vir_score[j][1]+=ecs
                    else:
                        vir_score[j][1]-=ecs
            memory=memory[1:]+result[-1] #Update memory and states
            state=int(2**4*int(memory[0])+2**3*int(memory[1])+2**2*int(memory[2])+2**1*int(memory[3])+int(memory[4]))
            t+=1   
        
        else:
            if t<iteq:
                for i in range(N):
                    if vir_score[i][0]>vir_score[i][1]:
                        choice[i]=int(stra_ags[i][0][state])
                        c1+=int(stra_ags[i][0][state])
                    elif vir_score[i][0]<vir_score[i][1]:
                        choice[i]=int(stra_ags[i][1][state])
                        c1+=int(stra_ags[i][1][state])
                    else:
                        choice[i]=int(np.random.choice(stra_ags[i])[state])
                        c1+=choice[i]
                ecs=abs(float((N-c1-c1))/N) #Linear Score
                        
                if c1>N//2:
                    result.append("0")
                    for i in range(N):
                        if choice[i]==0:
                            real_score[i]+=ecs
                            real_score2[i]+=ecs
                        else:
                            real_score[i]-=ecs
                            real_score2[i]-=ecs
                    for j in range(N):
                        if stra_ags[j][0][state]=="0":
                            vir_score[j][0]+=ecs
                        else:
                            vir_score[j][0]-=ecs
                        if stra_ags[j][1][state]=="0":
                            vir_score[j][1]+=ecs
                        else:
                            vir_score[j][1]-=ecs
                else:
                    result.append("1")
                    for i in range(N):
                        if choice[i]==1:
                            real_score[i]+=ecs
                            real_score2[i]+=ecs
                        else:
                            real_score[i]-=ecs
                            real_score2[i]-=ecs
                    for j in range(N):
                        if stra_ags[j][0][state]=="1":
                            vir_score[j][0]+=ecs
                        else:
                            vir_score[j][0]-=ecs
                        if stra_ags[j][1][state]=="1":
                            vir_score[j][1]+=ecs
                        else:
                            vir_score[j][1]-=ecs
                memory=memory[1:]+result[-1]
                state=int(2**4*int(memory[0])+2**3*int(memory[1])+2**2*int(memory[2])+2**1*int(memory[3])+int(memory[4]))
                st=stra_ags[real_score.index(max(real_score))]
                if t%tim_stp==0:
                    rs_order=np.argsort(real_score2)[:int(N*p_level)+1]
                    rs_mutate=np.random.choice(rs_order, len(rs_order)//2)
                    for j in rs_order:
                        real_score[j]=0
                    for i in rs_mutate:
                        vir_score[i]=[0,0]
                        stra_ags[i]=[]
                        c0=0
                        while c0<2:
                            stp = "".join([np.random.choice(["0", "1"]) for k in range(int(2**m))])
                            if stp not in stra_ags[i]:
                                stra_ags[i].append(stp)
                                c0+=1
                            else:
                                continue
                    real_score2=[0 for i in range(N)]
                        
                t+=1
                        
            elif t>=iteq and t<=ite:
                for i in range(N):
                    if vir_score[i][0]>vir_score[i][1]:
                        choice[i]=int(stra_ags[i][0][state])
                        c1+=int(stra_ags[i][0][state])
                    elif vir_score[i][0]<vir_score[i][1]:
                        choice[i]=int(stra_ags[i][1][state])
                        c1+=int(stra_ags[i][1][state])
                    else:
                        choice[i]=int(np.random.choice(stra_ags[i])[state])
                        c1+=choice[i]
                count.append(N-c1)
                ecs=abs(float((N-c1-c1))/N)
                if c1>N//2:
                    result.append("0")
                    for i in range(N):
                        if choice[i]==0:
                            real_score[i]+=ecs
                        else:
                            real_score[i]-=ecs
                    for j in range(N):
                        if stra_ags[j][0][state]=="0":
                            vir_score[j][0]+=ecs
                        else:
                            vir_score[j][0]-=ecs
                        if stra_ags[j][1][state]=="0":
                            vir_score[j][1]+=ecs
                        else:
                            vir_score[j][1]-=ecs
                else:
                    result.append("1")
                    for i in range(N):
                        if choice[i]==1:
                            real_score[i]+=ecs
                        else:
                            real_score[i]-=ecs
                    for j in range(N):
                        if stra_ags[j][0][state]=="1":
                            vir_score[j][0]+=ecs
                        else:
                            vir_score[j][0]-=ecs
                        if stra_ags[j][1][state]=="1":
                            vir_score[j][1]+=ecs
                        else:
                            vir_score[j][1]-=ecs
                memory=memory[1:]+result[-1]
                state=int(2**4*int(memory[0])+2**3*int(memory[1])+2**2*int(memory[2])+2**1*int(memory[3])+int(memory[4]))
                r_max.append(max(real_score))
                r_min.append(min(real_score))
                r_mean.append(np.average(real_score))
                st=stra_ags[real_score.index(max(real_score))]
                k=min(real_score)
                if t%tim_stp==0:
                    rs_order=np.argsort(real_score2)[:int(N*p_level)+1]
                    rs_mutate=np.random.choice(rs_order, len(rs_order)//2)
                    for j in rs_order:
                        real_score[j]=0
                    for i in rs_mutate:
                        vir_score[i]=[0,0]
                        stra_ags[i]=[]
                        c0=0
                        while c0<2:
                            stp = "".join([np.random.choice(["0", "1"]) for k in range(int(2**m))])
                            if stp not in stra_ags[i]:
                                stra_ags[i].append(stp)
                                c0+=1
                            else:
                                continue
                    real_score2=[0 for i in range(N)]
                    
            t+=1
    return count, real_score, r_max, r_mean, r_min

#Lo
def simulate_minority3(N, R, iteq):
    R0=0.2 # Width of stratgy changes
    m=5 #Memory size
    c1=0#Number of 0 results.
    c2=0#Number of 1 results
    result=[]
    d=-4 #Threshold
    LP=[[[] for i in range(101)] for j in range(N)]
    tchange=[[0] for i in range(N)]
    Att=[]
    r_max=[]
    r_min=[]
    r_mean=[]

    p_players=[round(np.random.random(), 2) for i in range(N)]
    ps=[[p_players[i]] for i in range(N)]
    c3=0
    memory={}
    while c3<2**5:
        ml=np.random.choice(['0', '1'], 5)
        ml_str="".join(list(ml))
        if ml_str not in memory.keys():
            memory[ml_str]=np.random.randint(0,1)
        else:
            continue
        c3+=1
    t = 0 #Record the times of game played
    choice=[[] for i in range(N)]
    score=[0 for i in range(N)]

    while t<m:
        c=[np.random.randint(0,1) for i in range(N)]
        for i in range(N):
            choice[i].append(c)
        ecs=abs(float((N-sum(c)-sum(c)))/N)
        if sum(c)>N//2:
            c1+=1
            result.append("0")
            for i in range(N):
                if choice[i][-1]==0:
                    score[i]+=ecs
                else:
                    score[i]-=ecs
        else:
            c2+=1
            result.append("1")
            for i in range(N):
                if choice[i][-1]==1:
                    score[i]+=ecs
                else:
                    score[i]-=ecs
        t+=1
        
    while t>=m and t<iteq:
        rp = "".join(result[-5:])
        s=0
        for i in range(N):
            p=p_players[i]
            cset = [memory[rp]]*int(p*100)+[1-memory[rp]]*int((1-p)*100)
            c=np.random.choice(cset)
            choice[i].append(c)
            s+=c
        ecs=abs(float((N-s-s))/N)
        if s>N//2:
            c1+=1
            result.append("0")
            memory[rp]=0
            for i in range(N):
                if choice[i][-1]==0:
                    score[i]+=ecs
                else:
                    score[i]-=ecs
        else:
            c2+=1
            result.append("1")
            memory[rp]=1
            for i in range(N):
                if choice[i][-1]==1:
                    score[i]+=ecs
                else:
                    score[i]-=ecs
        for i in range(N):
            if score[i]<d:
                LP[i][int(100*p_players[i])].append(t-tchange[i][-1])
                p_players[i]=round(np.random.uniform(max(p_players[i]-R0/2, 0), min(p_players[i]+R0/2, 1)), 2)
                ps[i].append(p_players[i])
                tchange[i].append(t)
                score[i]=0
        t+=1  
    while t>=iteq and t<R:
        rp = "".join(result[-5:])
        s=0
        for i in range(N):
            p=p_players[i]
            cset = [memory[rp]]*int(p*100)+[1-memory[rp]]*int((1-p)*100)
            c=np.random.choice(cset)
            choice[i].append(c)
            s+=c
        Att.append(s)
        ecs=abs(float((N-s-s))/N)
        if s>N//2:
            c1+=1
            result.append("0")
            memory[rp]=0
            for i in range(N):
                if choice[i][-1]==0:
                    score[i]+=ecs
                else:
                    score[i]-=ecs
        else:
            c2+=1
            result.append("1")
            memory[rp]=1
            for i in range(N):
                if choice[i][-1]==1:
                    score[i]+=ecs
                else:
                    score[i]-=ecs
        r_max.append(max(score))
        r_min.append(min(score))
        r_mean.append(np.average(np.array(score, dtype=float)))
        for i in range(N):
            if score[i]<d:
                LP[i][int(100*p_players[i])].append(t-tchange[i][-1])
                p_players[i]=round(np.random.uniform(max(p_players[i]-R0/2, 0), min(p_players[i]+R0/2, 1)), 2)
                ps[i].append(p_players[i])
                tchange[i].append(t)
                score[i]=0
        t+=1  
    return Att, score, r_max, r_min, r_mean

#Basic
def simulate_minority4(N, ite, iteq):
    s=2 # Strategy Size
    m=5
    stra_ags=[[] for i in range(N)]
    stra_total=[]
    choice=[0 for i in range(N)]
    result=[]
    count=[]
    memory="00000"
    state=0
    r_max=[]
    r_mean=[]
    r_min=[]
    
    
    for i in range(N):
        c0=0
        while c0<2:
            stp = "".join([np.random.choice(["0", "1"]) for k in range(int(2**m))])
            if stp not in stra_ags[i]:
                stra_ags[i].append(stp)
                stra_total.append(stp)
                c0+=1
            else:
                continue
    l=len(stra_total)
    vir_score=[0 for i in range(l)]
    real_score=[0 for i in range(N)]

    t=0

    while t<ite:
        c1 = 0
        if t==0:
            for i in range(N):
                choice[i]=int(np.random.choice(stra_ags[i])[state])
                c1+=choice[i]
            ecs=abs(float((N-c1-c1))/N)
            if c1>N//2:
                result.append("0")
                for i in range(N):
                    if choice[i]==0:
                        real_score[i]+=ecs
                    else:
                        real_score[i]-=ecs
                for j in range(l):
                    if stra_total[j][state]=="0":
                        vir_score[j]+=ecs
                    else:
                        vir_score[j]-=ecs
            else:
                result.append("1")
                for i in range(N):
                    if choice[i]==1:
                        real_score[i]+=ecs
                    else:
                        real_score[i]-=ecs
                for j in range(l):
                    if stra_total[j][state]=="1":
                        vir_score[j]+=ecs
                    else:
                        vir_score[j]-=ecs
            memory=memory[1:]+result[-1]
            state=int(2**4*int(memory[0])+2**3*int(memory[1])+2**2*int(memory[2])+2**1*int(memory[3])+int(memory[4]))
            t+=1   
        
        else:
            if t<iteq:
                for i in range(N):
                    if vir_score[stra_total.index(stra_ags[i][0])]>vir_score[stra_total.index(stra_ags[i][1])]:
                        choice[i]=int(stra_ags[i][0][state])
                        c1+=int(stra_ags[i][0][state])
                    elif vir_score[stra_total.index(stra_ags[i][0])]<vir_score[stra_total.index(stra_ags[i][1])]:
                        choice[i]=int(stra_ags[i][1][state])
                        c1+=int(stra_ags[i][1][state])
                    else:
                        choice[i]=int(np.random.choice(stra_ags[i])[state])
                        c1+=choice[i]
                ecs=abs(float((N-c1-c1))/N)
                        
                if c1>N//2:
                    result.append("0")
                    for i in range(N):
                        if choice[i]==0:
                            real_score[i]+=ecs
                        else:
                            real_score[i]-=ecs
                    for j in range(l):
                        if stra_total[j][state]=="0":
                            vir_score[i]+=ecs
                        else:
                            vir_score[i]-=ecs
                else:
                    result.append("1")
                    for i in range(N):
                        if choice[i]==1:
                            real_score[i]+=ecs
                        else:
                            real_score[i]-=ecs
                    for j in range(l):
                        if stra_total[j][state]=="1":
                            vir_score[j]+=ecs
                        else:
                            vir_score[j]-=ecs
                memory=memory[1:]+result[-1]
                state=int(2**4*int(memory[0])+2**3*int(memory[1])+2**2*int(memory[2])+2**1*int(memory[3])+int(memory[4]))
            elif t>=iteq and t<ite:
                for i in range(N):
                    if vir_score[stra_total.index(stra_ags[i][0])]>vir_score[stra_total.index(stra_ags[i][1])]:
                        choice[i]=int(stra_ags[i][0][state])
                        c1+=int(stra_ags[i][0][state])
                    elif vir_score[stra_total.index(stra_ags[i][0])]<vir_score[stra_total.index(stra_ags[i][1])]:
                        choice[i]=int(stra_ags[i][1][state])
                        c1+=int(stra_ags[i][1][state])
                    else:
                        choice[i]=int(np.random.choice(stra_ags[i])[state])
                        c1+=choice[i]
                count.append(N-c1)
                ecs=abs(float((N-c1-c1))/N)
                if c1>N//2:
                    result.append("0")
                    for i in range(N):
                        if choice[i]==0:
                                real_score[i]+=ecs
                        else:
                            real_score[i]-=ecs
                    for j in range(l):
                        if stra_total[j][state]=="0":
                            vir_score[j]+=ecs
                        else:
                            vir_score[j]-=ecs
                else:
                    result.append("1")
                    for i in range(N):
                        if choice[i]==1:
                            real_score[i]+=ecs
                        else:
                            real_score[i]-=ecs
                    for j in range(l):
                        if stra_total[j][state]=="1":
                            vir_score[j]+=ecs
                        else:
                            vir_score[j]-=ecs
                r_min.append(min(real_score))
                r_max.append(max(real_score))
                r_mean.append(np.average(real_score))
                memory=memory[1:]+result[-1]
                state=int(2**4*int(memory[0])+2**3*int(memory[1])+2**2*int(memory[2])+2**1*int(memory[3])+int(memory[4]))
                    
            t+=1
    return count, r_max, r_mean, r_min

#Zhang
def simulate_minority5(N, ite, iteq, tim_stp):
    s=2 # Strategy Size
    m=5 # Memory Size
    stra_ags=[[] for i in range(N)] # Strategies of agents, each with a list containing s stratwgies
    choice=[0 for i in range(N)] #Record the choice of agents each round.
    result=[] #Results of all rounds.
    count=[] #Number of buyers in all rounds.
    r_max=[] # For each round, record the wealthiest agent's real score.
    r_min=[] # For each round, record the poorest agent's real score.
    r_mean=[] # For each round, record the agents' average real score.
    memory="00000"
    state=0
    d=-4#Threshold
    
    
    for i in range(N):
        c0=0
        while c0<2: #Prepare strategies for all agents.
            stp = "".join([np.random.choice(["0", "1"]) for k in range(int(2**m))])
            if stp not in stra_ags[i]:
                stra_ags[i].append(stp)
                c0+=1
            else:
                continue
    vir_score=[[0,0] for i in range(N)]
    real_score=[0 for i in range(N)]

    t=0

    while t<=ite:
        c1 = 0
        if t==0:
            for i in range(N):
                choice[i]=int(np.random.choice(stra_ags[i])[state])
                c1+=choice[i]
            ecs=abs(float((N-c1-c1))/N)
            if c1>N//2:
                result.append("0")
                for i in range(N):
                    if choice[i]==0:
                        real_score[i]+=ecs
                    else:
                        real_score[i]-=ecs
                for j in range(N):
                    if stra_ags[j][0][state]=="0":
                        vir_score[j][0]+=ecs
                    else:
                        vir_score[j][0]-=ecs
                    if stra_ags[j][1][state]=="0":
                        vir_score[j][1]+=ecs
                    else:
                        vir_score[j][1]-=ecs
            else:
                result.append("1")
                for i in range(N):
                    if choice[i]==1:
                        real_score[i]+=ecs
                    else:
                        real_score[i]-=ecs
                for j in range(N):
                    if stra_ags[j][0][state]=="1":
                        vir_score[j][0]+=ecs
                    else:
                        vir_score[j][0]-=ecs
                    if stra_ags[j][1][state]=="1":
                        vir_score[j][1]+=ecs
                    else:
                        vir_score[j][1]-=ecs
            memory=memory[1:]+result[-1] #Update memory and states
            state=int(2**4*int(memory[0])+2**3*int(memory[1])+2**2*int(memory[2])+2**1*int(memory[3])+int(memory[4]))
            t+=1   
        
        else:
            if t<iteq:
                for i in range(N):
                    if vir_score[i][0]>vir_score[i][1]:
                        choice[i]=int(stra_ags[i][0][state])
                        c1+=int(stra_ags[i][0][state])
                    elif vir_score[i][0]<vir_score[i][1]:
                        choice[i]=int(stra_ags[i][1][state])
                        c1+=int(stra_ags[i][1][state])
                    else:
                        choice[i]=int(np.random.choice(stra_ags[i])[state])
                        c1+=choice[i]
                ecs=abs(float((N-c1-c1))/N) #Linear Score
                        
                if c1>N//2:
                    result.append("0")
                    for i in range(N):
                        if choice[i]==0:
                            real_score[i]+=ecs
                        else:
                            real_score[i]-=ecs
                    for j in range(N):
                        if stra_ags[j][0][state]=="0":
                            vir_score[j][0]+=ecs
                        else:
                            vir_score[j][0]-=ecs
                        if stra_ags[j][1][state]=="0":
                            vir_score[j][1]+=ecs
                        else:
                            vir_score[j][1]-=ecs
                else:
                    result.append("1")
                    for i in range(N):
                        if choice[i]==1:
                            real_score[i]+=ecs
                        else:
                            real_score[i]-=ecs
                    for j in range(N):
                        if stra_ags[j][0][state]=="1":
                            vir_score[j][0]+=ecs
                        else:
                            vir_score[j][0]-=ecs
                        if stra_ags[j][1][state]=="1":
                            vir_score[j][1]+=ecs
                        else:
                            vir_score[j][1]-=ecs
                memory=memory[1:]+result[-1]
                state=int(2**4*int(memory[0])+2**3*int(memory[1])+2**2*int(memory[2])+2**1*int(memory[3])+int(memory[4]))
                st=stra_ags[real_score.index(max(real_score))]
                k=min(real_score)
                if t%tim_stp==0:
                    for i in range(N): #Eliminate poor players
                        if real_score[i]==k:
                            vir_score[i]=[0,0]
                            real_score[i]=0
                            stra_ags[i]=[np.random.choice(st)]
                            c0=0
                            while c0<1:
                                stp = "".join([np.random.choice(["0", "1"]) for k in range(int(2**m))])
                                if stp not in stra_ags[i]:
                                    stra_ags[i].append(stp)
                                    c0+=1
                                else:
                                    continue
                t+=1
                        
            elif t>=iteq and t<=ite:
                for i in range(N):
                    if vir_score[i][0]>vir_score[i][1]:
                        choice[i]=int(stra_ags[i][0][state])
                        c1+=int(stra_ags[i][0][state])
                    elif vir_score[i][0]<vir_score[i][1]:
                        choice[i]=int(stra_ags[i][1][state])
                        c1+=int(stra_ags[i][1][state])
                    else:
                        choice[i]=int(np.random.choice(stra_ags[i])[state])
                        c1+=choice[i]
                count.append(N-c1)
                ecs=abs(float((N-c1-c1))/N)
                if c1>N//2:
                    result.append("0")
                    for i in range(N):
                        if choice[i]==0:
                            real_score[i]+=ecs
                        else:
                            real_score[i]-=ecs
                    for j in range(N):
                        if stra_ags[j][0][state]=="0":
                            vir_score[j][0]+=ecs
                        else:
                            vir_score[j][0]-=ecs
                        if stra_ags[j][1][state]=="0":
                            vir_score[j][1]+=ecs
                        else:
                            vir_score[j][1]-=ecs
                else:
                    result.append("1")
                    for i in range(N):
                        if choice[i]==1:
                            real_score[i]+=ecs
                        else:
                            real_score[i]-=ecs
                    for j in range(N):
                        if stra_ags[j][0][state]=="1":
                            vir_score[j][0]+=ecs
                        else:
                            vir_score[j][0]-=ecs
                        if stra_ags[j][1][state]=="1":
                            vir_score[j][1]+=ecs
                        else:
                            vir_score[j][1]-=ecs
                memory=memory[1:]+result[-1]
                state=int(2**4*int(memory[0])+2**3*int(memory[1])+2**2*int(memory[2])+2**1*int(memory[3])+int(memory[4]))
                r_max.append(max(real_score))
                r_min.append(min(real_score))
                r_mean.append(np.average(real_score))
                st=stra_ags[real_score.index(max(real_score))]
                k=min(real_score)
                if t%tim_stp==0:
                    for i in range(N): #Eliminate poor players
                        if real_score[i]==k:
                            vir_score[i]=[0,0]
                            real_score[i]=0
                            stra_ags[i]=[np.random.choice(st)]
                            c0=0
                            while c0<1:
                                stp = "".join([np.random.choice(["0", "1"]) for k in range(int(2**m))])
                                if stp not in stra_ags[i]:
                                    stra_ags[i].append(stp)
                                    c0+=1
                                else:
                                    continue
                    
            t+=1
    return count, real_score, r_max, r_mean, r_min

# c=np.array([0 for i in range(ite-iteq)], dtype=int)
# real_sc=np.array([0 for i in range(N)], dtype=float)
# r_ml=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_ms=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_mn=np.array([0 for i in range(ite-iteq)], dtype=float)
# c1=np.array([0 for i in range(ite-iteq)], dtype=int)
# real_sc1=np.array([0 for i in range(N1)], dtype=float)
# r_ml1=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_ms1=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_mn1=np.array([0 for i in range(ite-iteq)], dtype=float)
# c2=np.array([0 for i in range(ite-iteq)], dtype=int)
# real_sc2=np.array([0 for i in range(N2)], dtype=float)
# r_ml2=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_ms2=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_mn2=np.array([0 for i in range(ite-iteq)], dtype=float)

# c4=np.array([0 for i in range(ite-iteq)], dtype=int)
# real_sc4=np.array([0 for i in range(N)], dtype=float)
# r_ml4=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_ms4=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_mn4=np.array([0 for i in range(ite-iteq)], dtype=float)
# c5=np.array([0 for i in range(ite-iteq)], dtype=int)
# real_sc5=np.array([0 for i in range(N1)], dtype=float)
# r_ml5=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_ms5=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_mn5=np.array([0 for i in range(ite-iteq)], dtype=float)
# c6=np.array([0 for i in range(ite-iteq)], dtype=int)
# real_sc6=np.array([0 for i in range(N2)], dtype=float)
# r_ml6=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_ms6=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_mn6=np.array([0 for i in range(ite-iteq)], dtype=float)

# c8=np.array([0 for i in range(ite-iteq)], dtype=int)
# real_sc8=np.array([0 for i in range(N)], dtype=float)
# r_ml8=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_ms8=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_mn8=np.array([0 for i in range(ite-iteq)], dtype=float)
# c9=np.array([0 for i in range(ite-iteq)], dtype=int)
# real_sc9=np.array([0 for i in range(N1)], dtype=float)
# r_ml9=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_ms9=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_mn9=np.array([0 for i in range(ite-iteq)], dtype=float)
# c101=np.array([0 for i in range(ite-iteq)], dtype=int)
# real_sc101=np.array([0 for i in range(N2)], dtype=float)
# r_ml101=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_ms101=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_mn101=np.array([0 for i in range(ite-iteq)], dtype=float)


# c103=np.array([0 for i in range(ite-iteq)], dtype=int)
# real_sc103=np.array([0 for i in range(N)], dtype=float)
# r_ml103=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_ms103=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_mn103=np.array([0 for i in range(ite-iteq)], dtype=float)
# c104=np.array([0 for i in range(ite-iteq)], dtype=int)
# real_sc104=np.array([0 for i in range(N1)], dtype=float)
# r_ml104=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_ms104=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_mn104=np.array([0 for i in range(ite-iteq)], dtype=float)
# c105=np.array([0 for i in range(ite-iteq)], dtype=int)
# real_sc105=np.array([0 for i in range(N2)], dtype=float)
# r_ml105=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_ms105=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_mn105=np.array([0 for i in range(ite-iteq)], dtype=float)


# c3=np.array([0 for i in range(ite-iteq)], dtype=int)
# r_ml3=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_ms3=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_mn3=np.array([0 for i in range(ite-iteq)], dtype=float)
# c7=np.array([0 for i in range(ite-iteq)], dtype=int)
# r_ml7=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_ms7=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_mn7=np.array([0 for i in range(ite-iteq)], dtype=float)
# c102=np.array([0 for i in range(ite-iteq)], dtype=int)
# r_ml102=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_ms102=np.array([0 for i in range(ite-iteq)], dtype=float)
# r_mn102=np.array([0 for i in range(ite-iteq)], dtype=float)

# for i in range(nsam):
#     # c0, real_sc0, r_ml0, r_mn0, r_ms0=simulate_minority(N=N, ite=ite, iteq=iteq)
#     # c+=c0
#     # real_sc+=real_sc0
#     # r_ml+=r_ml0
#     # r_mn+=r_mn0
#     # r_ms+=r_ms0
#     # c01, real_sc01, r_ml01, r_mn01, r_ms01=simulate_minority(N=N1, ite=ite, iteq=iteq)
#     # c1+=c01
#     # real_sc1+=real_sc01
#     # r_ml1+=r_ml01
#     # r_mn1+=r_mn01
#     # r_ms1+=r_ms01
#     # c02, real_sc02, r_ml02, r_mn02, r_ms02=simulate_minority(N=N2, ite=ite, iteq=iteq)
#     # c2+=c02
#     # real_sc2+=real_sc02
#     # r_ml2+=r_ml02
#     # r_mn2+=r_mn02
#     # r_ms2+=r_ms02
#     c10, real_sc10, r_ml10, r_mn10, r_ms10=simulate_minority2(N=N, ite=ite, iteq=iteq, tim_stp=tim_stp, p_level=p_level)
#     c4+=c10
#     real_sc4+=real_sc10
#     r_ml4+=r_ml10
#     r_mn4+=r_mn10
#     r_ms4+=r_ms10
#     # c11, real_sc11, r_ml11, r_mn11, r_ms11=simulate_minority2(N=N1, ite=ite, iteq=iteq, tim_stp=tim_stp, p_level=p_level)
#     # c5+=c11
#     # real_sc5+=real_sc11
#     # r_ml5+=r_ml11
#     # r_mn5+=r_mn11
#     # r_ms5+=r_ms11
#     # c12, real_sc12, r_ml12, r_mn12, r_ms12=simulate_minority2(N=N2, ite=ite, iteq=iteq, tim_stp=tim_stp, p_level=p_level)
#     # c6+=c12
#     # real_sc6+=real_sc12
#     # r_ml6+=r_ml12
#     # r_mn6+=r_mn12
#     # r_ms6+=r_ms12
#     # c20, real_sc20, r_ml20, r_mn20, r_ms20=simulate_minority3(N=N, R=ite, iteq=iteq)
#     # c8+=c20
#     # real_sc8+=real_sc20
#     # r_ml8+=r_ml20
#     # r_mn8+=r_mn20
#     # r_ms8+=r_ms20
#     # c21, real_sc21, r_ml21, r_mn21, r_ms21=simulate_minority3(N=N1, R=ite, iteq=iteq)
#     # c9+=c21
#     # real_sc9+=real_sc21
#     # r_ml9+=r_ml21
#     # r_mn9+=r_mn21
#     # r_ms2+=r_ms21
#     # c22, real_sc22, r_ml22, r_mn22, r_ms22=simulate_minority3(N=N2, R=ite, iteq=iteq)
#     # c101+=c22
#     # real_sc101+=real_sc22
#     # r_ml101+=r_ml22
#     # r_mn101+=r_mn22
#     # r_ms101+=r_ms22
#     c30, r_ml30, r_mn30, r_ms30=simulate_minority4(N=N, ite=ite, iteq=iteq)
#     c3+=c30
#     r_ml3+=r_ml30
#     r_mn3+=r_mn30
#     r_ms3+=r_ms30
#     # c31, r_ml31, r_mn31, r_ms31=simulate_minority4(N=N1, ite=ite, iteq=iteq)
#     # c7+=c31
#     # r_ml7+=r_ml31
#     # r_mn7+=r_mn31
#     # r_ms7+=r_ms31
#     # c32, r_ml32, r_mn32, r_ms32=simulate_minority4(N=N2, ite=ite, iteq=iteq)
#     # c102+=c32
#     # r_ml102+=r_ml32
#     # r_mn102+=r_mn32
#     # r_ms102+=r_ms32
#     # c40, real_sc40, r_ml40, r_mn40, r_ms40=simulate_minority5(N=N, ite=ite, iteq=iteq, tim_stp=tim_stp)
#     # c103+=c40
#     # real_sc103+=real_sc40
#     # r_ml103+=r_ml40
#     # r_mn103+=r_mn40
#     # r_ms103+=r_ms40
#     # c41, real_sc41, r_ml41, r_mn41, r_ms41=simulate_minority5(N=N1, ite=ite, iteq=iteq, tim_stp=tim_stp)
#     # c104+=c41
#     # real_sc104+=real_sc41
#     # r_ml104+=r_ml41
#     # r_mn104+=r_mn41
#     # r_ms104+=r_ms41
#     # c42, real_sc42, r_ml42, r_mn42, r_ms42=simulate_minority5(N=N2, ite=ite, iteq=iteq, tim_stp=tim_stp)
#     # c105+=c42
#     # real_sc105+=real_sc42
#     # r_ml105+=r_ml42
#     # r_mn105+=r_mn42
#     # r_ms105+=r_ms42

# c=c/nsam
# real_sc=real_sc/nsam
# r_ml=r_ml/nsam
# r_mn=r_mn/nsam
# r_ms=r_ms/nsam
# # c1=c1/nsam
# # real_sc1=real_sc1/nsam
# # r_ml1=r_ml1/nsam
# # r_mn1=r_mn1/nsam
# # r_ms1=r_ms1/nsam
# # c2=c2/nsam
# # real_sc2=real_sc2/nsam
# # r_ml2=r_ml2/nsam
# # r_mn2=r_mn2/nsam
# # r_ms2=r_ms2/nsam

# c4=c4/nsam
# real_sc4=real_sc4/nsam
# r_ml4=r_ml4/nsam
# r_mn4=r_mn4/nsam
# r_ms4=r_ms4/nsam
# # c5=c5/nsam
# # real_sc5=real_sc5/nsam
# # r_ml5=r_ml5/nsam
# # r_mn5=r_mn5/nsam
# # r_ms5=r_ms5/nsam
# # c6=c6/nsam
# # real_sc6=real_sc6/nsam
# # r_ml6=r_ml6/nsam
# # r_mn6=r_mn6/nsam
# # r_ms6=r_ms6/nsam

# c8=c8/nsam
# real_sc8=real_sc8/nsam
# r_ml8=r_ml8/nsam
# r_mn8=r_mn8/nsam
# r_ms8=r_ms8/nsam
# # c9=c9/nsam
# # real_sc9=real_sc9/nsam
# # r_ml9=r_ml9/nsam
# # r_mn9=r_mn9/nsam
# # r_ms9=r_ms9/nsam
# # c101=c101/nsam
# # real_sc101=real_sc101/nsam
# # r_ml101=r_ml101/nsam
# # r_mn101=r_mn101/nsam
# # r_ms101=r_ms101/nsam

# c3=c3/nsam
# r_ml3=r_ml3/nsam
# r_mn3=r_mn3/nsam
# r_ms3=r_ms3/nsam
# # c7=c7/nsam
# # r_ml7=r_ml7/nsam
# # r_mn7=r_mn7/nsam
# # r_ms7=r_ms7/nsam
# # c102=c102/nsam
# # r_ml102=r_ml102/nsam
# # r_mn102=r_mn102/nsam
# # r_ms102=r_ms102/nsam

# # c104=c104/nsam
# # real_sc104=real_sc104/nsam
# # r_ml104=r_ml104/nsam
# # r_mn104=r_mn104/nsam
# # r_ms104=r_ms104/nsam
# c103=c103/nsam
# real_sc103=real_sc103/nsam
# r_ml103=r_ml103/nsam
# r_mn103=r_mn103/nsam
# r_ms103=r_ms103/nsam
# # c105=c105/nsam
# # real_sc105=real_sc105/nsam
# # r_ml105=r_ml105/nsam
# # r_mn105=r_mn105/nsam
# # r_ms105=r_ms105/nsam



# t=np.linspace(iteq, ite, ite-iteq, True)
# # fig=plt.figure(figsize=(16, 8), dpi=100)
# # # plt.plot(t, c, color="r", label='With Threshold')
# # plt.plot(t, c4, color="g", label='EMG(V2)')
# # plt.plot(t, c8, color="blue", label='EMG(V3)')
# # plt.plot(t, c3, color="y", label='Basic MG')
# # plt.title('Attendance Number')
# # plt.legend()
# # plt.xlim(iteq, ite)
# # plt.ylim(80, 120)
# # plt.show()

# # titles=['EMG(V1)', 'EMG(V2)', 'EMG(V3)', 'EMG(V4)', 'Basic MG']
# # t=np.linspace(iteq, ite, ite-iteq, True)
# # c_list=[c, c1, c2, c3]
# # c_list2=[c103, c104, c105, c8, c9, c101, c, c1, c2, c4, c5, c6, c3, c7, c102]

# # # fig, axs=plt.subplots(2,2, figsize=(16,10))
# # # for i in range(2):
# # #     for j in range(2):
# # #         ax=axs[i, j]
# # #         ax.plot(t, c_list[2*i+j], color="blue")
# # #         ax.set_title(titles[i*2+j])
# # #         ax.set_xlabel('Rounds')
# # #         ax.set_ylabel('Attendance Number')
# # #         ax.set_xlim(iteq, ite)
# # #         ax.set_ylim(80, 120)
# # # fig.suptitle("Attendance Number")
# # # plt.show()

# # fig, axs=plt.subplots(2,3, figsize=(16,10))
# # for i in range(2):
# #     for j in range(3):
# #         if i==1 and j==2:
# #             ax=axs[1, 2]
# #             ax.axis('off')
# #             break
# #         ax=axs[i, j]
# #         ax.plot(t, c_list2[3*j+9*i], color="blue", label='N=201')
# #         # ax.plot(t, c_list2[3*j+9*i+1], color="r", label='N=351')
# #         # ax.plot(t, c_list2[3*j+9*i+2], color="g", label='N=501')
# #         ax.legend()
# #         ax.set_title(titles[i*3+j])
# #         ax.set_xlabel('Rounds')
# #         ax.set_ylabel('Attendance Number')
# #         ax.set_xlim(iteq, ite)
# #         ax.set_ylim(80, 120)
# #         # ax.set_ylim(80, 300)

# # fig.suptitle("Attendance Number")
# # plt.show()


# fig2=plt.figure(figsize=(16, 8), dpi=100)
# # plt.plot(t, r_ml, color="r", label="Best Agent, With Threshold")
# # plt.plot(t, r_mn, color="r", label="Mean Wealth, With Threshold")
# # plt.plot(t, r_ms, color="r", label="Worst Agent, With Threshold")
# # plt.plot(t, r_ms103, color="g", label="EMG(V1)")
# # plt.plot(t, r_mn1, color="g", label="Mean Wealth, EMG(V1)")
# # plt.plot(t, r_ms1, color="g", label="Worst Agent, EMG(V1)")
# plt.plot(t, r_ms8, color="blue", label="EMG(V2)")
# # plt.plot(t, r_mn2, color="blue", label="Mean Wealth, EMG(V2)")
# # plt.plot(t, r_ms2, color="blue", label="Worst Agent, EMG(V2)")
# # plt.plot(t, r_ms, color="r", label="EMG(V3)")
# # plt.plot(t, r_ms4, color="k", label="EMG(V4)")
# # plt.plot(t, r_ms3, color="y", label="Basic MG")
# # plt.plot(t, r_mn3, color="y", label="Mean Wealth, Basic MG")
# # plt.plot(t, r_ms3, color="y", label="Worst Agent, Basic MG")
# plt.xlim(iteq, ite)
# plt.legend()
# plt.xlabel("Rounds")
# plt.ylabel("Scores")
# # plt.ylim(-70,0)
# plt.ylim(-40,20)
# plt.xlim(iteq, ite)
# # plt.ylim(-20,0)
# # plt.ylim(-0.25, 1)
# # plt.ylim(0, 70)
# # plt.ylim(-125, 75)
# # plt.ylim(-40, 20)
# # plt.ylim(-40, 100)
# # plt.ylim(-400,400)
# # plt.title('Scores of Best, Worst, and Mean Agents')
# plt.title('Scores of Worst Agents')
# plt.show()

N_list=[]
variances1=[]
v_st1=[]
variances2=[]
v_st2=[]
variances3=[]
v_st3=[]
variances4=[]
v_st4=[]
variances5=[]
v_st5=[]
for N in np.array(np.linspace(21, 301, 15, True), dtype=int):
    vtmp1=[]
    vtmp2=[]
    vtmp3=[]
    vtmp4=[]
    vtmp5=[]
    for i in range(nsam):
        if N%2==1:
            c1, real_sc1, r_ml1, r_mn1, r_ms1=simulate_minority(N=N, ite=ite, iteq=iteq)
            c2, real_sc2, r_ml2, r_mn2, r_ms2=simulate_minority2(N=N, ite=ite, iteq=iteq, tim_stp=tim_stp, p_level=p_level)
            c3, real_sc3, r_ml3, r_mn3, r_ms3=simulate_minority3(N=N, R=ite, iteq=iteq)
            c4, r_ml4, r_mn4, r_ms4=simulate_minority4(N=N, ite=ite, iteq=iteq)
            c5, real_sc5, r_ml5, r_mn5, r_ms5=simulate_minority5(N=N, ite=ite, iteq=iteq, tim_stp=tim_stp)
            vtmp1.append(np.var(np.array(c1)))
            vtmp2.append(np.var(np.array(c2)))
            vtmp3.append(np.var(np.array(c3)))
            vtmp4.append(np.var(np.array(c4)))
            vtmp5.append(np.var(np.array(c5)))
        else:
            N+=1
            c1, real_sc1, r_ml1, r_mn1, r_ms1=simulate_minority(N=N, ite=ite, iteq=iteq)
            c2, real_sc2, r_ml2, r_mn2, r_ms2=simulate_minority2(N=N, ite=ite, iteq=iteq, tim_stp=tim_stp, p_level=p_level)
            c3, real_sc3, r_ml3, r_mn3, r_ms3=simulate_minority3(N=N, R=ite, iteq=iteq)
            c4, r_ml4, r_mn4, r_ms4=simulate_minority4(N=N, ite=ite, iteq=iteq)
            c5, real_sc5, r_ml5, r_mn5, r_ms5=simulate_minority5(N=N, ite=ite, iteq=iteq, tim_stp=tim_stp)
            vtmp1.append(np.var(np.array(c1)))
            vtmp2.append(np.var(np.array(c2)))
            vtmp3.append(np.var(np.array(c3)))
            vtmp4.append(np.var(np.array(c4)))
            vtmp5.append(np.var(np.array(c5)))
    N_list.append(N)
    v_st1.append(vtmp1)
    variances1.append(np.average(np.array(vtmp1)))
    v_st2.append(vtmp2)
    variances2.append(np.average(np.array(vtmp2)))
    v_st3.append(vtmp3)
    variances3.append(np.average(np.array(vtmp3)))
    v_st4.append(vtmp4)
    variances4.append(np.average(np.array(vtmp4)))
    v_st5.append(vtmp5)
    variances5.append(np.average(np.array(vtmp5)))

fig4=plt.figure(figsize=(16, 8), dpi=100)
plt.scatter(2**5/np.array(N_list), np.array(variances1)/np.array(N_list), color="blue", marker='o', label='EMG(V3)')
plt.scatter(2**5/np.array(N_list), np.array(variances2)/np.array(N_list), color="r", marker='^', label='EMG(V4)')
plt.scatter(2**5/np.array(N_list), np.array(variances3)/np.array(N_list), color="black", marker='s', label='EMG(V2)')
plt.scatter(2**5/np.array(N_list), np.array(variances4)/np.array(N_list), color="g", marker='>', label='Basic MG')
plt.scatter(2**5/np.array(N_list), np.array(variances5)/np.array(N_list), color="y", marker='*', label='EMG(V1)')
plt.xlabel('$2^m$/N')
plt.ylabel('$\sigma^2$/N')
plt.title('$\sigma^2$/N Versus $2^m$/N')
plt.xlim(0.1,)
plt.ylim(0, 0.5)
plt.legend()
plt.show()
psr1=list(np.array(variances1)/np.array(N_list))
psr2=list(np.array(variances2)/np.array(N_list))
psr3=list(np.array(variances3)/np.array(N_list))
psr4=list(np.array(variances4)/np.array(N_list))
psr5=list(np.array(variances5)/np.array(N_list))
print('The minimum value in EMG(V3) is obtained at {}'.format(2**5/N_list[psr1.index(min(psr1))]))
print('The minimum value in EMG(V4) is obtained at {}'.format(2**5/N_list[psr2.index(min(psr2))]))
print('The minimum value in EMG(V2) is obtained at {}'.format(2**5/N_list[psr3.index(min(psr3))]))
print('The minimum value in basic MG is obtained at {}'.format(2**5/N_list[psr4.index(min(psr4))]))
print('The minimum value in EMG(V1) is obtained at {}'.format(2**5/N_list[psr5.index(min(psr5))]))

