# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import matplotlib.pyplot as plt

nsam=10
iteq=1200
ite=10000
N=201
tim_stp=50
p_level=0.1

def simulate_minority(N, ite, iteq, tim_stp, p_level):
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
                        c1+=int(np.random.choice(stra_ags[i])[state])
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
                        c1+=int(np.random.choice(stra_ags[i])[state])
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

N1=501
N2=351
c=np.array([0 for i in range(ite-iteq)], dtype=int)
real_sc=np.array([0 for i in range(N)], dtype=float)
r_ml=np.array([0 for i in range(ite-iteq)], dtype=float)
r_ms=np.array([0 for i in range(ite-iteq)], dtype=float)
r_mn=np.array([0 for i in range(ite-iteq)], dtype=float)
c1=np.array([0 for i in range(ite-iteq)], dtype=int)
real_sc1=np.array([0 for i in range(N)], dtype=float)
r_ml1=np.array([0 for i in range(ite-iteq)], dtype=float)
r_ms1=np.array([0 for i in range(ite-iteq)], dtype=float)
r_mn1=np.array([0 for i in range(ite-iteq)], dtype=float)
c2=np.array([0 for i in range(ite-iteq)], dtype=int)
real_sc2=np.array([0 for i in range(N)], dtype=float)
r_ml2=np.array([0 for i in range(ite-iteq)], dtype=float)
r_ms2=np.array([0 for i in range(ite-iteq)], dtype=float)
r_mn2=np.array([0 for i in range(ite-iteq)], dtype=float)
c3=np.array([0 for i in range(ite-iteq)], dtype=int)
real_sc3=np.array([0 for i in range(N)], dtype=float)
r_ml3=np.array([0 for i in range(ite-iteq)], dtype=float)
r_ms3=np.array([0 for i in range(ite-iteq)], dtype=float)
r_mn3=np.array([0 for i in range(ite-iteq)], dtype=float)
v_st=[]


for i in range(nsam):
    c0, real_sc0, r_ml0, r_mn0, r_ms0=simulate_minority(N=N, ite=ite, iteq=iteq, tim_stp=tim_stp, p_level=p_level)
    c+=c0
    real_sc+=real_sc0
    r_ml+=r_ml0
    r_mn+=r_mn0
    r_ms+=r_ms0
    c10, real_sc10, r_ml10, r_mn10, r_ms10=simulate_minority(N=N, ite=ite, iteq=iteq, tim_stp=tim_stp, p_level=0.3)
    c1+=c10
    real_sc1+=real_sc10
    r_ml1+=r_ml10
    r_mn1+=r_mn10
    r_ms1+=r_ms10
    c20, real_sc20, r_ml20, r_mn20, r_ms20=simulate_minority(N=N, ite=ite, iteq=iteq, tim_stp=tim_stp, p_level=0.4)
    c2+=c20
    real_sc2+=real_sc20
    r_ml2+=r_ml20
    r_mn2+=r_mn20
    r_ms2+=r_ms20
    c30, real_sc30, r_ml30, r_mn30, r_ms30=simulate_minority(N=N, ite=ite, iteq=iteq, tim_stp=tim_stp, p_level=0.5)
    c3+=c30
    real_sc3+=real_sc30
    r_ml3+=r_ml30
    r_mn3+=r_mn30
    r_ms3+=r_ms30
c=c/nsam
real_sc=real_sc/nsam
r_ml=r_ml/nsam
r_mn=r_mn/nsam
r_ms=r_ms/nsam
c1=c1/nsam
real_sc1=real_sc1/nsam
r_ml1=r_ml1/nsam
r_mn1=r_mn1/nsam
r_ms1=r_ms1/nsam
c2=c2/nsam
real_sc2=real_sc2/nsam
r_ml2=r_ml2/nsam
r_mn2=r_mn2/nsam
r_ms2=r_ms2/nsam
c3=c3/nsam
real_sc3=real_sc3/nsam
r_ml3=r_ml3/nsam
r_mn3=r_mn3/nsam
r_ms3=r_ms3/nsam


#N_list=[]
#variances=[]
#for N in np.array(np.linspace(51, 301, 15, True), dtype=int):
 #   vtmp=[]
  #  for i in range(nsam):
   #     if N%2==1:
    #        c1, real_sc1, r_ml1, r_mn1, r_ms1=simulate_minority(N=N, ite=ite, iteq=iteq)
     #       vtmp.append(np.var(np.array(c1)))
      #  else:
       #     N+=1
        #    c1, real_sc1, r_ml1, r_mn1, r_ms1=simulate_minority(N=N, ite=ite, iteq=iteq)
         #   vtmp.append(np.var(np.array(c1)))
    #N_list.append(N)
    #v_st.append(vtmp)
    #variances.append(np.average(np.array(vtmp)))

t=np.linspace(iteq, ite, ite-iteq, True)
fig1=plt.figure(figsize=(16, 8), dpi=100)
plt.plot(t, c, color="r", label="p=0.1")
plt.plot(t, c1, color="blue", label="p=0.3")
plt.plot(t, c2, color="y", label="p=0.4")
plt.plot(t, c2, color="g", label="p=0.5")
plt.xlim(iteq, ite)
plt.legend()
plt.xlabel("Rounds")
plt.ylabel("Attendance Number(A(t))")
plt.title('Attendance Number')
plt.show()

#fig2=plt.figure(figsize=(16, 8), dpi=100)
#ags=np.linspace(1, N, N, True)
#plt.plot(ags, real_sc, color="blue")
#plt.xlabel("Agents")
#plt.ylabel('Scores')
#plt.xlim(1, N)
#plt.show()

fig3=plt.figure(figsize=(16, 8), dpi=100)
plt.plot(t, r_ml, color="r", label="Best Agent, p=0.1")
plt.plot(t, r_mn, color="r", label="Mean Wealth, p=0.1")
plt.plot(t, r_ms, color="r", label="Worst Agent, p=0.1")
plt.plot(t, r_ml1, color="y", label="Best Agent, p=0.3")
plt.plot(t, r_mn1, color="y", label="Mean Wealth, p=0.3")
plt.plot(t, r_ms1, color="y", label="Worst Agent, p=0.3")
plt.plot(t, r_ml2, color="blue", label="Best Agent, p=0.4")
plt.plot(t, r_mn2, color="blue", label="Mean Wealth, p=0.4")
plt.plot(t, r_ms2, color="blue", label="Worst Agent, p=0.4")
plt.plot(t, r_ml3, color="g", label="Best Agent, p=0.5")
plt.plot(t, r_mn3, color="g", label="Mean Wealth, p=0.5")
plt.plot(t, r_ms3, color="g", label="Worst Agent, p=0.5")
plt.xlim(iteq, ite)
plt.legend()
plt.xlabel("Rounds")
plt.ylabel("Wealth")
plt.title('Scores of Best, Worst, and Mean Agents')
plt.show()


#fig4=plt.figure(figsize=(16, 8), dpi=100)
#plt.scatter(2**5/np.array(N_list), np.array(variances)/np.array(N_list), color="blue")
#plt.xlabel("$\frac{2^5}{N}$")
#plt.ylabel('$\frac{\sigma^2}{N}$')
#plt.show()
#psr=list(np.array(variances)/np.array(N_list))
#print('The minimum value is obtained at {}'.format(2**5/N_list[psr.index(min(psr))]))
