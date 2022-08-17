import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import xlwt
import random
import fancoilunit_coilingload
"""

"""
#####################  hyper parameters  ####################

LR_ACTOR = 0.001
LR_CRITIC = 0.002
GAMMA = 0.001
TAU = 0.01
MEMORY_CAPACITY = 2000
BATCH_SIZE = 64


########################## DDPG Framework ######################
class ActorNet(nn.Module):  # define the network structure for actor and critic
    def __init__(self, s_dim, a_dim):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(s_dim, 30)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization of FC1
        self.out = nn.Linear(30, a_dim)
        self.out.weight.data.normal_(0, 0.1)  # initilizaiton of OUT

    # s_dim=1
    # a_dim=2
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = torch.tanh(x/5)
        actions = x * 2  # for the game "Pendulum-v0", action range is [-2, 2]

        return actions


class CriticNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(CriticNet, self).__init__()
        self.fcs = nn.Linear(s_dim, 30)
        self.fcs.weight.data.normal_(0, 0.1)
        self.fca = nn.Linear(a_dim, 30)
        self.fca.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, 1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        x = self.fcs(s)
        y = self.fca(a)
        actions_value = self.out(F.relu(x + y))
        return actions_value


class DDPG(object):
    def __init__(self, a_dim, s_dim, ):
        self.a_dim, self.s_dim,  = a_dim, s_dim,
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0  # serves as updating the memory data
        # Create the 4 network objects
        self.actor_eval = ActorNet(s_dim, a_dim)
        self.actor_target = ActorNet(s_dim, a_dim)
        self.critic_eval = CriticNet(s_dim, a_dim)
        self.critic_target = CriticNet(s_dim, a_dim)
        # create 2 optimizers for actor and critic
        self.actor_optimizer = torch.optim.Adam(self.actor_eval.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic_eval.parameters(), lr=LR_CRITIC)
        # Define the loss function for critic network update
        self.loss_func = nn.MSELoss()

    def store_transition(self, s, a, r, s_):  # how to store the episodic data to buffer
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old data with new data
        self.memory[index, :] = transition
        self.pointer += 1

    def choose_action(self, s):
        # print(s)
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        return self.actor_eval(s)[0].detach()

    def learn(self):
        # softly update the target networks
        for x in self.actor_target.state_dict().keys():
            eval('self.actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.actor_target.' + x + '.data.add_(TAU*self.actor_eval.' + x + '.data)')
        for x in self.critic_target.state_dict().keys():
            eval('self.critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.critic_target.' + x + '.data.add_(TAU*self.critic_eval.' + x + '.data)')
            # sample from buffer a mini-batch data
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        batch_trans = self.memory[indices, :]
        # extract data from mini-batch of transitions including s, a, r, s_
        batch_s = torch.FloatTensor(batch_trans[:, :self.s_dim])
        batch_a = torch.FloatTensor(batch_trans[:, self.s_dim:self.s_dim + self.a_dim])
        batch_r = torch.FloatTensor(batch_trans[:, -self.s_dim - 1: -self.s_dim])
        batch_s_ = torch.FloatTensor(batch_trans[:, -self.s_dim:])
        # make action and evaluate its action values
        a = self.actor_eval(batch_s)
        q = self.critic_eval(batch_s, a)
        actor_loss = -torch.mean(q)
        # optimize the loss of actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # compute the target Q value using the information of next state
        a_target = self.actor_target(batch_s_)
        q_tmp = self.critic_target(batch_s_, a_target)
        q_target = batch_r + GAMMA * q_tmp
        # compute the current q value and the loss
        q_eval = self.critic_eval(batch_s, batch_a)
        td_error = self.loss_func(q_target, q_eval)
        # optimize the loss of critic network
        self.critic_optimizer.zero_grad()
        td_error.backward()
        self.critic_optimizer.step()


############################### Training ######################################
# Define the env in gym

workbook = xlwt.Workbook(encoding = 'utf-8')
# 创建一个worksheet
worksheet_reward = workbook.add_sheet('reward')
worksheet_a_water = workbook.add_sheet('a_water')
worksheet_a_air = workbook.add_sheet('a_air')
worksheet_Pfan = workbook.add_sheet('Pfan')
worksheet_Ppump = workbook.add_sheet('Ppump')

worksheet_Q = workbook.add_sheet('Q')
worksheet_PfandPp = workbook.add_sheet('PfandPp')
worksheet_ccd_Q = workbook.add_sheet('ccd_Q')
total_reward = 0
total_power = 0
k=300
s_dim = 1
a_dim = 2
a_water_bound_low=210
a_water_bound_high=580
a_air_bound_low=210
a_air_bound_high=1020
state = np.load('state_4.npy')
ddpg_base=DDPG(a_dim,s_dim)



var_base_0 = 3  # the controller of exploration which will decay during training process
var_base_1 = 3
var_limiti=0.15
var_pre=0.9995

EPS             = 1
EPS_MIN         = 0.01
EPS_DECAY       = 0.00005


for i_episode in range(0,20):
    for i in range(4392):
        s = state[i:i + 1, i_episode:i_episode+1]
        s_ = state[i + 1:i + 2, i_episode:i_episode+1]
        if s > 0:
            s_normalized = s / 5000
            s__normalized = s_ / 5000
            rand_num = random.random()
            EPS = EPS - EPS_DECAY
            if rand_num < EPS:
                if s<2700:#(200-300,200-400)水 空气 1
                    a = ddpg_1.choose_action(s_normalized)
                    a = (a.numpy()).squeeze()
                    a[0] = np.clip(np.random.normal(a[0], var_1_0), -2, 2)
                    a[1] = np.clip(np.random.normal(a[1], var_1_1), -2, 2)
                    a_water = a[0] * 31.25 + 250
                    a_air = a[1] * 62.5 + 300
                    environment = fancoilunit_coilingload.fan_coil_unit(cls=s, tw1=7, V_air=a_air / 3600,
                                                                        W_water=a_water / 3600, t_in_dryair=28,
                                                                        t_in_relative_humidity=0.6)
                    cls, Q, t2, d2, tw2, W_water, V_air, Ppump, Pfan = environment.approch()
                    # 水流量：(210-580)step5  75；风量：（210-1020）step10  82
                    if a_water >= 210 and a_water <= 580 and a_air >= 210 and a_air <= 1020:
                        Punish_coefficient = 0
                    else:
                        Punish_coefficient = 1000
                    Punish_coefficient_1 = k - k * math.exp(-((Q - s) ** 2) / (2 * (s * 0.1) ** 2))
                    r = -Punish_coefficient_1 - (Pfan + Ppump) - Punish_coefficient
                if s >= 2700 and s<3500:#(300-400,400-600)水 空气 2
                    a = ddpg_2.choose_action(s_normalized)
                    a = (a.numpy()).squeeze()
                    a[0] = np.clip(np.random.normal(a[0], var_2_0), -2, 2)
                    a[1] = np.clip(np.random.normal(a[1], var_2_1), -2, 2)
                    a_water = a[0] * 31.25 + 350
                    a_air = a[1] * 62.5 + 500
                    environment = fancoilunit_coilingload.fan_coil_unit(cls=s, tw1=7, V_air=a_air / 3600,
                                                                        W_water=a_water / 3600, t_in_dryair=28,
                                                                        t_in_relative_humidity=0.6)
                    cls, Q, t2, d2, tw2, W_water, V_air, Ppump, Pfan = environment.approch()
                    # 水流量：(210-580)step5  75；风量：（210-1020）step10  82
                    if a_water >= 210 and a_water <= 580 and a_air >= 210 and a_air <= 1020:
                        Punish_coefficient = 0
                    else:
                        Punish_coefficient = 1000
                    Punish_coefficient_1 = k - k * math.exp(-((Q - s) ** 2) / (2 * (s * 0.1) ** 2))
                    r = -Punish_coefficient_1 - (Pfan + Ppump) - Punish_coefficient

                if s >=3500 and s<4100:#(400-500,600-800)水 空气3
                    a = ddpg_3.choose_action(s_normalized)
                    a = (a.numpy()).squeeze()
                    a[0] = np.clip(np.random.normal(a[0], var_3_0), -2, 2)
                    a[1] = np.clip(np.random.normal(a[1], var_3_1), -2, 2)
                    a_water = a[0] * 31.25 + 450
                    a_air = a[1] * 62.5 + 700
                    environment = fancoilunit_coilingload.fan_coil_unit(cls=s, tw1=7, V_air=a_air / 3600,
                                                                        W_water=a_water / 3600, t_in_dryair=28,
                                                                        t_in_relative_humidity=0.6)
                    cls, Q, t2, d2, tw2, W_water, V_air, Ppump, Pfan = environment.approch()
                    # 水流量：(210-580)step5  75；风量：（210-1020）step10  82
                    if a_water >= 210 and a_water <= 580 and a_air >= 210 and a_air <= 1020:
                        Punish_coefficient = 0
                    else:
                        Punish_coefficient = 1000
                    Punish_coefficient_1 = k - k * math.exp(-((Q - s) ** 2) / (2 * (s * 0.1) ** 2))
                    r = -Punish_coefficient_1 - (Pfan + Ppump) - Punish_coefficient

                if s >= 4100:#(500-600,800-1000) 水 空气4
                    a = ddpg_4.choose_action(s_normalized)
                    a = (a.numpy()).squeeze()
                    a[0] = np.clip(np.random.normal(a[0], var_4_0), -2, 2)
                    a[1] = np.clip(np.random.normal(a[1], var_4_1), -2, 2)
                    a_water = a[0] * 31.25 + 550
                    a_air = a[1] * 62.5 + 900
                    environment = fancoilunit_coilingload.fan_coil_unit(cls=s, tw1=7, V_air=a_air / 3600,
                                                                        W_water=a_water / 3600, t_in_dryair=28,
                                                                        t_in_relative_humidity=0.6)
                    cls, Q, t2, d2, tw2, W_water, V_air, Ppump, Pfan = environment.approch()
                    # 水流量：(210-580)step5  75；风量：（210-1020）step10  82
                    if a_water >= 210 and a_water <= 580 and a_air >= 210 and a_air <= 1020:
                        Punish_coefficient = 0
                    else:
                        Punish_coefficient = 1000
                    Punish_coefficient_1 = k - k * math.exp(-((Q - s) ** 2) / (2 * (s * 0.1) ** 2))
                    r = -Punish_coefficient_1 - (Pfan + Ppump) - Punish_coefficient

                # a_water = a[0] * 115.625 + 395
                # a_air = a[1] * 253.125 + 615
                a[0]=(a_water-395)/115.625
                a[1]=(a_air-615)/253.125
                ddpg_base.store_transition(s_normalized.squeeze(), a, r.squeeze() / 1500,
                                        s__normalized.squeeze())  # store the transition to memory
                if ddpg_base.pointer > MEMORY_CAPACITY:
                    if var_base_0 > var_limiti:
                        var_base_0 *= var_pre  # decay the exploration controller factor
                        var_base_1 *= var_pre
                    else:
                        var_base_0 = var_limiti  # decay the exploration controller factor
                        var_base_1 = var_limiti
                    ddpg_base.learn()
            else:
                a = ddpg_base.choose_action(s_normalized)
                a = (a.numpy()).squeeze()
                # print(a)
                a[0] = np.clip(np.random.normal(a[0], var_base_0), -2, 2)
                a[1] = np.clip(np.random.normal(a[1], var_base_1), -2, 2)
                a_water = a[0] * 115.625 + 395
                a_air = a[1] * 253.125 + 615
                environment = fancoilunit_coilingload.fan_coil_unit(cls=s, tw1=7, V_air=a_air / 3600,
                                                                    W_water=a_water / 3600, t_in_dryair=28,
                                                                    t_in_relative_humidity=0.6)
                cls, Q, t2, d2, tw2, W_water, V_air, Ppump, Pfan = environment.approch()
                # 水流量：(210-580)step5  75；风量：（210-1020）step10  82
                if a_water >= 210 and a_water <= 580 and a_air >= 210 and a_air <= 1020:
                    Punish_coefficient = 0
                else:
                    Punish_coefficient = 1000
                # s_up=s+s*0.05
                # s_down=s-s*0.05
                # if Q<=s_up and Q>=s_down:
                #     Punish_coefficient_1=0
                # else:
                #     Punish_coefficient_1=500
                # r = -math.fabs(s - Q)-(Pfan+Ppump)/3-Punish_coefficient
                Punish_coefficient_1 = 300 - 300 * math.exp(-((Q - s) ** 2) / (2 * (s * 0.1) ** 2))
                r = -Punish_coefficient_1 - (Pfan + Ppump) - Punish_coefficient
                # print(s_normalized.squeeze(), a, r.squeeze() / 1500, s__normalized.squeeze())
                ddpg_base.store_transition(s_normalized.squeeze(), a, r.squeeze() / 1500,
                                      s__normalized.squeeze())  # store the transition to memory
                # print(s_normalized, a, r / 5000, s__normalized)
                if ddpg_base.pointer > MEMORY_CAPACITY:

                    if var_base_0 > 0.1:
                        var_base_0 *= 0.9995  # decay the exploration controller factor
                        var_base_1 *= 0.9995
                    else:
                        var_base_0 = 0.1  # decay the exploration controller factor
                        var_base_1 = 0.1

                    ddpg_base.learn()
            # print('Ep: ', i, ' |', 'r:', r)
            # print(a_water, a_air)
            worksheet_reward.write(i, i_episode, label=float(r))
            worksheet_a_water.write(i, i_episode, label=a_water)
            worksheet_a_air.write(i, i_episode, label=a_air)
            worksheet_Pfan.write(i, i_episode, label=Pfan)
            worksheet_Ppump.write(i, i_episode, label=Ppump)
            worksheet_Q.write(i, i_episode, label=Q)
            worksheet_PfandPp.write(i, i_episode, label=Ppump + Pfan)
            worksheet_ccd_Q.write(i, i_episode, label=float(math.fabs(Q - s) / s))
            total_reward = total_reward + float(r)
            total_power = total_power + Ppump + Pfan
    worksheet_reward.write(4392, i_episode, label=total_reward)
    worksheet_PfandPp.write(4392, i_episode, label=total_power)
    total_reward = 0
    total_power = 0
workbook.save('FCU_DDPG_PK_1.xls')



